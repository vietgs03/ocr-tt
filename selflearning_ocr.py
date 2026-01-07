import os
import time
import ollama
import logging
import hashlib
import json
import sqlite3
from PIL import Image
import io
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class SelfLearningOCR:
    """
    Self-Learning OCR với:
    1. Cache Layer: Lưu kết quả OCR -> Trả về instant nếu gặp lại ảnh giống
    2. Vocabulary Learning: Học từ vựng từ user corrections
    3. Context Database: Lưu examples để improve accuracy qua prompting
    """
    
    def __init__(self, 
                 model_name="deepseek-ocr",
                 cache_db="ocr_cache.db",
                 vocab_file="learned_vocabulary.json",
                 keep_alive="60m"):
        
        self.client = ollama.Client(host='http://127.0.0.1:11434')
        self.model_name = model_name
        self.keep_alive = keep_alive
        self.cache_db = cache_db
        self.vocab_file = vocab_file
        
        # Initialize components
        self._init_cache_db()
        self._load_vocabulary()
        self._preload_model()
        
        logging.info("🚀 Self-Learning OCR Ready!")
        logging.info(f"📦 Cache: {self.cache_db}")
        logging.info(f"📚 Vocabulary: {len(self.vocabulary)} terms")
    
    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        self.conn = sqlite3.connect(self.cache_db)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_cache (
                image_hash TEXT PRIMARY KEY,
                image_path TEXT,
                ocr_result TEXT,
                confidence REAL DEFAULT 1.0,
                timestamp INTEGER,
                usage_count INTEGER DEFAULT 1
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_type TEXT,
                example_text TEXT,
                timestamp INTEGER
            )
        ''')
        self.conn.commit()
        logging.info("✅ Cache database initialized")
    
    def _load_vocabulary(self):
        """Load learned vocabulary"""
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                self.vocabulary = json.load(f)
        else:
            self.vocabulary = {}
        logging.info(f"📖 Vocabulary loaded: {len(self.vocabulary)} entries")
    
    def _save_vocabulary(self):
        """Save vocabulary to disk"""
        with open(self.vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=2)
    
    def _preload_model(self):
        """Preload model into RAM"""
        try:
            logging.info("🔄 Preloading model...")
            self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'warmup'}],
                keep_alive=self.keep_alive
            )
            logging.info("✅ Model ready in memory")
        except Exception as e:
            logging.warning(f"⚠️ Preload failed: {e}")
    
    def _compute_image_hash(self, image_path):
        """Compute perceptual hash of image for cache lookup"""
        try:
            img = Image.open(image_path).convert('L').resize((32, 32))
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            bits = ''.join('1' if p > avg else '0' for p in pixels)
            return hashlib.md5(bits.encode()).hexdigest()
        except Exception as e:
            logging.warning(f"⚠️ Hash failed: {e}")
            # Fallback to file hash
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
    
    def _check_cache(self, image_hash):
        """Check if result exists in cache"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT ocr_result, usage_count FROM ocr_cache WHERE image_hash = ?',
            (image_hash,)
        )
        result = cursor.fetchone()
        if result:
            # Update usage count
            cursor.execute(
                'UPDATE ocr_cache SET usage_count = usage_count + 1 WHERE image_hash = ?',
                (image_hash,)
            )
            self.conn.commit()
            return result[0]
        return None
    
    def _save_to_cache(self, image_hash, image_path, ocr_result):
        """Save OCR result to cache"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ocr_cache 
            (image_hash, image_path, ocr_result, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (image_hash, image_path, ocr_result, int(time.time())))
        self.conn.commit()
    
    def _apply_vocabulary_corrections(self, text):
        """Apply learned vocabulary corrections"""
        corrected = text
        for wrong, correct in self.vocabulary.items():
            corrected = corrected.replace(wrong, correct)
        return corrected
    
    def parse_grounding_output(self, raw_output):
        """Parse grounding tags"""
        import re
        pattern = r'<\|ref\|\>(.*?)<\|/ref\|\>\s*<\|det\|\>(.*?)<\|/det\|\>'
        matches = re.findall(pattern, raw_output, re.DOTALL)
        clean_text = '\n'.join([match[0] for match in matches])
        return clean_text if clean_text else raw_output
    
    def process_image(self, image_path, use_cache=True, prompt="Free OCR."):
        """
        Process image với self-learning:
        1. Check cache trước (instant response)
        2. Nếu miss cache → chạy OCR
        3. Apply vocabulary corrections
        4. Save to cache
        """
        if not os.path.exists(image_path):
            return f"❌ File not found: {image_path}"
        
        start_time = time.time()
        
        # Step 1: Check cache
        image_hash = self._compute_image_hash(image_path)
        
        if use_cache:
            cached = self._check_cache(image_hash)
            if cached:
                logging.info(f"⚡ CACHE HIT! Instant response in {time.time()-start_time:.3f}s")
                return cached
        
        # Step 2: OCR Processing
        logging.info(f"📸 Processing: {image_path}")
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
            
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_data]
                }],
                options={'temperature': 0.0},
                keep_alive=self.keep_alive
            )
            
            ocr_result = response['message']['content']
            ocr_result = self.parse_grounding_output(ocr_result)
            
            # Step 3: Apply vocabulary corrections
            corrected_result = self._apply_vocabulary_corrections(ocr_result)
            
            # Step 4: Save to cache
            self._save_to_cache(image_hash, image_path, corrected_result)
            
            duration = time.time() - start_time
            logging.info(f"✅ Completed in {duration:.2f}s (saved to cache)")
            
            return corrected_result
            
        except Exception as e:
            logging.error(f"❌ Error: {e}")
            return str(e)
    
    def learn_correction(self, wrong_text, correct_text):
        """
        Học từ user correction.
        User chỉnh sửa kết quả OCR → System học để lần sau tự động sửa
        """
        if wrong_text != correct_text:
            self.vocabulary[wrong_text] = correct_text
            self._save_vocabulary()
            logging.info(f"📚 Learned: '{wrong_text}' → '{correct_text}'")
    
    def get_cache_stats(self):
        """Thống kê cache performance"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*), SUM(usage_count) FROM ocr_cache')
        count, total_hits = cursor.fetchone()
        return {
            'cached_documents': count or 0,
            'total_cache_hits': total_hits or 0,
            'vocabulary_size': len(self.vocabulary)
        }
    
    def clear_cache(self):
        """Xóa cache (khi cần reset)"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM ocr_cache')
        self.conn.commit()
        logging.info("🗑️ Cache cleared")

if __name__ == "__main__":
    # Initialize self-learning OCR
    ocr = SelfLearningOCR(keep_alive="60m")
    
    # Test 1: First time (slow)
    print("\n" + "="*60)
    print("TEST 1: First OCR (will be cached)")
    print("="*60)
    result1 = ocr.process_image("unnamed.jpg")
    print(result1[:200] + "...")
    
    # Test 2: Second time - same image (instant!)
    print("\n" + "="*60)
    print("TEST 2: Same image (should hit cache)")
    print("="*60)
    result2 = ocr.process_image("unnamed.jpg")
    print(result2[:200] + "...")
    
    # Test 3: Learn from correction
    print("\n" + "="*60)
    print("TEST 3: Learning from user correction")
    print("="*60)
    ocr.learn_correction("djnh", "định")
    ocr.learn_correction("vOn", "vốn")
    
    # Stats
    print("\n" + "="*60)
    print("CACHE STATISTICS")
    print("="*60)
    stats = ocr.get_cache_stats()
    print(f"📦 Cached documents: {stats['cached_documents']}")
    print(f"⚡ Total cache hits: {stats['total_cache_hits']}")
    print(f"📚 Vocabulary size: {stats['vocabulary_size']}")
