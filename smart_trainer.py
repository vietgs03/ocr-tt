import os
import shutil
import logging
from fast_local_ocr import LocalOCR
from collections import Counter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class SmartTrainer:
    def __init__(self):
        # Initialize the Fast Engine (PaddleOCR + SymSpell)
        self.engine = LocalOCR(dictionary_path='vn_dictionary.txt', use_ollama=False)
        self.new_words = Counter()
        self.trash_folder = "data/trash_images"
        if not os.path.exists(self.trash_folder):
            os.makedirs(self.trash_folder)

    def is_valid_document(self, text):
        """
        Heuristic to check if image is a valid administrative document
        """
        text = text.lower()
        
        # 1. Length check (Documents usually have plenty of text)
        if len(text) < 50:
            return False, "Too short"
            
        # 2. Keyword check
        keywords = [
            "cộng hòa", "xã hội", "chủ nghĩa", "việt nam", "độc lập", "tự do", "hạnh phúc",
            "biên bản", "quyết định", "hợp đồng", "thông báo", "báo cáo", "tờ trình",
            "công ty", "doanh nghiệp", "giám đốc", "nghiệm thu", "thi công"
        ]
        
        # Count how many keywords appear
        keyword_hits = sum(1 for k in keywords if k in text)
        
        if keyword_hits < 1:
            return False, "No administrative keywords found"
            
        return True, "Valid"

    def train_and_clean(self, folder_path="data/train_images"):
        logging.info(f"🔄 Starting Smart Training in {folder_path}...")
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logging.info(f"📂 Found {len(files)} images.")
        
        valid_count = 0
        trash_count = 0
        
        for filename in files:
            img_path = os.path.join(folder_path, filename)
            
            # Use Fast Mode to read text quickly
            # We bypass correct_text to get raw tokens for learning, 
            # but here let's just use the engine's internal OCR
            try:
                # Get raw lines from OCR engine directly to skip SymSpell for validation phase
                # Accessing internal variable _ocr_raw for simplicity as defined in fast_local_ocr.py
                raw_lines = self.engine._ocr_raw(img_path)
                full_text = " ".join(raw_lines)
                
                is_valid, reason = self.is_valid_document(full_text)
                
                if not is_valid:
                    logging.warning(f"🗑️ TRASH: {filename} ({reason}) -> Moving to {self.trash_folder}")
                    shutil.move(img_path, os.path.join(self.trash_folder, filename))
                    trash_count += 1
                    continue
                
                logging.info(f"✅ LEARN: {filename}")
                valid_count += 1
                
                # Logic to add words to dictionary
                # Only trust words with enough length and characters
                for line in raw_lines:
                     words = line.split()
                     for word in words:
                         if len(word) > 1 and any(c.isalpha() for c in word):
                             self.new_words[word.lower()] += 1
                             
            except Exception as e:
                logging.error(f"❌ Error processing {filename}: {e}")
        
        logging.info(f"🏁 Filtering Complete. Valid: {valid_count}, Trash: {trash_count}")
        self.update_dictionary()

    def update_dictionary(self):
        logging.info("💾 Updating Dictionary...")
        current_dict = {}
        
        # Load existing dict
        if os.path.exists('vn_dictionary.txt'):
            with open('vn_dictionary.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = " ".join(parts[:-1])
                        try:
                            count = int(parts[-1])
                            current_dict[word] = count
                        except:
                            pass
        
        # Merge new words (Boosting frequency)
        learned_count = 0
        for word, count in self.new_words.items():
            if count > 1: # Only learn words that appear at least twice in the dataset for noise reduction
                if word in current_dict:
                    current_dict[word] += count * 100 # Boost existing words
                else:
                    current_dict[word] = count * 10 # New words start with moderate frequency
                    learned_count += 1
        
        # Save
        with open('vn_dictionary.txt', 'w', encoding='utf-8') as f:
            for word, count in sorted(current_dict.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{word} {count}\n")
        
        logging.info(f"✅ Dictionary updated! Learned {learned_count} new words.")

if __name__ == "__main__":
    trainer = SmartTrainer()
    trainer.train_and_clean()
