import os
import time
import cv2
import json
import logging
from paddleocr import PaddleOCR
from symspellpy import SymSpell
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LocalOCR:
    def __init__(self, dictionary_path='vn_dictionary.txt', use_ollama=False):
        self.use_ollama = use_ollama
        
        logging.info("🚀 Initializing Local OCR Pro Max...")
        start_init = time.time()
        
        # 1. Load OCR Engine (In-Memory)
        # use_angle_cls=False for speed
        # enable_mkldnn=True for CPU Optimization
        self.ocr = PaddleOCR(use_angle_cls=False, lang='vi', use_gpu=False, enable_mkldnn=True, show_log=False, ocr_version='PP-OCRv4')
        
        # 2. Load Dictionary (In-Memory)
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        if os.path.exists(dictionary_path):
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, separator=" ", encoding="utf-8")
            logging.info(f"✅ Dictionary loaded: {dictionary_path}")
        else:
            logging.warning("⚠️ Dictionary not found. Corrections might be weak.")
            
        logging.info(f"⚡ Initialization complete in {time.time() - start_init:.2f}s")

    def _ocr_raw(self, img_path):
        """Standard raw OCR extraction"""
        start = time.time()
        result = self.ocr.ocr(img_path, cls=False)
        if not result or not result[0]:
            return []
        
        # Sort top-to-bottom
        lines = sorted(result[0], key=lambda x: x[0][1])
        text_lines = [line[1][0] for line in lines]
        logging.info(f"📷 OCR Raw Processing time: {time.time() - start:.2f}s")
        return text_lines

    def process_fast(self, img_path):
        """
        Hot Path: OCR + SymSpell (Speed Focus)
        Best for: Real-time scanning, simple docs
        """
        raw_lines = self._ocr_raw(img_path)
        
        corrected_lines = []
        for line in raw_lines:
            suggestions = self.sym_spell.lookup_compound(line, max_edit_distance=2)
            if suggestions:
                corrected_lines.append(suggestions[0].term)
            else:
                corrected_lines.append(line)
        
        return "\n".join(corrected_lines)

    def process_smart(self, img_path, model="llama3"):
        """
        Cold Path: OCR + Ollama (Quality Focus)
        Best for: Complex extraction, formatting (JSON/Markdown)
        """
        if not HAS_OLLAMA:
            return "❌ Error: Ollama not installed or library missing."
            
        raw_text = "\n".join(self._ocr_raw(img_path))
        
        prompt = f"""
        Nhiệm vụ: Chuyển đổi văn bản OCR lỗi thành văn bản chuẩn xác.
        Yêu cầu bắt buộc:
        1. CHỈ TRẢ VỀ văn bản đã sửa. KHÔNG giải thích, KHÔNG chào hỏi, KHÔNG chèn thêm hướng dẫn.
        2. Dữ liệu đầu vào là văn bản hành chính Việt Nam bị lỗi OCR. Hãy đoán và sửa lại dấu tiếng Việt, tên riêng, và lỗi máy đánh chữ.
        3. Giữ nguyên cấu trúc xuống dòng.

        Văn bản OCR lỗi:
        ```
        {raw_text}
        ```

        Văn bản đã sửa (Markdown):
        """
        
        logging.info(f"🧠 Sending to Ollama ({model})...")
        start_llm = time.time()
        try:
            response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
            logging.info(f"🤖 LLM Processing time: {time.time() - start_llm:.2f}s")
            return response['message']['content']
        except Exception as e:
            return f"❌ Ollama Error: {e}"

if __name__ == "__main__":
    # Test
    engine = LocalOCR()
    
    test_img = 'unnamed.jpg'
    if os.path.exists(test_img):
        print("\n--- ⚡ FAST MODE (SymSpell) ---")
        start = time.time()
        print(engine.process_fast(test_img))
        print(f"\n⏱️ Total Fast Time: {time.time() - start:.2f}s")
        
        # Uncomment to test Smart Mode once Ollama is ready
        print("\n--- 🧠 SMART MODE (Ollama) ---")
        print(engine.process_smart(test_img, model="llama2:7b"))
