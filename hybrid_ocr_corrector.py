import os
import time
import cv2
from paddleocr import PaddleOCR
from symspellpy import SymSpell, Verbosity

class HybridOCR:
    def __init__(self, dictionary_path='vn_dictionary.txt'):
        print("⚡ Initializing Engines...")
        
        # 1. OCR Engine
        self.ocr = PaddleOCR(use_angle_cls=False, lang='vi')
        
        # 2. Correction Engine (SymSpell)
        # max_dictionary_edit_distance=2 cho phép sai tối đa 2 ký tự (đủ để sửa dấu)
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        # Load dictionary
        if os.path.exists(dictionary_path):
            # Term index is the column of the term and count index is the column of the term frequency
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, separator=" ", encoding="utf-8")
            print("✅ Dictionary Loaded.")
        else:
            print("⚠️ Warning: Dictionary file not found.")

    def correct_text(self, text):
        """
        Sửa lỗi chính tả sử dụng SymSpell
        """
        # lookup_compound hỗ trợ sửa lỗi trong cả câu dài (chia từ tự động)
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        if suggestions:
            # suggestions[0].term chứa câu đã sửa
            return suggestions[0].term
        return text

    def process_tiled_stream(self, img_path, tile_height=1000, overlap=100):
        if not os.path.exists(img_path):
            yield f"Error: {img_path} not found."
            return

        yield f"🚀 Processing {img_path} with Hybrid Correction...\n"
        start_time = time.time()
        
        img = cv2.imread(img_path)
        if img is None:
            yield "Error reading image."
            return
            
        h, w, _ = img.shape
        
        current_y = 0
        
        while current_y < h:
            y_end = min(current_y + tile_height, h)
            y_start = max(0, current_y - overlap) if current_y > 0 else 0
            
            tile_img = img[y_start:y_end, 0:w]
            
            # OCR
            result = self.ocr.ocr(tile_img, cls=True)
            
            if result and result[0]:
                blocks = sorted(result[0], key=lambda x: x[0][1])
                
                line_buffer = []
                curr_line_y = -1
                
                for line in blocks:
                    text_content = line[1][0]
                    box = line[0]
                    local_y = box[0][1]
                    
                    # Skip overlap to avoid duplicates
                    if current_y > 0 and local_y < overlap:
                        continue
                        
                    # Gom dòng
                    if curr_line_y != -1 and abs(local_y - curr_line_y) > 15:
                        yield self._process_line(line_buffer)
                        line_buffer = []
                    
                    line_buffer.append(text_content)
                    curr_line_y = local_y if curr_line_y == -1 else curr_line_y

                if line_buffer:
                    yield self._process_line(line_buffer)

            current_y += tile_height - overlap if current_y + tile_height < h else tile_height

        yield f"\n✅ Done in {time.time() - start_time:.2f}s"

    def _process_line(self, buffer):
        raw_line = " ".join(buffer)
        # Sửa lỗi chính tả ngay lập tức
        corrected_line = self.correct_text(raw_line)
        return corrected_line

if __name__ == "__main__":
    corrector = HybridOCR()
    print("-" * 50)
    for chunk in corrector.process_tiled_stream('bbnghiemthucongtrinh.jpg'):
        print(chunk)

