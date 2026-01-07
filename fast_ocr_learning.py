import json
import os
import time
from paddleocr import PaddleOCR
import difflib

class FastOCRLearning:
    def __init__(self, map_file='correction_map.json'):
        self.map_file = map_file
        self.correction_map = self._load_map()
        # Initialize PaddleOCR (Use lightweight model if possible for speed)
        # lang='vi', use_angle_cls=True
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False) 

    def _load_map(self):
        if os.path.exists(self.map_file):
            with open(self.map_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_map(self):
        with open(self.map_file, 'w', encoding='utf-8') as f:
            json.dump(self.correction_map, f, ensure_ascii=False, indent=4)

    def learn_new_correction(self, wrong_word, correct_word):
        """Hàm để training/update từ điển thủ công sau này"""
        self.correction_map[wrong_word.lower()] = correct_word
        self._save_map()
        print(f"Learned: '{wrong_word}' -> '{correct_word}'")

    def apply_correction(self, text):
        """
        Sửa lỗi dựa trên từ điển đã học.
        Có thể nâng cấp dùng Fuzzy Matching để bắt các từ gần giống.
        """
        text_lower = text.lower()
        
        # 1. Direct Match (Nhanh nhất)
        if text_lower in self.correction_map:
            return self.correction_map[text_lower]
        
        # 2. Substring Replace (Thay thế cụm từ trong câu)
        # Sắp xếp key theo độ dài giảm dần để ưu tiên cụm từ dài trước
        sorted_keys = sorted(self.correction_map.keys(), key=len, reverse=True)
        corrected_text = text
        for key in sorted_keys:
            if key in corrected_text.lower():
                # Thay thế thông minh (case insensitive replacement logic can be complex, simple replace for now)
                # Đây là bản đơn giản, bản promax cần regex để giữ case
                corrected_text = corrected_text.replace(key, self.correction_map[key])
                corrected_text = corrected_text.replace(key.upper(), self.correction_map[key].upper())
                corrected_text = corrected_text.replace(key.title(), self.correction_map[key].title())
        
        return corrected_text

    def process_image(self, img_path):
        if not os.path.exists(img_path):
            print(f"File {img_path} not found.")
            return

        print(f"Processing {img_path}...")
        start_time = time.time()

        # 1. Run OCR
        result = self.ocr.ocr(img_path, cls=True)
        
        if result is None or result[0] is None:
            print("No text detected.")
            return

        blocks = []
        for line in result[0]:
            text_content = line[1][0]
            score = line[1][1]
            box = line[0]
            
            # 2. Apply Instant Correction (Mapping)
            corrected_text = self.apply_correction(text_content)
            
            blocks.append({
                "text": corrected_text,
                "original_text": text_content, # Giữ lại để đối chiếu/training
                "box": box,
                "confidence": score
            })

        # 3. Smart Sort (Logic sắp xếp tọa độ)
        sorted_lines = self._smart_sort(blocks)
        
        # 4. Format Output
        final_text = self._format_layout(sorted_lines)
        
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")
        
        return final_text

    def _smart_sort(self, blocks):
        # Sắp xếp theo Y
        blocks.sort(key=lambda b: b['box'][0][1])
        
        sorted_lines = []
        current_line = []
        current_y = -1
        y_threshold = 15 
        
        for block in blocks:
            y = block['box'][0][1]
            if current_y == -1 or abs(y - current_y) <= y_threshold:
                current_line.append(block)
                if current_y == -1: current_y = y
            else:
                current_line.sort(key=lambda b: b['box'][0][0])
                sorted_lines.append(current_line)
                current_line = [block]
                current_y = y
                
        if current_line:
            current_line.sort(key=lambda b: b['box'][0][0])
            sorted_lines.append(current_line)
        return sorted_lines

    def _format_layout(self, sorted_lines):
        output = ""
        for line in sorted_lines:
            line_str = ""
            last_x_end = 0
            for block in line:
                text = block['text']
                x_start = block['box'][0][0]
                
                if last_x_end == 0:
                    # Căn lề trái giả lập
                    if x_start > 300: line_str += "\t\t\t"
                    elif x_start > 100: line_str += "\t"
                else:
                    gap = x_start - last_x_end
                    if gap > 40: line_str += "\t"
                    else: line_str += " "
                
                line_str += text
                last_x_end = block['box'][1][0]
            output += line_str.strip() + "\n"
        return output

if __name__ == "__main__":
    # Khởi tạo engine
    engine = FastOCRLearning()
    
    # Chạy thử trên file mới
    img_path = 'bbnghiemthucongtrinh.jpg'
    result = engine.process_image(img_path)
    
    if result:
        print("\n--- FAST OCR RESULT (with Learning) ---")
        print(result)
        
        with open('fast_ocr_result.txt', 'w', encoding='utf-8') as f:
            f.write(result)
        print("\nSaved to fast_ocr_result.txt")

