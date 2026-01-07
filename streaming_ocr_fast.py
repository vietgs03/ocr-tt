import json
import os
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re

class StreamingOCR:
    def __init__(self, map_file='correction_map.json'):
        self.map_file = map_file
        self.correction_map = self._load_map()
        print("⚡ Initializing PaddleOCR Engine...")
        # show_log=False để log sạch sẽ hơn
        self.ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)
        print("✅ Engine Ready!")

    def _load_map(self):
        if os.path.exists(self.map_file):
            with open(self.map_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def fuzzy_correct(self, text):
        """
        Sửa lỗi dùng Dictionary Mapping trong bộ nhớ.
        """
        text_lower = text.lower()
        
        # 1. Direct Match
        if text_lower in self.correction_map:
            return self.correction_map[text_lower]

        corrected_text = text
        # 2. Substring Replace
        # Loop qua từ điển để replace. 
        # Cần sort key theo độ dài để replace từ dài trước (tránh lỗi replace chồng chéo)
        sorted_keys = sorted(self.correction_map.keys(), key=len, reverse=True)
        
        for wrong in sorted_keys:
            if wrong in corrected_text.lower():
                correct = self.correction_map[wrong]
                # Regex replace case-insensitive
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                corrected_text = pattern.sub(correct, corrected_text)
                
        return corrected_text

    def process_stream_tiled(self, img_path, tile_height=1000, overlap=100):
        """
        Cắt ảnh thành từng phần (tile) và xử lý streaming từng phần.
        Giúp trả về kết quả ngay lập tức cho ảnh dài.
        """
        if not os.path.exists(img_path):
            yield f"Error: File {img_path} not found."
            return

        yield f"🚀 Start processing {img_path} (Tiled Streaming)...\n"
        start_time = time.time()
        
        # Đọc ảnh bằng OpenCV
        img = cv2.imread(img_path)
        if img is None:
            yield "Error: Unable to read image."
            return
            
        h, w, _ = img.shape
        yield f"📏 Image Size: {w}x{h}\n"

        # Chia nhỏ ảnh và xử lý từng phần
        current_y = 0
        tile_idx = 0
        
        while current_y < h:
            tile_idx += 1
            # Tính toán vùng cắt
            y_end = min(current_y + tile_height, h)
            # Thêm overlap để tránh cắt đôi chữ ở biên, trừ tile đầu tiên
            y_start = max(0, current_y - overlap) if current_y > 0 else 0
            
            # Cắt tile
            tile_img = img[y_start:y_end, 0:w]
            
            # yield f"   Processing Tile {tile_idx} (Y: {y_start}-{y_end})..."
            
            # OCR trên tile này
            result = self.ocr.ocr(tile_img, cls=True)
            
            if result and result[0]:
                # Sắp xếp và in ra ngay
                blocks = result[0]
                blocks.sort(key=lambda x: x[0][1]) # Sort theo Y
                
                line_buffer = []
                curr_line_y = -1
                
                for line in blocks:
                    text_content = line[1][0]
                    box = line[0]
                    # Tọa độ Y cục bộ trong tile
                    local_y = box[0][1]
                    # Tọa độ Y toàn cục
                    global_y = local_y + y_start
                    
                    # Nếu text nằm trong vùng overlap phía trên (đã xử lý ở tile trước), bỏ qua
                    # Để tránh in trùng lặp
                    if current_y > 0 and local_y < overlap:
                        continue
                        
                    # Logic gom dòng
                    if curr_line_y != -1 and abs(local_y - curr_line_y) > 15:
                        yield self._process_line_buffer(line_buffer)
                        line_buffer = []
                    
                    line_buffer.append(text_content)
                    if curr_line_y == -1: curr_line_y = local_y
                    else: curr_line_y = local_y

                if line_buffer:
                    yield self._process_line_buffer(line_buffer)

            # Cập nhật Y cho vòng lặp sau
            current_y += tile_height - overlap if current_y + tile_height < h else tile_height

        end_time = time.time()
        yield f"\n✅ Done in {end_time - start_time:.2f}s total."

    def _process_line_buffer(self, buffer):
        raw_line = " ".join(buffer)
        corrected_line = self.fuzzy_correct(raw_line)
        return corrected_line

if __name__ == "__main__":
    streamer = StreamingOCR()
    print("-" * 50)
    
    img = 'bbnghiemthucongtrinh.jpg'
    
    # Sử dụng generator để nhận kết quả ngay khi có
    for chunk in streamer.process_stream_tiled(img, tile_height=800, overlap=50):
        print(chunk)

