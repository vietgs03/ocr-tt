import os
from hybrid_ocr_corrector import HybridOCR
from collections import Counter

class AutoTrainer:
    def __init__(self):
        self.engine = HybridOCR(dictionary_path='vn_dictionary.txt')
        self.new_words = Counter()

    def train_from_folder(self, folder_path="data/train_images"):
        if not os.path.exists(folder_path):
            print("Folder not found.")
            return

        print(f"🔄 Scanning folder {folder_path} for training...")
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                print(f"   - Learning from {filename}...")
                
                # Chạy OCR
                # Ở đây ta chỉ cần lấy text thô để phân tích tần suất từ
                # Thay vì gọi process_tiled_stream (trả về string đã sửa), 
                # ta cần can thiệp để lấy text GỐC từ PaddleOCR trước khi sửa
                # Tuy nhiên, để đơn giản, ta dùng kết quả đã sửa để củng cố từ điển
                
                # Logic thông minh: 
                # Nếu PaddleOCR nhận ra 1 từ với confidence > 0.95 -> Coi là từ đúng
                # Thêm từ đó vào từ điển với tần suất +1
                
                img = self.engine.ocr.ocr(img_path, cls=True)
                if img and img[0]:
                    for line in img[0]:
                        text = line[1][0]
                        score = line[1][1]
                        
                        if score > 0.95: # Chỉ học từ những từ model chắc chắn đúng
                            words = text.split()
                            for word in words:
                                # Chỉ học từ có tiếng Việt (bỏ qua số, ký tự lạ)
                                if any(c.isalpha() for c in word):
                                    self.new_words[word.lower()] += 1

        self.update_dictionary()

    def update_dictionary(self):
        print("💾 Updating Dictionary...")
        current_dict = {}
        
        # Đọc từ điển cũ
        if os.path.exists('vn_dictionary.txt'):
            with open('vn_dictionary.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = " ".join(parts[:-1])
                        count = int(parts[-1])
                        current_dict[word] = count
        
        # Merge từ mới
        for word, count in self.new_words.items():
            if word in current_dict:
                current_dict[word] += count
            else:
                current_dict[word] = count # Từ mới
                print(f"   + New word learned: {word}")

        # Lưu lại
        with open('vn_dictionary.txt', 'w', encoding='utf-8') as f:
            for word, count in sorted(current_dict.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{word} {count}\n")
        
        print("✅ Dictionary updated successfully!")

if __name__ == "__main__":
    trainer = AutoTrainer()
    trainer.train_from_folder()

