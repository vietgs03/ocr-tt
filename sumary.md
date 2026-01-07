# 🚀 Hệ thống OCR Tiếng Việt Tự Học & Tối Ưu trên CPU

Hệ thống này được thiết kế để:
1.  **OCR cực nhanh**: Sử dụng PaddleOCR (bản nhẹ) để trích xuất văn bản từ ảnh/PDF.
2.  **Tự sửa lỗi (Auto-Correction)**: Sử dụng thuật toán SymSpell + Từ điển tần suất (`vn_dictionary.txt`) để sửa lỗi dấu tiếng Việt ngay lập tức (In-memory, không cần LLM nặng).
3.  **Tự học (Self-learning)**: Có khả năng quét thư mục ảnh mẫu, tự động học từ mới và độ tin cậy cao để bổ sung vào từ điển.
4.  **Tối ưu cho VPS CPU**: Chạy mượt mà trên VPS không có GPU (1-2 Core, 2GB RAM là đủ).

---

## 📂 Cấu trúc dự án

```
/ocr_system
├── 📜 main_ocr.py             # Script chính: Chạy OCR + Sửa lỗi + Streaming kết quả
├── 📜 hybrid_ocr_corrector.py # Engine lõi: Kết hợp PaddleOCR & SymSpell
├── 📜 trainer.py              # Script "Spam Training": Quét ảnh -> Học từ -> Update từ điển
├── 📜 crawler.py              # Script tải dữ liệu mẫu từ internet (Cần cấu hình thêm nguồn)
├── 📚 vn_dictionary.txt       # "Bộ não" của hệ thống (Từ điển tần suất)
├── ⚙️ correction_map.json     # Map sửa lỗi cứng (Hard-coded fixes) cho các lỗi đặc thù
└── 📁 data
    └── 📁 train_images        # Nơi chứa ảnh để chạy trainer.py
```

---

## 🛠️ Cài đặt (Setup)

Chạy trên môi trường Linux (Ubuntu/Debian) hoặc Windows đều được.

**1. Cài đặt thư viện:**
```bash
pip install paddlepaddle paddleocr symspellpy opencv-python-headless rapidfuzz requests beautifulsoup4
```

**2. Cấu hình PaddleOCR (Lần đầu):**
Lần đầu chạy, hệ thống sẽ tự tải model OCR nhẹ (~15MB) về thư mục `~/.paddleocr`.

---

## 🚀 Hướng dẫn sử dụng

### 1. Chạy OCR văn bản (Production)
Sử dụng script chính để xử lý ảnh và nhận kết quả sạch ngay lập tức:

```bash
python3 hybrid_ocr_corrector.py
# (Mặc định đang trỏ tới file ảnh mẫu, hãy sửa code để trỏ tới file bạn cần)
```

**Luồng xử lý:** `Ảnh` -> `PaddleOCR` -> `SymSpell Correction` -> `Kết quả Text`

### 2. Chế độ "Spam Training" (Làm giàu từ điển)
Để hệ thống thông minh hơn, hãy ném hàng trăm/hàng nghìn ảnh văn bản hành chính vào thư mục `data/train_images`, sau đó chạy:

```bash
python3 trainer.py
```

**Cơ chế:**
- Hệ thống sẽ quét toàn bộ ảnh.
- Lọc ra những từ PaddleOCR nhận diện với độ tin cậy > 95%.
- Tự động thêm/cập nhật tần suất từ đó vào `vn_dictionary.txt`.
- Lần sau chạy OCR, những từ này sẽ được ưu tiên sửa đúng.

### 3. Thu thập dữ liệu (Crawler)
Sử dụng script crawler để tải ảnh mẫu về train:

```bash
python3 crawler.py
```

---

## 💡 Mẹo tối ưu (Pro Tips)

1.  **Từ điển là chìa khóa**: File `vn_dictionary.txt` càng chuẩn, OCR càng chính xác. Bạn có thể mở file này ra và sửa tay các từ quan trọng (tăng số tần suất lên, ví dụ `nghiệm thu 9999999`) để ép hệ thống luôn chọn từ đó.
2.  **Xử lý PDF**: Nếu nguồn dữ liệu là PDF, hãy dùng `pdf2image` để chuyển thành ảnh trước khi đưa vào `data/train_images`.
3.  **Tiling**: Với ảnh dài (biên bản nhiều trang), `hybrid_ocr_corrector.py` đã tích hợp sẵn chế độ cắt lớp (Tiling) để xử lý mượt mà không lo tràn RAM.

