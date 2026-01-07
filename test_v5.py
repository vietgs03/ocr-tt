from paddleocr import PaddleOCR

try:
    print("Attempting to load PP-OCRv5...")
    ocr = PaddleOCR(use_angle_cls=False, lang='vi', ocr_version='PP-OCRv5')
    print("✅ PP-OCRv5 loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load PP-OCRv5: {e}")
