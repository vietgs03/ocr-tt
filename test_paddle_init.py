from paddleocr import PaddleOCR
import logging

# Set logging to DEBUG
logging.basicConfig(level=logging.DEBUG)

print("Attempting to initialize PaddleOCR...")
try:
    # Minimal init
    ocr = PaddleOCR(use_angle_cls=False, lang='vi')
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
