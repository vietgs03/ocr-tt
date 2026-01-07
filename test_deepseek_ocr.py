import os
import sys
# Ensure we can import from local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from selflearning_ocr import SelfLearningOCR

def test_ocr():
    image_path = "bbnghiemthucongtrinh.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found in {os.getcwd()}")
        return

    print(f"🚀 Initializing DeepSeek OCR...")
    ocr = SelfLearningOCR(keep_alive="5m")
    
    print(f"📸 Processing Image: {image_path}")
    print("⏳ Please wait, this might take a moment...")
    
    # We set use_cache=False to force a fresh inference for testing
    result = ocr.process_image(image_path, use_cache=False)
    
    print("\n" + "="*50)
    print("📝 OCR RESULT")
    print("="*50)
    print(result)
    print("="*50)

if __name__ == "__main__":
    test_ocr()
