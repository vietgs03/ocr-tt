import time
import pytesseract
from PIL import Image
from paddleocr import PaddleOCR
import ollama
import os

def benchmark_tesseract(image_path):
    """Tesseract OCR - Fastest"""
    start = time.time()
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='vie')
    duration = time.time() - start
    return text, duration

def benchmark_paddle(image_path):
    """PaddleOCR - Balanced"""
    start = time.time()
    ocr = PaddleOCR(use_angle_cls=False, lang='vi', use_gpu=False, show_log=False)
    result = ocr.ocr(image_path, cls=False)
    text = '\n'.join([line[1][0] for line in result[0]])
    duration = time.time() - start
    return text, duration

def benchmark_deepseek(image_path):
    """DeepSeek OCR - Most Accurate"""
    start = time.time()
    client = ollama.Client()
    
    with open(image_path, 'rb') as f:
        img_data = f.read()
    
    response = client.chat(
        model='deepseek-ocr',
        messages=[{'role': 'user', 'content': 'Free OCR.', 'images': [img_data]}],
        options={'temperature': 0.0}
    )
    
    text = response['message']['content']
    
    # Parse grounding tags
    import re
    pattern = r'<\|ref\|\>(.*?)<\|/ref\|\>\s*<\|det\|\>(.*?)<\|/det\|\>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        text = '\n'.join([match[0] for match in matches])
    
    duration = time.time() - start
    return text, duration

if __name__ == "__main__":
    test_img = "unnamed.jpg"
    
    if not os.path.exists(test_img):
        print(f"❌ Test image not found: {test_img}")
        exit(1)
    
    print("="*70)
    print("🏆 3-TIER OCR ENGINE BENCHMARK")
    print("="*70)
    print(f"Test image: {test_img}\n")
    
    results = {}
    
    # TIER 1: Tesseract
    print("⚡ TIER 1: Tesseract OCR (Lightning Fast)")
    print("-"*70)
    try:
        text, duration = benchmark_tesseract(test_img)
        results['tesseract'] = {'text': text, 'duration': duration}
        print(f"✅ Time: {duration:.2f}s")
        print(f"📏 Length: {len(text)} chars")
        print(f"Preview: {text[:200]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['tesseract'] = {'text': '', 'duration': 0}
    print()
    
    # TIER 2: PaddleOCR
    print("⚡⚡ TIER 2: PaddleOCR (Balanced)")
    print("-"*70)
    try:
        text, duration = benchmark_paddle(test_img)
        results['paddle'] = {'text': text, 'duration': duration}
        print(f"✅ Time: {duration:.2f}s")
        print(f"📏 Length: {len(text)} chars")
        print(f"Preview: {text[:200]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['paddle'] = {'text': '', 'duration': 0}
    print()
    
    # TIER 3: DeepSeek (skip để test nhanh)
    print("🎯 TIER 3: DeepSeek OCR (Most Accurate)")
    print("-"*70)
    print("⏸️  Skipped (takes ~170s, already benchmarked)")
    print("   Use chunked version for ~60s processing")
    print()
    
    # Summary
    print("="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    
    if results['tesseract']['duration'] > 0:
        speedup_t_vs_p = results['paddle']['duration'] / results['tesseract']['duration']
        print(f"Tesseract:  {results['tesseract']['duration']:.2f}s")
        print(f"PaddleOCR:  {results['paddle']['duration']:.2f}s ({speedup_t_vs_p:.1f}x slower)")
        print(f"DeepSeek:   ~170s (chunked: ~60s)")
        print()
        print(f"🚀 Tesseract is {speedup_t_vs_p:.0f}x faster than PaddleOCR!")
        print(f"⚡ Tesseract is ~{170 / results['tesseract']['duration']:.0f}x faster than DeepSeek!")
    
    print()
    print("💡 RECOMMENDED USAGE:")
    print("   - Draft/Preview: Tesseract (instant)")
    print("   - Daily work: PaddleOCR (fast + accurate)")
    print("   - Critical docs: DeepSeek chunked (best quality)")
