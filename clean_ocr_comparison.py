#!/usr/bin/env python3
"""
Clean OCR Comparison - NO CACHE
Test cùng 1 file với các engine khác nhau
So sánh quality của text extraction
"""

import pytesseract
from PIL import Image
from paddleocr import PaddleOCR
import ollama
import time
import os
import re

TEST_IMAGE = "unnamed.jpg"
PROMPT = "Free OCR."  # Cùng prompt cho tất cả

def parse_grounding_tags(text):
    """Remove grounding tags"""
    pattern = r'<\|ref\|\>(.*?)<\|/ref\|\>\s*<\|det\|\>(.*?)<\|/det\|\>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return '\n'.join([match[0] for match in matches])
    return text

def test_tesseract(image_path):
    print("\n" + "="*70)
    print("⚡ TESSERACT OCR")
    print("="*70)
    
    start = time.time()
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='vie')
    duration = time.time() - start
    
    print(f"⏱️  Time: {duration:.2f}s")
    print(f"📏 Characters: {len(text)}")
    print(f"📄 Lines: {len(text.splitlines())}")
    print(f"\n--- Full Output ---")
    print(text)
    print("="*70)
    
    return {
        'engine': 'Tesseract',
        'text': text,
        'duration': duration,
        'length': len(text),
        'lines': len(text.splitlines())
    }

def test_paddle(image_path):
    print("\n" + "="*70)
    print("⚡ PADDLEOCR")
    print("="*70)
    
    start = time.time()
    ocr = PaddleOCR(use_angle_cls=False, lang='vi', use_gpu=False, show_log=False)
    result = ocr.ocr(image_path, cls=False)
    
    # Extract text
    lines = [line[1][0] for line in result[0]]
    text = '\n'.join(lines)
    
    duration = time.time() - start
    
    print(f"⏱️  Time: {duration:.2f}s")
    print(f"📏 Characters: {len(text)}")
    print(f"📄 Lines: {len(text.splitlines())}")
    print(f"\n--- Full Output ---")
    print(text)
    print("="*70)
    
    return {
        'engine': 'PaddleOCR',
        'text': text,
        'duration': duration,
        'length': len(text),
        'lines': len(text.splitlines())
    }

def test_deepseek(image_path):
    print("\n" + "="*70)
    print("🎯 DEEPSEEK-OCR")
    print("="*70)
    print("⚠️  This will take ~2-3 minutes on CPU...")
    
    start = time.time()
    client = ollama.Client()
    
    with open(image_path, 'rb') as f:
        img_data = f.read()
    
    response = client.chat(
        model='deepseek-ocr',
        messages=[{
            'role': 'user',
            'content': PROMPT,
            'images': [img_data]
        }],
        options={'temperature': 0.0}
    )
    
    text = response['message']['content']
    text = parse_grounding_tags(text)
    
    duration = time.time() - start
    
    print(f"⏱️  Time: {duration:.2f}s")
    print(f"📏 Characters: {len(text)}")
    print(f"📄 Lines: {len(text.splitlines())}")
    print(f"\n--- Full Output ---")
    print(text)
    print("="*70)
    
    return {
        'engine': 'DeepSeek-OCR',
        'text': text,
        'duration': duration,
        'length': len(text),
        'lines': len(text.splitlines())
    }

def compare_results(results):
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    
    # Table
    print(f"\n{'Engine':<15} {'Time':<10} {'Chars':<10} {'Lines':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['engine']:<15} {r['duration']:<10.2f} {r['length']:<10} {r['lines']:<10}")
    
    # Speed comparison
    print(f"\n⚡ SPEED:")
    fastest = min(results, key=lambda x: x['duration'])
    for r in results:
        speedup = r['duration'] / fastest['duration']
        print(f"   {r['engine']}: {speedup:.1f}x slower than {fastest['engine']}")
    
    # Text length comparison
    print(f"\n📏 TEXT EXTRACTION:")
    longest = max(results, key=lambda x: x['length'])
    for r in results:
        pct = (r['length'] / longest['length']) * 100
        print(f"   {r['engine']}: {r['length']} chars ({pct:.1f}% of {longest['engine']})")
    
    print("\n💡 OBSERVATION:")
    print("   - More text ≠ better (could be noise)")
    print("   - Check actual content for accuracy")
    print("   - DeepSeek should have best Vietnamese diacritics")

if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Test image not found: {TEST_IMAGE}")
        exit(1)
    
    print("="*70)
    print("🧪 CLEAN OCR COMPARISON TEST (NO CACHE)")
    print("="*70)
    print(f"Test file: {TEST_IMAGE}")
    print(f"Prompt: {PROMPT}")
    print(f"\nTesting 3 engines sequentially...")
    
    results = []
    
    # Test 1: Tesseract (fastest)
    results.append(test_tesseract(TEST_IMAGE))
    
    # Test 2: PaddleOCR (balanced)
    results.append(test_paddle(TEST_IMAGE))
    
    # Test 3: DeepSeek (accurate but slow)
    print("\n⏰ Starting DeepSeek test (this will take 2-3 minutes)...")
    results.append(test_deepseek(TEST_IMAGE))
    
    # Compare
    compare_results(results)
    
    # Save results
    with open('ocr_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("OCR COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for r in results:
            f.write(f"\n{'='*70}\n")
            f.write(f"{r['engine']} - {r['duration']:.2f}s\n")
            f.write(f"{'='*70}\n")
            f.write(r['text'])
            f.write("\n\n")
    
    print(f"\n💾 Full results saved to: ocr_comparison.txt")
