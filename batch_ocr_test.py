from selflearning_ocr import SelfLearningOCR
import glob
import os
import time

# Initialize OCR
print("🚀 Initializing Self-Learning OCR...")
ocr = SelfLearningOCR(keep_alive="60m")

# Get all images
image_files = sorted(glob.glob("test_images/*.png"))
print(f"\n📚 Found {len(image_files)} images to process\n")

# Create output folder
os.makedirs("ocr_results", exist_ok=True)

# Process all images
total_start = time.time()
results = []

for i, img_path in enumerate(image_files, 1):
    print(f"{'='*60}")
    print(f"[{i}/{len(image_files)}] {os.path.basename(img_path)}")
    print(f"{'='*60}")
    
    start = time.time()
    text = ocr.process_image(img_path)
    duration = time.time() - start
    
    # Save result
    output_file = os.path.join(
        "ocr_results",
        os.path.basename(img_path).replace('.png', '.txt')
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    results.append({
        'file': os.path.basename(img_path),
        'duration': duration,
        'output': output_file
    })
    
    print(f"⏱️  Time: {duration:.2f}s")
    print(f"💾 Saved: {output_file}\n")

# Summary
total_duration = time.time() - total_start
avg_time = total_duration / len(image_files)

print("="*60)
print("📊 BATCH OCR SUMMARY")
print("="*60)
print(f"✅ Total images: {len(results)}")
print(f"⏱️  Total time: {total_duration:.2f}s")
print(f"📈 Average time: {avg_time:.2f}s per image")
print(f"📁 Results saved to: ocr_results/")

# Cache stats
stats = ocr.get_cache_stats()
print(f"\n💾 CACHE STATISTICS:")
print(f"   Cached documents: {stats['cached_documents']}")
print(f"   Cache hits: {stats['total_cache_hits']}")
print(f"   Vocabulary size: {stats['vocabulary_size']}")

print("\n🎯 To test cache speed, run this script again!")
print("   (Second run will be MUCH faster for duplicate pages)")
