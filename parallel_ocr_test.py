from selflearning_ocr import SelfLearningOCR
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def process_single_image(image_path):
    """Worker function for parallel processing"""
    # Each process needs its own OCR instance
    ocr = SelfLearningOCR(keep_alive="60m")
    
    start = time.time()
    text = ocr.process_image(image_path)
    duration = time.time() - start
    
    return {
        'file': os.path.basename(image_path),
        'path': image_path,
        'text': text,
        'duration': duration
    }

def parallel_batch_ocr(image_folder="test_images", max_workers=None):
    """
    Parallel batch OCR processing
    
    Args:
        image_folder: Folder containing images
        max_workers: Number of parallel workers (default: CPU cores)
    """
    # Get all images
    image_files = sorted(glob.glob(f"{image_folder}/*.png"))
    
    if not image_files:
        print(f"❌ No images found in {image_folder}/")
        return
    
    # Determine number of workers
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 core for system
    
    print("="*60)
    print("🚀 PARALLEL BATCH OCR")
    print("="*60)
    print(f"📚 Total images: {len(image_files)}")
    print(f"⚡ Parallel workers: {max_workers}")
    print(f"🖥️  CPU cores: {multiprocessing.cpu_count()}")
    print("="*60 + "\n")
    
    # Create output folder
    os.makedirs("ocr_results_parallel", exist_ok=True)
    
    # Process in parallel
    total_start = time.time()
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, img): img 
            for img in image_files
        }
        
        # Process as they complete
        for future in as_completed(future_to_image):
            img_path = future_to_image[future]
            try:
                result = future.result()
                completed += 1
                
                # Save result
                output_file = os.path.join(
                    "ocr_results_parallel",
                    os.path.basename(result['path']).replace('.png', '.txt')
                )
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                
                results.append(result)
                
                print(f"[{completed}/{len(image_files)}] ✅ {result['file']}")
                print(f"    ⏱️  Time: {result['duration']:.2f}s")
                print(f"    💾 Saved: {output_file}\n")
                
            except Exception as e:
                print(f"[{completed}/{len(image_files)}] ❌ {os.path.basename(img_path)}")
                print(f"    Error: {e}\n")
    
    total_duration = time.time() - total_start
    
    # Summary
    print("="*60)
    print("📊 PARALLEL BATCH SUMMARY")
    print("="*60)
    print(f"✅ Total images: {len(results)}")
    print(f"⏱️  Total time: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")
    print(f"📈 Average time: {total_duration/len(results):.2f}s per image")
    print(f"⚡ Speedup vs sequential: {(len(results) * 170) / total_duration:.1f}x")
    print(f"📁 Results saved to: ocr_results_parallel/")
    print("="*60)
    
    return results

if __name__ == "__main__":
    print("\n🔍 Testing with 4 images first (to avoid long wait)...\n")
    
    # Test with subset first
    import shutil
    
    # Create test subset
    test_folder = "test_subset"
    os.makedirs(test_folder, exist_ok=True)
    
    all_images = sorted(glob.glob("test_images/*.png"))[:4]  # First 4 images
    
    for img in all_images:
        shutil.copy(img, test_folder)
    
    print(f"📋 Testing with {len(all_images)} images")
    print(f"   Workers will run in parallel, reducing total time!\n")
    
    # Run parallel processing
    results = parallel_batch_ocr(test_folder, max_workers=2)
    
    # Cleanup
    shutil.rmtree(test_folder)
    
    print("\n💡 To process all 12 images:")
    print("   parallel_batch_ocr('test_images', max_workers=4)")
