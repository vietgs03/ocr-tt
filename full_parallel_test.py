from parallel_ocr_test import parallel_batch_ocr

print("🚀 Processing ALL 12 images with 4 parallel workers!\n")
print("⏰ Estimated time:")
print("   - All cached: ~1 minute")
print("   - Mix cached/new: ~3-5 minutes")
print("   - All new: ~10-12 minutes (vs 34 minutes sequential!)\n")

results = parallel_batch_ocr('test_images', max_workers=4)
