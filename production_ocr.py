import os
import time
import ollama
import logging
from PIL import Image
import io

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ProductionOCR:
    """
    Production-ready OCR optimized for:
    1. Accuracy: Image preprocessing + optimized prompts
    2. Speed: Model keep-alive + memory optimization
    """
    
    def __init__(self, model_name="deepseek-ocr", keep_alive="60m"):
        self.client = ollama.Client(host='http://127.0.0.1:11434')
        self.model_name = model_name
        self.keep_alive = keep_alive  # Keep model in RAM for 60 minutes
        logging.info(f"🚀 Production OCR initialized: {self.model_name}")
        logging.info(f"⚡ Model keep-alive: {self.keep_alive}")
        
        # Preload model into memory
        self._preload_model()
    
    def _preload_model(self):
        """Preload model into RAM to avoid first-request delay"""
        try:
            logging.info("🔄 Preloading model into memory...")
            # Send a dummy request to load model
            self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'warmup'}],
                keep_alive=self.keep_alive
            )
            logging.info("✅ Model preloaded and ready")
        except Exception as e:
            logging.warning(f"⚠️ Preload failed (will load on first use): {e}")
    
    def preprocess_image(self, image_path, target_size=1024, enhance=True):
        """
        Preprocess image for better OCR accuracy:
        1. Resize to optimal size (1024x1024 recommended by DeepSeek)
        2. Optional: Enhance contrast/sharpness
        Returns: bytes of processed image
        """
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate scaling to fit target_size while preserving aspect ratio
            width, height = img.size
            max_dim = max(width, height)
            
            if max_dim > target_size:
                scale = target_size / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Optional enhancement
            if enhance:
                from PIL import ImageEnhance
                # Slight contrast boost helps with faded documents
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                
                # Slight sharpness boost
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return img_byte_arr
            
        except Exception as e:
            logging.warning(f"⚠️ Preprocessing failed, using original: {e}")
            # Fallback to original
            with open(image_path, 'rb') as f:
                return f.read()
    
    def parse_grounding_output(self, raw_output):
        """Parse grounding tags to extract clean text"""
        import re
        pattern = r'<\|ref\|\>(.*?)<\|/ref\|\>\s*<\|det\|\>(.*?)<\|/det\|\>'
        matches = re.findall(pattern, raw_output, re.DOTALL)
        clean_text = '\n'.join([match[0] for match in matches])
        return clean_text if clean_text else raw_output
    
    def process_image(self, 
                     image_path, 
                     prompt="Free OCR.",
                     clean_output=True,
                     preprocess=True,
                     temperature=0.0):
        """
        Process image with optimizations:
        - preprocess: Apply image enhancement (recommended: True)
        - temperature: Lower = more deterministic (0.0 best for OCR)
        - clean_output: Remove grounding tags
        """
        if not os.path.exists(image_path):
            return f"❌ Error: File {image_path} not found"

        logging.info(f"📸 Processing: {image_path}")
        start_time = time.time()
        
        try:
            # Preprocess image if enabled
            if preprocess:
                img_data = self.preprocess_image(image_path)
            else:
                with open(image_path, 'rb') as f:
                    img_data = f.read()
            
            # OCR with optimized parameters
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_data]
                }],
                options={
                    'temperature': temperature,
                    'num_predict': -1,  # No limit on output tokens
                },
                keep_alive=self.keep_alive  # Keep model in memory
            )
            
            duration = time.time() - start_time
            content = response['message']['content']
            
            # Parse grounding tags if requested
            if clean_output:
                content = self.parse_grounding_output(content)
            
            logging.info(f"✅ Completed in {duration:.2f}s")
            return content
            
        except Exception as e:
            logging.error(f"❌ Error: {e}")
            return str(e)
    
    def batch_process(self, image_paths, **kwargs):
        """
        Batch process multiple images efficiently.
        Model stays in memory between requests.
        """
        results = []
        total_start = time.time()
        
        for i, img_path in enumerate(image_paths, 1):
            logging.info(f"📋 Batch [{i}/{len(image_paths)}]")
            result = self.process_image(img_path, **kwargs)
            results.append({
                'image': img_path,
                'text': result
            })
        
        total_duration = time.time() - total_start
        avg_time = total_duration / len(image_paths)
        
        logging.info(f"🎯 Batch complete: {len(image_paths)} images in {total_duration:.2f}s (avg: {avg_time:.2f}s/image)")
        return results
    
    def unload_model(self):
        """Manually unload model from memory to free RAM"""
        try:
            self.client.chat(
                model=self.model_name,
                messages=[],
                keep_alive=0
            )
            logging.info("💾 Model unloaded from memory")
        except Exception as e:
            logging.warning(f"⚠️ Unload failed: {e}")

if __name__ == "__main__":
    # Initialize once - model stays in memory
    ocr = ProductionOCR(keep_alive="60m")
    
    # Single image test
    test_img = "unnamed.jpg"
    if os.path.exists(test_img):
        print("\n" + "="*60)
        print("🧠 PRODUCTION OCR RESULT")
        print("="*60)
        result = ocr.process_image(test_img, preprocess=True)
        print(result)
        print("="*60)
    
    # Batch processing example (commented out)
    # images = ["doc1.jpg", "doc2.jpg", "doc3.jpg"]
    # results = ocr.batch_process(images)
    # for r in results:
    #     print(f"{r['image']}: {len(r['text'])} characters")
    
    # Unload model when done (optional)
    # ocr.unload_model()
