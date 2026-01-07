import os
import time
import ollama
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class VlmOCR:
    def __init__(self, model_name="deepseek-ocr"):
        self.client = ollama.Client(host='http://127.0.0.1:11434')
        self.model_name = model_name
        logging.info(f"🚀 Initialized VlmOCR with model: {self.model_name}")

    def parse_grounding_output(self, raw_output):
        """
        Parse DeepSeek-OCR grounding output to extract clean text.
        Input: text with <|ref|>content<|/ref|><|det|>coords<|/det|> tags
        Output: clean text without tags
        """
        import re
        # Pattern to match paired grounding tags
        pattern = r'<\|ref\|\>(.*?)<\|/ref\|\>\s*<\|det\|\>(.*?)<\|/det\|\>'
        
        # Extract all ref content (the actual text)
        matches = re.findall(pattern, raw_output, re.DOTALL)
        
        # Combine all text parts
        clean_text = '\n'.join([match[0] for match in matches])
        
        return clean_text if clean_text else raw_output

    def process_image(self, image_path, prompt="Free OCR.", clean_output=True):
        if not os.path.exists(image_path):
            return f"❌ Error: File {image_path} not found"

        logging.info(f"📸 Processing: {image_path}")
        start_time = time.time()
        
        try:
            # Send to Ollama (VLM mode)
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path] # Ollama python lib handles file reading if path is provided
                }]
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

if __name__ == "__main__":
    ocr = VlmOCR(model_name="deepseek-ocr")
    
    test_img = "unnamed.jpg"
    if os.path.exists(test_img):
        print("\n--- 🧠 VLM Result ---")
        print(ocr.process_image(test_img))
