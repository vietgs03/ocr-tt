import ollama
import os
import time

def process_image_with_vlm(image_path):
    print(f"Processing {image_path} with Moondream (Lightweight VLM)...")
    start_time = time.time()
    
    # Prompt for Moondream - English prompt usually works better for instructions
    prompt = """
    Please transcribe the text in this image exactly as it appears. 
    Maintain the layout and line breaks.
    """

    try:
        response = ollama.chat(
            model='moondream',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }
            ]
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    img_path = 'unnamed.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: File {img_path} not found.")
    else:
        result = process_image_with_vlm(img_path)
        
        print("\n--- KẾT QUẢ TỪ LLaVA (VLM) ---")
        print(result)
        
        # Save to file
        with open('vlm_result.md', 'w', encoding='utf-8') as f:
            f.write(result)
        print("\nResult saved to vlm_result.md")
