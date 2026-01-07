from paddleocr import PaddleOCR
import json
import ollama

# Initialize PaddleOCR
# use_angle_cls=True allows detecting text at different angles
# lang='vi' for Vietnamese support
ocr = PaddleOCR(use_angle_cls=True, lang='vi')

img_path = 'unnamed.jpg'

# Run OCR on the image
print("Running OCR...")
result = ocr.ocr(img_path, cls=True)

# Process results
output_data = []
formatted_input = ""

if result is not None and result[0] is not None:
    for idx, res in enumerate(result):
        if res is None:
            continue
        section_data = []
        for line in res:
            text_content = line[1][0]
            score = line[1][1]
            box = line[0]
            section_data.append({
                "text": text_content,
                "confidence": score,
                "box": box
            })
            formatted_input += text_content + "\n"
        output_data.append(section_data)

# Export raw OCR results to JSON
with open('ocr_result.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
print("Raw OCR results exported to ocr_result.json")

# Use Ollama to correct and format text
print("\nSending text to Ollama (qwen2.5:1.5b) for correction...")

prompt = f"""
Bạn là chuyên gia soạn thảo văn bản hành chính Việt Nam. Nhiệm vụ của bạn là phục hồi và định dạng lại văn bản từ kết quả OCR thô dưới đây.

YÊU CẦU:
1. Sửa lỗi chính tả tiếng Việt (Ví dụ: "Qun4" -> "Quận 4", "ngayathang" -> "ngày tháng", "giam sat" -> "giám sát").
2. Giữ nguyên các thông tin quan trọng: Số hiệu (290/BC-UBND), ngày tháng, tên người ký.
3. Định dạng lại văn bản cho dễ đọc, chia đoạn hợp lý theo cấu trúc văn bản hành chính:
   - Phần đầu: Quốc hiệu, Tiêu ngữ, Tên cơ quan, Số hiệu, Ngày tháng.
   - Phần giữa: Tên Báo cáo, Căn cứ pháp lý (các dòng bắt đầu bằng "Căn cứ"), Nội dung báo cáo.
   - Phần cuối: Nơi nhận và Người ký (Chủ tịch).

NỘI DUNG OCR THÔ:
{formatted_input}

VĂN BẢN ĐÃ CHỈNH SỬA VÀ ĐỊNH DẠNG:
"""

try:
    response = ollama.chat(model='qwen2.5:1.5b', messages=[
        {'role': 'user', 'content': prompt},
    ])
    
    corrected_text = response['message']['content']
    
    print("\n--- KẾT QUẢ ĐÃ CHỈNH SỬA (OLLAMA) ---")
    print(corrected_text)
    
    # Save corrected text to file
    with open('ocr_corrected.txt', 'w', encoding='utf-8') as f:
        f.write(corrected_text)
    print("\nCorrected text saved to ocr_corrected.txt")
    
except Exception as e:
    print(f"\nError calling Ollama: {e}")
    print("Make sure Ollama is running and the model qwen2.5:1.5b is pulled.")
