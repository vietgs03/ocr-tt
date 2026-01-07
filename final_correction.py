import ollama
import os

if not os.path.exists('ocr_smart_formatted.txt'):
    print("Error: File ocr_smart_formatted.txt not found. Please run ocr_smart_format.py first.")
    exit()

# Đọc file text đã được sắp xếp sơ bộ
with open('ocr_smart_formatted.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Đang xử lý từng dòng với Ollama (qwen2.5:1.5b)...")

corrected_lines = []

for line in lines:
    line = line.strip()
    if not line:
        continue
        
    print(f"Processing: {line[:50]}...")
    
    prompt = f"""
    Bạn là công cụ sửa lỗi chính tả tự động chuyên nghiệp cho văn bản hành chính Việt Nam (đặc biệt khu vực TP.HCM).
    
    NGỮ CẢNH: Văn bản hành chính của Quận 4, TP.HCM.
    
    NHIỆM VỤ:
    1. Sửa lỗi chính tả và dấu tiếng Việt.
    2. Tách các từ bị dính (ví dụ "Quan4" -> "Quận 4", "Dichvu" -> "Dịch vụ").
    3. Giữ nguyên các thông tin quan trọng (Số hiệu, Ngày tháng, Tên riêng).
    
    QUY TẮC TUYỆT ĐỐI:
    - CHỈ trả về dòng văn bản đã sửa.
    - KHÔNG thêm câu chào hỏi (như "Chào bạn", "Dưới đây là...").
    - KHÔNG giải thích.
    - KHÔNG thêm nội dung bịa đặt (như địa danh tỉnh khác).

    VÍ DỤ MẪU:
    Input: "Cong ty TNHH MTV Dichvu Cong lch Quan4g"
    Output: Công ty TNHH MTV Dịch vụ Công ích Quận 4

    Input: "Qun4.ngayathang 8nm2015"
    Output: Quận 4, ngày 14 tháng 8 năm 2015

    Input: "giam sat tai chinh"
    Output: giám sát tài chính

    Input: "{line}"
    Output:
    """

    try:
        response = ollama.chat(
            model='qwen2.5:1.5b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Lấy kết quả và làm sạch (đôi khi model trả về cả "Output: ...")
        corrected_text = response['message']['content'].strip()
        if "Output:" in corrected_text:
            corrected_text = corrected_text.split("Output:")[-1].strip()
        if corrected_text.startswith('"') and corrected_text.endswith('"'):
            corrected_text = corrected_text[1:-1]
            
        corrected_lines.append(corrected_text)
        
    except Exception as e:
        print(f"Error: {e}")
        corrected_lines.append(line) # Giữ nguyên nếu lỗi

# Ghép lại và lưu
final_content = "\n".join(corrected_lines)

print("\n--- KẾT QUẢ SỬA LỖI TỪNG DÒNG (FINAL) ---")
print(final_content)

with open('final_result_optimized.txt', 'w', encoding='utf-8') as f:
    f.write(final_content)
print("\nĐã lưu vào final_result_optimized.txt")

