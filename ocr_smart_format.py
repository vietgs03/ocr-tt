import json
import math

def sort_ocr_blocks(blocks):
    """
    Sắp xếp các khối text dựa trên tọa độ Y (dòng) và X (cột).
    Nếu chênh lệch Y nhỏ hơn threshold (ví dụ 10px) thì coi là cùng một dòng.
    """
    # Sắp xếp sơ bộ theo Y
    blocks.sort(key=lambda b: b['box'][0][1])
    
    sorted_lines = []
    current_line = []
    current_y = -1
    y_threshold = 15 # Pixel threshold để xác định dòng mới
    
    for block in blocks:
        y = block['box'][0][1]
        
        # Nếu dòng mới (chênh lệch Y đủ lớn)
        if current_y == -1 or abs(y - current_y) <= y_threshold:
            current_line.append(block)
            # Giữ Y của phần tử đầu tiên làm mốc cho dòng
            if current_y == -1:
                current_y = y
        else:
            # Kết thúc dòng cũ, sắp xếp các phần tử trong dòng theo X
            current_line.sort(key=lambda b: b['box'][0][0])
            sorted_lines.append(current_line)
            
            # Bắt đầu dòng mới
            current_line = [block]
            current_y = y
            
    # Thêm dòng cuối cùng
    if current_line:
        current_line.sort(key=lambda b: b['box'][0][0])
        sorted_lines.append(current_line)
        
    return sorted_lines

def format_text_layout(sorted_lines):
    """
    Tái tạo layout dựa trên vị trí X.
    """
    output_text = ""
    
    for line in sorted_lines:
        line_str = ""
        last_x_end = 0
        
        for block in line:
            text = block['text']
            x_start = block['box'][0][0]
            
            # Tính khoảng cách từ lề trái (cho phần tử đầu tiên)
            if last_x_end == 0:
                if x_start > 350: # Lệch phải nhiều (ví dụ chữ ký, ngày tháng)
                    line_str += "\t\t\t\t"
                elif x_start > 150: # Lệch giữa/phải vừa
                    line_str += "\t\t"
            else:
                # Tính khoảng cách giữa các khối trên cùng 1 dòng
                gap = x_start - last_x_end
                if gap > 50: # Khoảng cách lớn giữa các chữ -> Tab
                    line_str += "\t"
                else:
                    line_str += " "
            
            line_str += text
            last_x_end = block['box'][1][0] # X kết thúc của block này
            
        output_text += line_str.strip() + "\n"
        
    return output_text

# Main execution
if __name__ == "__main__":
    try:
        with open('ocr_result.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Lấy trang đầu tiên
        blocks = data[0]
        
        # Sắp xếp
        sorted_lines = sort_ocr_blocks(blocks)
        
        # Format
        final_text = format_text_layout(sorted_lines)
        
        print("--- SMART FORMAT RESULT ---")
        print(final_text)
        
        with open('ocr_smart_formatted.txt', 'w', encoding='utf-8') as f:
            f.write(final_text)
        print("\nResult saved to ocr_smart_formatted.txt")
            
    except Exception as e:
        print(f"Error: {e}")

