import os
import requests
import time
import random
import re

# Danh sách từ khóa chuyên dụng để tìm biên bản/báo cáo
KEYWORDS = [
    "biên bản nghiệm thu công trình xây dựng scan",
    "quyết định ủy ban nhân dân scan",
    "báo cáo giám sát thi công scan",
    "công văn hành chính nhà nước scan",
    "hợp đồng kinh tế xây dựng scan",
    "giấy phép xây dựng scan",
    "biên bản làm việc hành chính scan",
    "tờ trình phê duyệt scan",
    "thông báo kết luận cuộc họp scan"
]

def search_images_bing(query, limit=20):
    """
    Tìm ảnh trên Bing (đơn giản hơn Google) để lấy URL ảnh văn bản.
    """
    print(f"🔎 Searching for: '{query}'...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    # Bing image search endpoint
    search_url = f"https://www.bing.com/images/async?q={query}&first=0&count={limit}&adlt=off"
    
    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            # Bing trả về HTML chứa link ảnh trong pattern murl":"url"
            # Regex đơn giản để bắt link ảnh
            image_links = re.findall(r'murl&quot;:&quot;(http[^&]+?\.(?:jpg|jpeg|png))&quot;', response.text)
            return image_links
    except Exception as e:
        print(f"Error searching: {e}")
    return []

def download_images(folder="data/train_images"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    total_downloaded = 0
    
    for keyword in KEYWORDS:
        links = search_images_bing(keyword, limit=30)
        print(f"   -> Found {len(links)} potential images for '{keyword}'")
        
        for url in links:
            try:
                # Tạo tên file ngẫu nhiên
                filename = f"doc_crawler_{int(time.time())}_{random.randint(1000,9999)}.jpg"
                save_path = os.path.join(folder, filename)
                
                # Tải ảnh với timeout ngắn
                response = requests.get(url, timeout=5)
                img_data = response.content
                
                # Check kích thước file (bỏ qua file < 30KB vì khả năng cao là thumbnail mờ, không tốt để train OCR)
                if len(img_data) > 30 * 1024: 
                    with open(save_path, 'wb') as f:
                        f.write(img_data)
                    print(f"   ✅ Saved: {filename}")
                    total_downloaded += 1
                    # Nghỉ ngẫu nhiên từ 0.5 - 1.5s để tránh bị chặn IP
                    time.sleep(random.uniform(0.5, 1.5))
                else:
                    pass # Bỏ qua không in log cho đỡ rối
                    
            except Exception as e:
                pass # Lỗi mạng bỏ qua
                
    print(f"\n🎉 Completed! Total images downloaded: {total_downloaded}")
    print(f"👉 Images saved to: {folder}")

if __name__ == "__main__":
    download_images()

