import requests

url = "http://localhost:8000/ocr/accurate"
files = {'file': open('unnamed.jpg', 'rb')}

try:
    print("Sending request to Accurate Mode (this may take time)...")
    response = requests.post(url, files=files, timeout=300)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
