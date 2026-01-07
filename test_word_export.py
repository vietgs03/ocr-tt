import requests

url = "http://localhost:8000/export/docx"
data = {'text': "This is a test document.\nWith Times New Roman font."}

try:
    response = requests.post(url, data=data)
    if response.status_code == 200:
        with open('test_output.docx', 'wb') as f:
            f.write(response.content)
            print("Successfully saved test_output.docx")
    else:
        print(f"Failed: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error: {e}")
