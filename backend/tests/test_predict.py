import requests
import base64
from PIL import Image
from io import BytesIO


img = Image.new("L", (28, 28), color=255)
buf = BytesIO()
img.save(buf, format="PNG")
img_base64 = base64.b64encode(buf.getvalue()).decode()

url = "http://localhost:8000/predict"

response = requests.post(url, json={"image_base64": img_base64})

print("Status code:", response.status_code)
print("Response:", response.json())
