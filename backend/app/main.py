from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch.nn as nn


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image_base64: str  

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output

model = Net()
model.load_state_dict(torch.load("app/models/mnist_cnn.pth", map_location="cpu"))
model.eval()

def preprocess_image(image_base64: str):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert("L")  
        image = image.resize((28, 28))
        image_np = np.array(image) / 255.0
        image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

@app.post("/predict")
def predict(request: PredictRequest):
    img_tensor = preprocess_image(request.image_base64)
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1, keepdim=True).item()
    return {"prediction": int(pred)}
