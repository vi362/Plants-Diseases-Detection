from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Add the package directory to Python path
sys.path.append(str(Path(__file__).parent))
from model import CNN_NeuralNet

# Initialize FastAPI app
app = FastAPI()

# Mount static and templates directories
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="D:\\My_Projects\\package\\static"), name="static")


templates = Jinja2Templates(directory="D:\\My_Projects\\package\\templates")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Match your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_labels = ['Tomato___Leaf_Mold', 'Potato___Late_blight', 'Soybean___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Tomato___Early_blight', 'Squash___Powdery_mildew', 'Tomato___Tomato_mosaic_virus', 'Apple___healthy', 'Strawberry___healthy', 'Raspberry___healthy', 'Tomato___Target_Spot', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Cherry_(including_sour)___healthy', 'Apple___Cedar_apple_rust', 'Apple___Apple_scab', 'Potato___healthy', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Grape___healthy', 'Grape___Black_rot', 'Orange___Haunglongbing_(Citrus_greening)', 'Tomato___healthy', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Corn_(maize)___healthy', 'Peach___Bacterial_spot', 'Strawberry___Leaf_scorch', 'Corn_(maize)___Common_rust_', 'Peach___healthy', 'Blueberry___healthy', 'Tomato___Late_blight', 'Potato___Early_blight', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Northern_Leaf_Blight']
# Class labels - adjust to match your model
#class_labels = ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Healthy"]

# Load trained PyTorch model
model_path = os.path.join(os.path.dirname(__file__), "model_Plant_Diseases.pth")
model = CNN_NeuralNet(in_channels=3, num_classes=5)

checkpoint = torch.load(model_path, map_location="cpu")
filtered_checkpoint = {
    k: v for k, v in checkpoint.items()
    if not k.startswith('classifier.2')  # Ignore last FC if shape mismatch
}
model.load_state_dict(filtered_checkpoint, strict=False)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Predict route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0)

    # Predict disease
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]

    # Infected area calculation using OpenCV
    # Infected area mask (yellow/brown regions)
    np_img = np.array(image_pil)
    image_cv = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 40])
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Estimate scale (pixels per cm)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        reference_object = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(reference_object)
        known_cm = 2.0  # assumed width of leaf
        pixels_per_cm = w / known_cm
    else:
        pixels_per_cm = 50  # fallback

    # Calculate infected area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    infected_area_px = sum(cv2.contourArea(cnt) for cnt in contours)
    infected_area_cm2 = infected_area_px / (pixels_per_cm ** 2)

    # Calculate total leaf area (bounding box or convex hull)
    green_lower = np.array([25, 40, 40])
    green_upper = np.array([85, 255, 255])
    leaf_mask = cv2.inRange(hsv, green_lower, green_upper)

    leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_area_px = sum(cv2.contourArea(cnt) for cnt in leaf_contours)
    leaf_area_cm2 = leaf_area_px / (pixels_per_cm ** 2)


    # Avoid divide by zero
    if leaf_area_cm2 > 0:
        infected_percentage = (infected_area_cm2 / leaf_area_cm2) * 100
    else:
        infected_percentage = 0.0

    return {
    "predicted_disease": predicted_class,
    "infected_area_cm2": round(infected_area_cm2, 2),
    "infected_percentage": round(infected_percentage, 2)
    }
