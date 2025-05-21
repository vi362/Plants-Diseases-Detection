from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from fastapi.responses import FileResponse
from fpdf import FPDF
import io
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from fastapi import Query
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from datetime import datetime
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fpdf import FPDF
import uuid
import sys
from pathlib import Path
from fastapi.responses import JSONResponse
import json
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import uuid
import io
from io import BytesIO
from PIL import Image
import json
from fastapi.responses import StreamingResponse

from fastapi import FastAPI, Depends, HTTPException






# Add the package directory to Python path
sys.path.append(str(Path(__file__).parent))
from model import CNN_NeuralNet

# Initialize FastAPI app
app = FastAPI()

# Mount static and templates directories
from fastapi.staticfiles import StaticFiles

#app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
# at top, after BASE_DIR is defined




#app.mount("/static", StaticFiles(directory="D:\\My_Projects\\package\\static"), name="static")
#templates = Jinja2Templates(directory="D:\\My_Projects\\package\\templates")

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))



# Absolute path to your processed folder
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#STATIC_DIR = os.path.join(BASE_DIR, "static")
#PROCESSED_DIR = os.path.join(STATIC_DIR, "processed")

STATIC_DIR = BASE_DIR / "static"
PROCESSED_DIR = STATIC_DIR / "processed"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],  # Match your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class_labels = ['Tomato___Leaf_Mold', 'Potato___Late_blight', 'Soybean___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Tomato___Early_blight', 'Squash___Powdery_mildew', 'Tomato___Tomato_mosaic_virus', 'Apple___healthy', 'Strawberry___healthy', 'Raspberry___healthy', 'Tomato___Target_Spot', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Cherry_(including_sour)___healthy', 'Apple___Cedar_apple_rust', 'Apple___Apple_scab', 'Potato___healthy', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Grape___healthy', 'Grape___Black_rot', 'Orange___Haunglongbing_(Citrus_greening)', 'Tomato___healthy', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Corn_(maize)___healthy', 'Peach___Bacterial_spot', 'Strawberry___Leaf_scorch', 'Corn_(maize)___Common_rust_', 'Peach___healthy', 'Blueberry___healthy', 'Tomato___Late_blight', 'Potato___Early_blight', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Northern_Leaf_Blight']
# Class labels - adjust to match your model
#class_labels = ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Healthy"]

# Load disease data
with open("D:\\My_Projects\\package\\diseases.json", "r") as f:
    disease_data = json.load(f)

# Load trained PyTorch model
model_path = os.path.join(os.path.dirname(__file__), "mobilenet_model.pth")
model = CNN_NeuralNet(in_channels=3, num_classes=38)

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
    transforms.ToTensor()
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
    try:
        contents = await file.read()
        image_pil = Image.open(BytesIO(contents)).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0)  # [1, 3, 224, 224]

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_labels[predicted.item()]

        np_img = np.array(image_pil)
        image_cv = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 40, 40])
        upper = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            reference_object = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(reference_object)
            known_cm = 2.0
            pixels_per_cm = w / known_cm
        else:
            pixels_per_cm = 50

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        infected_area_px = sum(cv2.contourArea(cnt) for cnt in contours)
        infected_area_cm2 = infected_area_px / (pixels_per_cm ** 2)

        green_lower = np.array([25, 40, 40])
        green_upper = np.array([85, 255, 255])
        leaf_mask = cv2.inRange(hsv, green_lower, green_upper)

        leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        leaf_area_px = sum(cv2.contourArea(cnt) for cnt in leaf_contours)
        leaf_area_cm2 = leaf_area_px / (pixels_per_cm ** 2)

        infected_percentage = (infected_area_cm2 / leaf_area_cm2) * 100 if leaf_area_cm2 > 0 else 0.0

        marked_image = image_cv.copy()
        cv2.drawContours(marked_image, contours, -1, (0, 0, 255), 2)

        # Add label text
        #text = f"Disease: {predicted_class}"
        #cv2.putText(marked_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"marked_{uuid.uuid4().hex}.png"
        save_path = PROCESSED_DIR / filename
        cv2.imwrite(str(save_path), marked_image)
        image_path = str(save_path)
        marked_image_url = f"/static/processed/{filename}"

        return{
            "predicted_disease": predicted_class,
            "infected_area_cm2": round(infected_area_cm2, 2),
            "infected_percentage": round(infected_percentage, 2),
            "image_path": image_path,
            "marked_image_url": marked_image_url
        }

    except Exception as e:
        # Log error for debugging
        import traceback
        traceback.print_exc()
        # Return readable error message
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/remedies")
async def get_remedy(disease: str = Query(...)):
    with open("diseases.json", "r") as f:
        diseases = json.load(f)

    disease_info = diseases.get(disease)
    if disease_info:
        return {
            "brief": disease_info.get("brief", "No info available"),
            "remedies": disease_info.get("remedies", "No remedies listed"),
            "pesticides": disease_info.get("pesticides", "No pesticides listed")
        }
    else:
        return {
            "brief": "No information available.",
            "remedies": "N/A",
            "pesticides": "N/A"
        }

class ReportRequest(BaseModel):
    disease: str
    area_cm2: float
    percentage: float
    description: str
    remedy: str
    pesticide: str
    image_url: str





@app.post("/generate-report")
async def generate_report(data: dict):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=24)
        pdf.cell(200, 10, txt="PlantX Disease Report", ln=True, align='C')

        # Horizontal line
        pdf.set_draw_color(0, 0, 0)  # Black color
        pdf.set_line_width(1.5)
        pdf.line(x1=10, y1=pdf.get_y(), x2=200, y2=pdf.get_y())  # Draw line from left to right
        pdf.ln(5)  # Add vertical space

        # Current date & time at top-right corner
        now = datetime.now().strftime("%d-%m-%Y %I:%M %p")
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, txt=f"{now}", ln=True, align='R')
        pdf.ln(5)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Disease: {data['disease']}", ln=True)
        pdf.cell(200, 10, txt=f"Infected Area: {data['area_cm2']} cm²", ln=True)
        pdf.cell(200, 10, txt=f"Infected Percentage: {data['percentage']}%", ln=True)

        if "brief" in data:
            pdf.ln(5)
            pdf.multi_cell(0, 10, f"Brief:\n{data['brief']}")
        if "remedies" in data:
            pdf.ln(5)
            pdf.multi_cell(0, 10, f"Remedies:\n{data['remedies']}")
        if "pesticides" in data:
            pdf.ln(5)
            pdf.multi_cell(0, 10, f"Pesticides:\n{data['pesticides']}")

        image_url = data.get("image_path")
        print("Received image_path:", image_url)
        if image_url:
            relative_path = image_url.lstrip("/")
            if relative_path.startswith("static/"):
                relative_path = relative_path[len("static/"):]
            image_path = os.path.join(STATIC_DIR, relative_path)
            print(f"Looking for image at: {image_path}")

            # Include marked image
            image_path = data.get("image_path")
            print("Received image path:", image_path)

            if image_path and os.path.exists(image_path):
                pdf.ln(10)
                pdf.cell(200, 10, txt="Marked Image:", ln=True)
                pdf.image(image_path, x=10, w=180)
            else:
                print("Image not found or not provided:", image_path)


        filename = f"report_{uuid.uuid4()}.pdf"
        filepath = os.path.join("reports", filename)
        os.makedirs("reports", exist_ok=True)
        pdf.output(filepath)

        return FileResponse(filepath, media_type='application/pdf', filename=filename)

    except Exception as e:
        print("Error generating PDF:", e)
        return JSONResponse(status_code=500, content={"error": "PDF generation failed"})

# Utility function to wrap long text
def split_text(text, max_length):
    import textwrap
    return textwrap.wrap(text, width=max_length)
