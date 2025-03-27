import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# إعداد المجلد لتخزين الصور المرفوعة
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# دالة المعالجة المسبقة للصور
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img

# تحميل النموذج
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.7),
    nn.Linear(num_ftrs, 2)
)
device = torch.device("cpu")
model.load_state_dict(torch.load("best_waste_model_subset.pth"))
model = model.to(device)
model.eval()

# دالة التنبؤ
def predict_image(image_path, confidence_threshold=0.5):
    img = preprocess
