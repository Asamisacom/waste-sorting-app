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
def preprocess_image(image_path, target_size=(64, 64)):
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
model.load_state_dict(torch.load("best_waste_model_subset.pth", map_location=device))
model = model.to(device)
model.eval()

# دالة التنبؤ
def predict_image(image_path, confidence_threshold=0.5):
    img = preprocess_image(image_path)
    if img is None:
        return "Error: Could not process the image.", 0.0
    
    with torch.no_grad():
        outputs = model(img.to(device))
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = predicted.item()
        confidence_score = confidence.item()

    # تحرير الذاكرة
    del img, outputs, probabilities
    torch.cuda.empty_cache()

    # تحديد الفئات
    class_names = ["Recyclable", "Non-Recyclable"]
    if confidence_score < confidence_threshold:
        return "Low confidence prediction. Please try another image.", confidence_score
    
    return class_names[predicted_class], confidence_score

# مسار الصفحة الرئيسية
@app.route('/')
def index():
    return render_template('index.html')

# مسار التنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    print("Received a request to /predict")
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    print(f"File received: {file.filename}")
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            filename = secure_filename(file.filename)
            print(f"Secure filename: {filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving file to: {file_path}")
            file.save(file_path)
            
            # التنبؤ
            print("Starting prediction...")
            prediction, confidence = predict_image(file_path)
            print(f"Prediction result: {prediction}, Confidence: {confidence}")
            
            # حذف الصورة بعد التنبؤ
            os.remove(file_path)
            print("File deleted after prediction")
            
            return jsonify({
                'prediction': prediction,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Error during prediction: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)