import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import io
import base64

# Create Flask app
app = Flask(__name__)
app.secret_key = 'multi_model_classifier'  # Needed for flashing messages

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the class names
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Model paths - update these with your actual model paths
MODEL_PATHS = {
    'efficientnet': 'efficientnet_b0_fold1.pth',
    'resnet50': 'resnet50_fold1.pth',
    'googlenet': 'googlenet_fold1.pth',
    'regnet_y_8gf': 'regnet_y_8gf_fold1.pth'
}

# Image transformations
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet, GoogLeNet, and RegNetY-8GF use 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_128 = transforms.Compose([
    transforms.Resize((128, 128)),  # EfficientNet can use smaller input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load models
def load_efficientnet():
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=num_ftrs, out_features=len(class_names))
    )
    
    # Load trained weights if available
    if os.path.exists(MODEL_PATHS['efficientnet']):
        model.load_state_dict(torch.load(MODEL_PATHS['efficientnet'], map_location=device))
        print(f"Loaded EfficientNet weights from {MODEL_PATHS['efficientnet']}")
    else:
        print(f"Warning: EfficientNet weights not found at {MODEL_PATHS['efficientnet']}")
    
    model.to(device)
    model.eval()
    return model

def load_resnet50():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Load trained weights if available
    if os.path.exists(MODEL_PATHS['resnet50']):
        model.load_state_dict(torch.load(MODEL_PATHS['resnet50'], map_location=device))
        print(f"Loaded ResNet50 weights from {MODEL_PATHS['resnet50']}")
    else:
        print(f"Warning: ResNet50 weights not found at {MODEL_PATHS['resnet50']}")
    
    model.to(device)
    model.eval()
    return model

def load_googlenet():
    # Change this line to enable aux_logits
    model = models.googlenet(weights=None, aux_logits=True)  # Enable aux_logits to match trained model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Also modify the auxiliary classifier layers
    if hasattr(model, 'aux1'):
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, len(class_names))
    if hasattr(model, 'aux2'):
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, len(class_names))
    
    # Load trained weights if available
    if os.path.exists(MODEL_PATHS['googlenet']):
        model.load_state_dict(torch.load(MODEL_PATHS['googlenet'], map_location=device))
        print(f"Loaded GoogLeNet weights from {MODEL_PATHS['googlenet']}")
    else:
        print(f"Warning: GoogLeNet weights not found at {MODEL_PATHS['googlenet']}")
        print("Using untrained model")
    
    model.to(device)
    model.eval()
    return model

def load_regnet_y_8gf():
    # Load pre-trained RegNetY-8GF model
    model = models.regnet_y_8gf(weights=None)
    
    # Modify the final fully connected layer for our number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Load trained weights if available
    if os.path.exists(MODEL_PATHS['regnet_y_8gf']):
        model.load_state_dict(torch.load(MODEL_PATHS['regnet_y_8gf'], map_location=device))
        print(f"Loaded RegNetY-8GF weights from {MODEL_PATHS['regnet_y_8gf']}")
    else:
        print(f"Warning: RegNetY-8GF weights not found at {MODEL_PATHS['regnet_y_8gf']}")
        print("Using untrained model")
    
    model.to(device)
    model.eval()
    return model

# Dictionary to store loaded models
models_dict = {}

def load_models():
    """Load all models into memory"""
    print("Loading models...")
    models_dict['efficientnet'] = load_efficientnet()
    models_dict['resnet50'] = load_resnet50()
    models_dict['googlenet'] = load_googlenet()
    models_dict['regnet_y_8gf'] = load_regnet_y_8gf()
    print("All models loaded successfully")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_with_model(img, model_name):
    """Make a prediction using the specified model"""
    model = models_dict.get(model_name)
    if not model:
        return {"error": f"Model {model_name} not found"}
    
    # Apply appropriate transform
    if model_name == 'efficientnet':
        img_tensor = transform_128(img).unsqueeze(0).to(device)
    else:  # ResNet50, GoogLeNet, and RegNetY-8GF all use 224x224
        img_tensor = transform_224(img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        # Handle GoogLeNet's auxiliary outputs during inference
        if model_name == 'googlenet' and hasattr(model, 'aux_logits') and model.aux_logits:
            # During inference, GoogLeNet returns only the main output even with aux_logits=True
            model.eval()  # Make sure model is in eval mode
            outputs = model(img_tensor)
        else:
            outputs = model(img_tensor)
            
        probabilities = F.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]
    
    # Convert probabilities to percentages
    probs = [float(p) * 100 for p in probabilities.cpu().numpy()]
    
    # Create a list of (class_name, probability) tuples
    prob_dict = [{'class': class_names[i], 'prob': probs[i]} for i in range(len(class_names))]
    
    # Sort by probability (highest first)
    prob_dict = sorted(prob_dict, key=lambda x: x['prob'], reverse=True)
    
    return {
        'prediction': prediction,
        'probabilities': prob_dict
    }

@app.route('/')
def index():
    return render_template('multi_model_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Open image for prediction
        img = Image.open(filepath).convert('RGB')
        
        # Get predictions from all models
        results = {}
        for model_name in models_dict.keys():
            results[model_name] = predict_with_model(img, model_name)
        
        # Render the results page
        return render_template('multi_model_result.html', 
                              filename=filename,
                              results=results,
                              class_names=class_names)
    
    flash('Invalid file type')
    return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    # Load models at startup
    load_models()
    app.run(debug=True)
