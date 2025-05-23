import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Load pre-trained ResNet50 model
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Feature extraction
def extract_features(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = resnet(image_tensor)
    return features.squeeze().numpy().flatten()

# Load DataFrame with features from pickle
features_path = "features.pkl"
df = pd.read_pickle(features_path)

# Function to find best match
def find_best_match(found_image_path):
    found_features = extract_features(found_image_path)
    stored_features = np.stack(df['features'].values)
    similarities = cosine_similarity([found_features], stored_features)
    best_match_idx = np.argmax(similarities)
    return df.iloc[best_match_idx]

# Flask Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")

@app.route("/match", methods=["POST"])
def match():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        os.makedirs("static/uploads", exist_ok=True)
        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)
        
        best_match = find_best_match(file_path)
        
        return render_template("result.html", 
                               image=file.filename,
                               item_name=best_match["item_name"],
                               person_name=best_match["person_name"],
                               description=best_match["description"],
                               location=best_match["location"],
                               owner_contact=best_match["owner_contact"])
    return redirect(url_for("index"))

# Production-safe entrypoint
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
