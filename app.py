import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ResNet50 model
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Extract image features
def extract_features(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        features = resnet(image_tensor)
    return features.squeeze().numpy().flatten()

# Load precomputed features
df = pd.read_csv("lost_items_dataset.csv")

# Convert stringified list to numpy array
df['features'] = df['features'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))

# Match found image with dataset
def find_best_match(found_image_path):
    found_features = extract_features(found_image_path)
    stored_features = np.stack(df['features'].values)
    similarities = cosine_similarity([found_features], stored_features)
    best_match_idx = np.argmax(similarities)
    return df.iloc[best_match_idx]

# Routes
@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/match", methods=["POST"])
def match():
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(request.url)

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    best_match = find_best_match(file_path)

    return render_template("result.html",
                           image=file.filename,
                           item_name=best_match["item_name"],
                           person_name=best_match["person_name"],
                           description=best_match["description"],
                           location=best_match["location"],
                           owner_contact=best_match["owner_contact"])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
