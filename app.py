from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ARCHIVE_FOLDER'] = 'archive/' 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN3(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.classifier(x)
        return x

class FeatureExt(nn.Module):
    def __init__(self):
        super(FeatureExt, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x

category_model = CNN3(num_classes=5).to(device)
category_model.load_state_dict(torch.load('model.pth', map_location=device))
category_model.eval()

feature_extractor = FeatureExt().to(device)
feature_extractor.eval()

category_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

siamese_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def find_closest_matches(features, num_matches=2):
    min_distances = [(float('inf'), None) for _ in range(num_matches)]
    for img_path, img_features in reference_features.items():
        distance = torch.norm(features - img_features)
        for i in range(num_matches):
            if distance < min_distances[i][0]:
                min_distances.insert(i, (distance.item(), img_path))
                min_distances.pop(-1)
                break
    return [(dist, path) for dist, path in min_distances if path is not None]

reference_features = {} 
for root, dirs, files in os.walk(app.config['ARCHIVE_FOLDER']):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(root, file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = siamese_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = feature_extractor(img_tensor)
            reference_features[img_path] = features.squeeze(0)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = Image.open(filepath).convert('RGB')
            category_tensor = category_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = category_model(category_tensor)
                _, predicted = torch.max(outputs, 1)
                category_index = predicted.item()
                category_map = {0: "Beauty", 1: "Family", 2: "Fashion", 3: "Fitness", 4: "Food"}
                category = category_map.get(category_index, "Unknown")

            siamese_tensor = siamese_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = feature_extractor(siamese_tensor).squeeze(0)
            closest_matches = find_closest_matches(features, num_matches=2)

            if len(closest_matches) > 1:
                _, second_closest_image = closest_matches[1]
                second_closest_distance = closest_matches[1][0]
                second_closest_image = second_closest_image.replace(app.config['ARCHIVE_FOLDER'], '')
            else:
                second_closest_image, second_closest_distance = "No match", "N/A"

            return render_template('results.html', category=category, filename=filename,
                                   second_closest_image=second_closest_image, second_distance=second_closest_distance)
        return redirect(url_for('upload_file'))
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/archive/<path:filename>')
def archive_file(filename):
    return send_from_directory(app.config['ARCHIVE_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
