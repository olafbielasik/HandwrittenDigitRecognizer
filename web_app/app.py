from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io, base64

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

app = Flask(__name__)
model = MNISTClassifier()
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    header, encoded = image_data.split(',', 1)
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded)).convert('L')
    image.save("debug_image.png")
    image = transform(image).unsqueeze(0)
    output = model(image)
    prediction = output.argmax(dim=1).item()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)




