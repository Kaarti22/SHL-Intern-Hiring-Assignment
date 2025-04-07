import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn_model import AudioCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "outputs/model_cnn.pth"
test_df = pd.read_csv("data/test.csv")
image_dir = "data/mel_spectrograms_test"
output_path = "outputs/submission.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = AudioCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

predictions = []
for fname in test_df["filename"]:
    image_path = os.path.join(image_dir, fname.replace(".wav", ".png"))
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image)
        predictions.append(pred.cpu().item())

submission_df = pd.DataFrame({
    "filename": test_df["filename"],
    "label": predictions
})
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")