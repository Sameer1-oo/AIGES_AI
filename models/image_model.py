import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

WEIGHTS_PATH = "training/weights/image_model.pth"

# ── Safe model load ──────────────────────────────────────────
try:
    if os.path.exists(WEIGHTS_PATH):
        _model = models.mobilenet_v2(weights=None)
        _model.classifier[1] = nn.Linear(1280, 2)
        _model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        _model.eval()
        print("✅ Image model loaded from weights!")
    else:
        # Weights nahi mili — pretrained use karo
        print("⚠️ Weights not found, using pretrained ImageNet model")
        _model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        _model.classifier[1] = nn.Linear(1280, 2)
        _model.eval()
except Exception as e:
    print(f"❌ Model load error: {e}")
    _model = None

_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def detect_fake_image(file_path: str) -> dict:
    try:
        if _model is None:
            return {"result": "Model Error", "confidence": 0.0}

        img    = Image.open(file_path).convert("RGB")
        tensor = _tfm(img).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(_model(tensor), dim=1)[0]

        fake_p = probs[0].item()
        real_p = probs[1].item()

        if fake_p > real_p:
            return {"result": "Fake",      "confidence": round(fake_p * 100, 2)}
        else:
            return {"result": "Authentic", "confidence": round(real_p * 100, 2)}

    except Exception as e:
        print(f"❌ detect_fake_image error: {e}")
        return {"result": "Error", "confidence": 0.0}