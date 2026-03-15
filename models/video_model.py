"""
AEGIS AI — Video Deepfake Detection
Strategy: Extract frames → Run image model on each → Average results
Reuses: training/weights/image_model.pth (already trained!)
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ── Image model load karo (same jo tune train kiya tha) ──────
_model = models.mobilenet_v2(weights=None)
_model.classifier[1] = nn.Linear(1280, 2)
_model.load_state_dict(torch.load(
    "training/weights/image_model.pth",
    map_location="cpu"
))
_model.eval()

_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ── Config ───────────────────────────────────────────────────
FRAMES_TO_SAMPLE = 10   # Video se kitne frames analyze karne hain
                         # Zyada = accurate, lekin slow

def _extract_frames(video_path: str, num_frames: int) -> list:
    """Video se evenly spaced frames nikalo."""
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total == 0:
        cap.release()
        return []

    # Evenly spaced frame indices
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV BGR → PIL RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def _analyze_frame(pil_image: Image.Image) -> dict:
    """Ek frame pe image model run karo."""
    tensor = _tfm(pil_image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]
    return {
        "fake_prob": probs[0].item(),   # Fake=0
        "real_prob": probs[1].item()    # Real=1
    }


def detect_fake_video(file_path: str) -> dict:
    """
    Video analyze karo — frames nikalo, har frame check karo,
    average confidence se final verdict do.
    """
    # Frames nikalo
    frames = _extract_frames(file_path, FRAMES_TO_SAMPLE)

    if not frames:
        return {"result": "Error", "confidence": 0.0}

    print(f"  Analyzing {len(frames)} frames...")

    # Har frame analyze karo
    fake_probs = []
    real_probs = []

    for frame in frames:
        result = _analyze_frame(frame)
        fake_probs.append(result["fake_prob"])
        real_probs.append(result["real_prob"])

    # Average nikalo
    avg_fake = sum(fake_probs) / len(fake_probs)
    avg_real = sum(real_probs) / len(real_probs)

    # Final verdict
    if avg_fake > avg_real:
        return {
            "result":     "Fake",
            "confidence": round(avg_fake * 100, 2)
        }
    else:
        return {
            "result":     "Authentic",
            "confidence": round(avg_real * 100, 2)
        }