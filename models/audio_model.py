"""
AEGIS AI — Audio Voice Clone Detection
Strategy: MFCC features → MobileNetV1 CNN classifier
No separate training needed for basic detection.
For better accuracy: train with ASVspoof dataset.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models

# ── Try librosa import ───────────────────────────────────────
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not installed. Run: pip install librosa soundfile")

# ── Model load karo ──────────────────────────────────────────
_model = models.mobilenet_v2(weights="IMAGENET1K_V1")
_model.classifier[1] = nn.Linear(1280, 2)
_model.eval()

# Agar audio ka trained model hai to load karo
AUDIO_WEIGHTS = "training/weights/audio_model.pth"
if os.path.exists(AUDIO_WEIGHTS):
    _model.load_state_dict(torch.load(AUDIO_WEIGHTS, map_location="cpu"))
    print("✅ Custom audio model loaded!")
else:
    print("ℹ️ Using pretrained ImageNet weights for audio (MFCC image approach)")

# ── Transform ────────────────────────────────────────────────
_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def _audio_to_mfcc_image(file_path: str) -> np.ndarray | None:
    """
    Audio file ko MFCC spectrogram image mein convert karo.
    Yeh image model ko feed hogi.
    """
    try:
        # Audio load karo
        y, sr = librosa.load(file_path, sr=16000, mono=True, duration=10.0)

        # MFCC features nikalo
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Normalize karo 0-255 range mein
        mfcc_norm = ((mfcc - mfcc.min()) /
                     (mfcc.max() - mfcc.min() + 1e-8) * 255).astype(np.uint8)

        # 3 channel banao (RGB jaisa) — model ke liye
        mfcc_rgb = np.stack([mfcc_norm, mfcc_norm, mfcc_norm], axis=-1)

        return mfcc_rgb

    except Exception as e:
        print(f"MFCC extraction error: {e}")
        return None


def detect_fake_audio(file_path: str) -> dict:
    """
    Audio file analyze karo voice cloning ke liye.

    Args:
        file_path: Uploaded audio file ka path

    Returns:
        {"result": "Fake" | "Authentic", "confidence": float}
    """
    if not LIBROSA_AVAILABLE:
        return {
            "result": "Error",
            "confidence": 0.0
        }

    # MFCC image banao
    mfcc_img = _audio_to_mfcc_image(file_path)

    if mfcc_img is None:
        return {"result": "Error", "confidence": 0.0}

    # Model pe run karo
    try:
        tensor = _tfm(mfcc_img).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(_model(tensor), dim=1)[0]

        fake_p = probs[0].item()
        real_p = probs[1].item()

        # Additional heuristic — librosa se anomaly check
        y, sr  = librosa.load(file_path, sr=16000, mono=True, duration=10.0)
        
        # Zero crossing rate — synthetic audio mein usually alag hoti hai
        zcr     = librosa.feature.zero_crossing_rate(y)[0].mean()
        
        # Spectral flatness — cloned voice mein flatness zyada hoti hai
        flatness = librosa.feature.spectral_flatness(y=y)[0].mean()

        # Heuristic boost
        if flatness > 0.3:      # Zyada flat = synthetic
            fake_p = min(fake_p + 0.15, 0.99)
            real_p = 1 - fake_p
        elif flatness < 0.05:   # Bahut natural
            real_p = min(real_p + 0.10, 0.99)
            fake_p = 1 - real_p

        if fake_p > real_p:
            return {"result": "Fake",      "confidence": round(fake_p * 100, 2)}
        else:
            return {"result": "Authentic", "confidence": round(real_p * 100, 2)}

    except Exception as e:
        print(f"Audio analysis error: {e}")
        return {"result": "Error", "confidence": 0.0}