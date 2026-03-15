"""
AEGIS AI - API Routes
All detection endpoints: image, video, audio, text, link.
"""

import os
import shutil
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from backend.utils import save_media, save_link
from models.image_model import detect_fake_image
from models.video_model import detect_fake_video
from models.audio_model import detect_fake_audio
from models.text_model import detect_fake_text
from models.link_model import detect_malicious_link

# ─── Setup ───────────────────────────────────────────────────────────────────
router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── Helper: Save Uploaded File ──────────────────────────────────────────────
def _save_upload(file: UploadFile) -> str:
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


# ─── /detect/image ───────────────────────────────────────────────────────────
@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        path = _save_upload(file)
        analysis = detect_fake_image(path)
        save_media(file.filename, "image", analysis["result"], analysis["confidence"])
        return JSONResponse({"filename": file.filename, **analysis})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── /detect/video ───────────────────────────────────────────────────────────
@router.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    try:
        path = _save_upload(file)
        analysis = detect_fake_video(path)
        save_media(file.filename, "video", analysis["result"], analysis["confidence"])
        return JSONResponse({"filename": file.filename, **analysis})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── /detect/audio ───────────────────────────────────────────────────────────
@router.post("/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    try:
        path = _save_upload(file)
        analysis = detect_fake_audio(path)
        save_media(file.filename, "audio", analysis["result"], analysis["confidence"])
        return JSONResponse({"filename": file.filename, **analysis})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── /detect/text ────────────────────────────────────────────────────────────
@router.post("/detect/text")
async def detect_text(content: str = Form(...)):
    try:
        analysis = detect_fake_text(content)
        save_media("text_input", "text", analysis["result"], analysis["confidence"])
        return JSONResponse({"input_preview": content[:80], **analysis})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── /detect/link ────────────────────────────────────────────────────────────
@router.post("/detect/link")
async def detect_link(url: str = Form(...)):
    try:
        analysis = detect_malicious_link(url)
        save_link(url, analysis["result"])
        return JSONResponse({"url": url, **analysis})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── /history/media ──────────────────────────────────────────────────────────
@router.get("/history/media")
async def history_media():
    from backend.utils import get_all_media
    return JSONResponse(get_all_media())


# ─── /history/links ──────────────────────────────────────────────────────────
@router.get("/history/links")
async def history_links():
    from backend.utils import get_all_links
    return JSONResponse(get_all_links())
