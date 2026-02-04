from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from audio_utils import analyze_voice
import os
import base64
import io
import librosa
import numpy as np

app = FastAPI(title="AI Voice Detection API")

# ------------------
# Config
# ------------------
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")
PORT = int(os.getenv("PORT", 8000))
SUPPORTED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

# ------------------
# Request model
# ------------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ------------------
# Health check (root)
# ------------------
@app.get("/")
def health_check():
    return {"status": "running", "message": "AI Voice Detection API is live"}

# ------------------
# API key verification
# ------------------
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ------------------
# Voice detection endpoint
# ------------------
@app.post("/api/voice-detection")
def voice_detection(request: VoiceRequest, x_api_key: str = Header(...)):
    # Check API key
    verify_api_key(x_api_key)

    # Validate language
    if request.language not in SUPPORTED_LANGUAGES:
        return {"status": "error", "message": f"Unsupported language: {request.language}"}

    # Validate format
    if request.audioFormat.lower() != "mp3":
        return {"status": "error", "message": f"Unsupported audio format: {request.audioFormat}"}

    # Decode Base64
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
        audio_file = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_file, sr=22050)  # standard sampling rate
    except Exception as e:
        return {"status": "error", "message": f"Audio decoding error: {str(e)}"}

    # Analyze
    classification, confidence, explanation = analyze_voice(y, sr)

    # Return response
    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
