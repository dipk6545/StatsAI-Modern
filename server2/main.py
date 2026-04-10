"""
StatsAI v3.0 Backend (Server2)
Leverages the MistralEngine for Reasoning, Vision, OCR, and Speech.
"""

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, os, json, logging, base64
from pathlib import Path
from mistral_engine import MistralEngine

# ── INITIALIZATION ────────────────────────────────────────────────────────────
app = FastAPI(title="StatsAI Deep Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = MistralEngine()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Server2")

# ── ENDPOINTS ──────────────────────────────────────────────────────────────────

@app.get("/api/config")
async def api_config():
    """Report available capabilities."""
    return {
        "models": ["Mistral Reasoning", "Mistral Vision", "Mistral OCR"],
        "features": ["reasoning", "vision", "ocr", "speech"]
    }

@app.post("/api/chat")
async def api_chat(
    message:  str = Form(""),
    history:  str = Form("[]"),
    image:    str = Form(None), # Base64 string
    reasoning: bool = Form(False)
):
    try:
        hist = json.loads(history)
        result = await engine.chat(
            message=message,
            history=hist,
            image_base64=image,
            reasoning=reasoning
        )
        
        if result.get("success"):
            return {"reply": result["reply"], "model": result["model"]}
        else:
            return {"error": result.get("error", "Unknown API Error")}, 500
            
    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return {"error": str(e)}, 500

@app.post("/api/ocr")
async def api_ocr(image: str = Form(None)):
    """Extract data from image via Mistral OCR."""
    if not image:
        return {"error": "No image provided"}, 400
        
    result = await engine.extract_data_ocr(image_base64=image)
    if result.get("success"):
        return {"markdown": result["markdown"]}
    return {"error": result.get("error")}, 500

@app.post("/api/speech")
async def api_speech(text: str = Form(...)):
    """Convert analyst response to audio."""
    result = await engine.generate_audio(text=text)
    if result.get("success"):
        return {"audio_base64": result["audio_base64"]}
    return {"error": result.get("error")}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3002) # Port 3002 for Server2
