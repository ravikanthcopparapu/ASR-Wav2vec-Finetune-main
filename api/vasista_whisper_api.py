import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
import shutil

# ==========================
# CONFIG
# ==========================

MODEL_ID = "vasista22/whisper-hindi-small"   # HuggingFace model
LANGUAGE = "hi"                             # Hindi
DEVICE = -1                                # -1 = CPU, 0 = GPU

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==========================
# APP INIT
# ==========================

app = FastAPI(
    title="Whisper Hindi ASR API",
    description="Upload an audio file and get Hindi transcription using Whisper",
    version="1.0.0"
)

print("Loading Whisper model from Hugging Face:", MODEL_ID)

asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_ID,
    device=DEVICE,
    framework="pt"  # Force PyTorch (avoid TensorFlow/Keras issues)
)

# Force language for decoding
asr_pipeline.model.config.forced_decoder_ids = (
    asr_pipeline.tokenizer.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")
)

print("Model loaded successfully!")

# ==========================
# ROUTES
# ==========================

@app.get("/")
def root():
    return {"message": "Whisper Hindi ASR API is running!"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and get transcription
    Supported formats: wav, mp3, flac, m4a (via ffmpeg)
    """

    # Validate file type
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Save file temporarily
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_audio_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run transcription
        result = asr_pipeline(temp_audio_path)

        return {
            "filename": file.filename,
            "transcription": result["text"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
