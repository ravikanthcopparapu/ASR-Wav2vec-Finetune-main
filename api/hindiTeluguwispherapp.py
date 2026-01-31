from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import tempfile
import shutil
import os

app = FastAPI(title="Whisper Multilingual ASR API (HI + TE)")

# ================== CORS ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== CONFIG ==================
MODEL_DIR = r"C:\Users\Ravi\Videos\AST\ASRWorkSpace\ASR-Wav2vec-Finetune-main\final_whisper_model_HN_TEL"
DEVICE = -1   # CPU (-1), GPU (0)
CHUNK_LENGTH = 30
# ============================================

print("Loading Whisper multilingual model...")

asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_DIR,
    chunk_length_s=CHUNK_LENGTH,
    device=DEVICE,
    framework="pt",  # force PyTorch
)

print("Model loaded successfully!")

# =====================================================
# HELPERS
# =====================================================

def force_language(lang: str, task: str = "transcribe"):
    """
    Force Whisper decoder language and task.
    """
    asr_pipeline.model.config.forced_decoder_ids = (
        asr_pipeline.tokenizer.get_decoder_prompt_ids(
            language=lang,
            task=task
        )
    )

def reset_language():
    """
    Enable auto language detection.
    """
    asr_pipeline.model.config.forced_decoder_ids = None


# =====================================================
# ENDPOINTS
# =====================================================

@app.post("/transcribe")
async def transcribe_auto(file: UploadFile = File(...)):
    """
    Auto language detection (Hindi / Telugu).
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    try:
        reset_language()  # 🔥 auto-detect
        result = asr_pipeline(temp_audio_path)
    finally:
        os.remove(temp_audio_path)

    return {
        "filename": file.filename,
        "language": result.get("language", "auto"),
        "transcription": result["text"]
    }


@app.post("/transcribe/{lang}")
async def transcribe_forced(
    lang: str,
    file: UploadFile = File(...)
):
    """
    Force language transcription.
    lang = hi | te
    """

    if lang not in ["hi", "te"]:
        return {"error": "Supported languages: hi, te"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    try:
        force_language(lang, task="transcribe")
        result = asr_pipeline(temp_audio_path)
    finally:
        os.remove(temp_audio_path)

    return {
        "filename": file.filename,
        "language": lang,
        "transcription": result["text"]
    }


@app.post("/translate/{lang}")
async def translate_to_english(
    lang: str,
    file: UploadFile = File(...)
):
    """
    Translate Hindi/Telugu speech to English.
    lang = hi | te
    """

    if lang not in ["hi", "te"]:
        return {"error": "Supported languages: hi, te"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    try:
        force_language(lang, task="translate")
        result = asr_pipeline(temp_audio_path)
    finally:
        os.remove(temp_audio_path)

    return {
        "filename": file.filename,
        "source_language": lang,
        "translation": result["text"]
    }
