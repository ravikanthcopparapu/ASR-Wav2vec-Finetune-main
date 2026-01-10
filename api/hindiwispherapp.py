from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import tempfile
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Whisper ASR API")
# Add CORS middleware - THIS IS CRITICAL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# ============ CONFIG ============
MODEL_DIR = "C:/Users/Ravi/Videos/AST/ASRWorkSpace/ASR-Wav2vec-Finetune-main/final_whisper_model"   # path to your model folder
LANGUAGE = "hi"                     # Hindi
DEVICE = -1                         # -1 = CPU, 0 = GPU
# ================================

print("Loading Whisper model...")

asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_DIR,
    chunk_length_s=30,
    device=DEVICE,
    framework="pt"   # 👈 FORCE PYTORCH, IGNORE TENSORFLOW
)

# Force Hindi transcription
asr_pipeline.model.config.forced_decoder_ids = (
    asr_pipeline.tokenizer.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")
)

print("Model loaded successfully!")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and get transcription text.
    """

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    try:
        result = asr_pipeline(temp_audio_path)
        text = result["text"]
    finally:
        os.remove(temp_audio_path)

    return {
        "filename": file.filename,
        "transcription": text
    }


@app.post("/translate")
async def translate_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and get English translation.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    try:
        # Switch model to translation mode
        asr_pipeline.model.config.forced_decoder_ids = (
            asr_pipeline.tokenizer.get_decoder_prompt_ids(language=LANGUAGE, task="translate")
        )

        result = asr_pipeline(temp_audio_path)
        text = result["text"]
    finally:
        os.remove(temp_audio_path)

    return {
        "filename": file.filename,
        "translation": text
    }