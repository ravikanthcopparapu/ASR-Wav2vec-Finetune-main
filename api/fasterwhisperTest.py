from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import shutil
import os

app = FastAPI()

# Load model once at startup (CPU optimized)
model_size = "small"
model = WhisperModel(
    model_size,
    device="cpu",
    compute_type="int8"   # Best for CPU
)

@app.post("/transcribe-translate/")
async def transcribe_translate(audio: UploadFile = File(...)):
    """
    Accepts an audio file and:
    - Detects language
    - Transcribes speech
    - Translates to English
    """

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        shutil.copyfileobj(audio.file, temp_audio)
        temp_audio_path = temp_audio.name

    try:
        # Run transcription + translation
        segments, info = model.transcribe(
            temp_audio_path,
            beam_size=5,
            task="translate"   # <-- This enables translation to English
        )

        result = {
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "segments": []
        }

        for segment in segments:
            result["segments"].append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text
            })

        return result

    finally:
        # Cleanup temp file
        os.remove(temp_audio_path)
