from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile, shutil, os
import torch

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
from peft import PeftModel

# ================= APP =================
app = FastAPI(title="Whisper Hindi ASR API (LoRA)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CONFIG =================
BASE_MODEL = "openai/whisper-medium"
LORA_DIR = r"C:/Users/Ravi/Videos/AST/ASRWorkSpace/ASR-Wav2vec-Finetune-main/whisper_medium_lora"
LANGUAGE = "hi"
DEVICE = -1  # CPU
CHUNK_LENGTH = 15
# ========================================

print("Loading Whisper processor...")
processor = WhisperProcessor.from_pretrained(
    BASE_MODEL,
    language=LANGUAGE,
    task="transcribe"
)

print("Loading base Whisper model (CPU)...")
base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

print("Loading LoRA adapter (noise-trained)...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

# 🔥 CRITICAL: force Hindi transcription
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=LANGUAGE,
    task="transcribe"
)
model.config.forced_decoder_ids = forced_decoder_ids
model.config.suppress_tokens = []

# Build pipeline explicitly
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=DEVICE,
    chunk_length_s=CHUNK_LENGTH
)

print("LoRA Whisper model loaded successfully!")

# ================= UTILITY =================

def save_temp_audio(upload_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name

# ================= ENDPOINT =================

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio_path = save_temp_audio(file)

    try:
        result = asr_pipeline(
            audio_path,
            generate_kwargs={
                "task": "transcribe",
                "language": LANGUAGE,
                "temperature": 0.0,
                "num_beams": 5,
                "no_repeat_ngram_size": 3,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6
            }
        )
        text = result["text"].strip()
    finally:
        os.remove(audio_path)

    return {
        "filename": file.filename,
        "transcription": text
    }
