import os
import shutil
import tempfile
import librosa
import torch

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2Processor, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC
from safetensors.torch import load_file


# ==========================================================
# CONFIG
# ==========================================================
HF_DIR = "C:/Users/Ravi/Videos/AST/ASRWorkSpace/ASR-Wav2vec-Finetune-main/ASR_Hindi_English_Model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# Load model ONCE
# ==========================================================
print("🔹 Loading processor...")
processor = Wav2Vec2Processor.from_pretrained(HF_DIR)

print("🔹 Loading model config...")
config = Wav2Vec2Config.from_pretrained(HF_DIR)

print("🔹 Initializing model...")
model = Wav2Vec2ForCTC(config)

print("🔹 Loading SafeTensors weights...")
state_dict = load_file(os.path.join(HF_DIR, "model.safetensors"))
model.load_state_dict(state_dict, strict=False)

model.to(DEVICE)
model.eval()
print("✅ Model loaded")


# ==========================================================
# FastAPI app
# ==========================================================
app = FastAPI(title="MultiLingual ASR API")
# Add CORS middleware - THIS IS CRITICAL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# ==========================================================
# Transcription
# ==========================================================
@torch.no_grad()
def transcribe_wav(wav_path: str) -> str:
    wav, _ = librosa.load(wav_path, sr=16000, mono=True)
    wav = wav / max(1e-7, abs(wav).max())

    inputs = processor(
        wav,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=False
    )

    logits = model(inputs.input_values.to(DEVICE)).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

    # Correct CTC blank token
    CTC_BLANK_ID = 0  # "|"

    collapsed = []
    prev = None
    for idx in pred_ids:
        if idx == CTC_BLANK_ID:
            collapsed.append(idx)   # keep delimiter
            prev = None
            continue
        if idx != prev:
            collapsed.append(idx)
        prev = idx

    raw_text = processor.tokenizer.decode(
        collapsed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Proper word spacing
    text = " ".join(raw_text.split("|"))
    text = " ".join(text.split())

    return text


# ==========================================================
# API Endpoint
# ==========================================================
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        return {"error": "Only .wav files supported"}

    # ✅ Create temp directory (Windows safe)
    temp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(temp_dir, file.filename)

    try:
        # ✅ Save file explicitly
        with open(wav_path, "wb") as f:
            content = await file.read()
            f.write(content)
            f.flush()

        # 🔍 DEBUG: confirm file exists & size
        file_size = os.path.getsize(wav_path)
        print(f"DEBUG: Saved WAV -> {wav_path} ({file_size} bytes)")

        if file_size == 0:
            return {"error": "Uploaded file is empty"}

        transcription = transcribe_wav(wav_path)

        return {
            "filename": file.filename,
            "filesize_bytes": file_size,
            "transcription": transcription
        }

    finally:
        shutil.rmtree(temp_dir)
