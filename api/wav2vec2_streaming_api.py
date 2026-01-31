import os
import json
import shutil
import tempfile
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import librosa
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import Wav2Vec2Processor, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC
from safetensors.torch import load_file


# ==========================================================
# CONFIG
# ==========================================================
HF_DIR = "C:/Users/Ravi/Videos/AST/ASRWorkSpace/ASR-Wav2vec-Finetune-main/huggingface-hub"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000
CHUNK_LENGTH_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH_SECONDS


# ==========================================================
# LOAD MODEL (ONCE)
# ==========================================================
print("🔹 Loading processor...")
processor = Wav2Vec2Processor.from_pretrained(HF_DIR)

print("🔹 Loading config...")
config = Wav2Vec2Config.from_pretrained(HF_DIR)

print("🔹 Initializing model...")
model = Wav2Vec2ForCTC(config)

print("🔹 Loading SafeTensors...")
state_dict = load_file(os.path.join(HF_DIR, "model.safetensors"))
model.load_state_dict(state_dict, strict=False)

model.to(DEVICE)
model.eval()
print("✅ Wav2Vec2 model loaded")


# ==========================================================
# FASTAPI APP
# ==========================================================
app = FastAPI(title="Wav2Vec2 Real-Time Streaming ASR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single worker = stable, low RAM
executor = ThreadPoolExecutor(max_workers=1)


# ==========================================================
# TRANSCRIBE SINGLE CHUNK (THREAD SAFE)
# ==========================================================
def transcribe_chunk(
    chunk_audio,
    sr,
    chunk_index,
    start_time,
    end_time
):
    try:
        # Normalize
        chunk_audio = chunk_audio / max(1e-7, abs(chunk_audio).max())

        inputs = processor(
            chunk_audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=False
        )

        with torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE)).logits

        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

        # -------- CTC collapse --------
        CTC_BLANK_ID = 0
        collapsed = []
        prev = None

        for idx in pred_ids:
            if idx == CTC_BLANK_ID:
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

        text = " ".join(raw_text.split("|"))
        text = " ".join(text.split())

        return {
            "chunk_index": chunk_index,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "transcription": text,
            "is_final": False
        }

    except Exception as e:
        return {
            "chunk_index": chunk_index,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "transcription": f"[ERROR: {str(e)}]",
            "is_final": False
        }


# ==========================================================
# STREAMING CHUNKED TRANSCRIPTION (REAL-TIME PACED)
# ==========================================================
@app.post("/transcribe-chunked")
async def transcribe_audio_chunked(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        return {"error": "Only WAV files supported"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    async def generate_chunks():
        try:
            audio, sr = librosa.load(temp_audio_path, sr=SAMPLE_RATE, mono=True)
            chunk_index = 0

            for start_sample in range(0, len(audio), CHUNK_SAMPLES):
                end_sample = min(start_sample + CHUNK_SAMPLES, len(audio))
                chunk_audio = audio[start_sample:end_sample]

                start_time = start_sample / sr
                end_time = end_sample / sr
                expected_duration = end_time - start_time

                wall_start = time.time()

                loop = asyncio.get_event_loop()
                chunk_result = await loop.run_in_executor(
                    executor,
                    transcribe_chunk,
                    chunk_audio,
                    sr,
                    chunk_index,
                    start_time,
                    end_time
                )

                processing_time = time.time() - wall_start

                # 🕒 REAL-TIME PACING FIX
                if processing_time < expected_duration:
                    await asyncio.sleep(expected_duration - processing_time)

                if end_sample >= len(audio):
                    chunk_result["is_final"] = True

                yield f"data: {json.dumps(chunk_result, ensure_ascii=False)}\n\n"
                chunk_index += 1

        finally:
            os.remove(temp_audio_path)

    return StreamingResponse(
        generate_chunks(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ==========================================================
# STANDARD FULL FILE TRANSCRIPTION (OPTIONAL)
# ==========================================================
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        return {"error": "Only WAV files supported"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    try:
        audio, sr = librosa.load(temp_audio_path, sr=SAMPLE_RATE, mono=True)
        result = transcribe_chunk(audio, sr, 0, 0, len(audio) / sr)
        return {
            "filename": file.filename,
            "transcription": result["transcription"]
        }
    finally:
        os.remove(temp_audio_path)
