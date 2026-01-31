from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from transformers import pipeline
import tempfile
import shutil
import os
import json
from fastapi.middleware.cors import CORSMiddleware
import librosa
import soundfile as sf
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Whisper ASR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ CONFIG ============
MODEL_DIR = "C:/Users/Ravi/Videos/AST/ASRWorkSpace/ASR-Wav2vec-Finetune-main/final_whisper_model"
LANGUAGE = "hi"
DEVICE = -1  # CPU
CHUNK_LENGTH_SECONDS = 2
# ================================

print("Loading Whisper model...")
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_DIR,
    chunk_length_s=30,
    device=DEVICE,
    framework="pt"
)

asr_pipeline.model.config.forced_decoder_ids = (
    asr_pipeline.tokenizer.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")
)
print("Model loaded successfully!")

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=1)


def transcribe_chunk(chunk_audio, sr, chunk_index, start_time, end_time):
    """Transcribe a single chunk - runs in thread pool"""
    try:
        # Create temporary file for chunk
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            chunk_path = tmp.name
        
        sf.write(chunk_path, chunk_audio, sr)
        
        # Transcribe
        result = asr_pipeline(chunk_path)
        text = result["text"].strip()
        
        # Clean up
        os.remove(chunk_path)
        
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
            "transcription": f"[Error: {str(e)}]",
            "is_final": False
        }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Standard transcription - entire file at once"""
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


@app.post("/transcribe-chunked")
async def transcribe_audio_chunked(file: UploadFile = File(...)):
    """
    Optimized chunked transcription with immediate streaming.
    Processes and sends results as soon as each chunk is ready.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    async def generate_chunks():
        try:
            # Load audio
            audio, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
            chunk_samples = int(CHUNK_LENGTH_SECONDS * sr)
            
            # Process chunks immediately
            chunk_index = 0
            for start_sample in range(0, len(audio), chunk_samples):
                end_sample = min(start_sample + chunk_samples, len(audio))
                chunk_audio = audio[start_sample:end_sample]
                
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                # Transcribe this chunk immediately (in thread to avoid blocking)
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
                
                # Mark last chunk
                if end_sample >= len(audio):
                    chunk_result["is_final"] = True
                
                # Send immediately
                yield f"data: {json.dumps(chunk_result)}\n\n"
                
                chunk_index += 1
                
        finally:
            os.remove(temp_audio_path)
    
    return StreamingResponse(
        generate_chunks(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/translate")
async def translate_audio(file: UploadFile = File(...)):
    """Translation endpoint"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    try:
        asr_pipeline.model.config.forced_decoder_ids = (
            asr_pipeline.tokenizer.get_decoder_prompt_ids(language=LANGUAGE, task="translate")
        )
        result = asr_pipeline(temp_audio_path)
        text = result["text"]
        
        # Reset to transcription mode
        asr_pipeline.model.config.forced_decoder_ids = (
            asr_pipeline.tokenizer.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")
        )
    finally:
        os.remove(temp_audio_path)

    return {
        "filename": file.filename,
        "translation": text
    }