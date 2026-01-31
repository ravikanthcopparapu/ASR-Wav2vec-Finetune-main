from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import tempfile
import os
import numpy as np
import soundfile as sf

app = FastAPI()

# Load model once (CPU optimized, multilingual)
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    audio_buffer = b""

    try:
        while True:
            # Receive raw PCM bytes from browser
            data = await websocket.receive_bytes()
            audio_buffer += data

            # Process every ~3 seconds of audio
            if len(audio_buffer) > 16000 * 2 * 3:  # 16kHz * 2 bytes * 3 sec

                # Convert raw PCM -> numpy array
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16)

                # Save as proper WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    sf.write(tmp.name, audio_np, samplerate=16000)
                    temp_audio_path = tmp.name

                try:
                    # Transcribe with auto language detection
                    segments, info = model.transcribe(
                        temp_audio_path,
                        beam_size=5,
                        vad_filter=True
                    )

                    text_output = ""
                    for segment in segments:
                        text_output += segment.text + " "

                    response = {
                        "detected_language": info.language,
                        "language_probability": round(info.language_probability, 3),
                        "text": text_output.strip()
                    }

                    # Send result back to client
                    await websocket.send_json(response)

                finally:
                    os.remove(temp_audio_path)
                    audio_buffer = b""  # clear buffer

    except WebSocketDisconnect:
        print("Client disconnected")