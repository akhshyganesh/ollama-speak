import queue
import sounddevice as sd
import sys
import json
import requests
from TTS.api import TTS
import os
import numpy as np
from faster_whisper import WhisperModel
import tempfile

# Settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:1b"
q = queue.Queue()

# Enhanced Whisper model for better accuracy
WHISPER_MODEL_SIZE = "medium"  # Try "medium" or "large" for even better accuracy (requires more RAM)
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def listen_and_transcribe():
    import collections
    import noisereduce as nr
    samplerate = 16000
    chunk_duration = 0.5  # seconds
    silence_limit = 2  # seconds of silence before stopping (faster response)
    silence_threshold = 400  # more robust to background noise
    max_chunks = int(30 / chunk_duration)
    silence_chunks = int(silence_limit / chunk_duration)
    print("Speak now (pause to finish)...")
    try:
        audio_chunks = []
        silent_chunks = 0
        for _ in range(max_chunks):
            chunk = sd.rec(int(samplerate * chunk_duration), samplerate=samplerate, channels=1, dtype='int16')
            sd.wait()
            audio_chunks.append(chunk)
            # Simple silence detection
            if np.abs(chunk).mean() < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks >= silence_chunks:
                break
        audio = np.concatenate(audio_chunks, axis=0)
        # Noise reduction (convert to float for noisereduce)
        audio_float = audio.astype(np.float32).flatten()
        reduced_noise = nr.reduce_noise(y=audio_float, sr=samplerate)
        # Convert back to int16 for saving
        reduced_noise_int16 = np.int16(reduced_noise / np.max(np.abs(reduced_noise)) * 32767)
        # Save to temp wav file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            import wave
            wf = wave.open(tmpfile.name, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(reduced_noise_int16.tobytes())
            wf.close()
            segments, info = whisper_model.transcribe(tmpfile.name)
            text = " ".join([segment.text for segment in segments])
        os.unlink(tmpfile.name)
        return text.strip()
    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""


# Maintain conversation history for context
conversation_history = []  # List of (role, message) tuples

def query_ollama(prompt):
    # Add the new user message to the history
    conversation_history.append(("user", prompt))
    # Build the context string
    context = ""
    for role, message in conversation_history:
        if role == "user":
            context += f"User: {message}\n"
        else:
            context += f"Assistant: {message}\n"
    # Send the context as the prompt
    payload = {"model": MODEL, "prompt": context, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    reply = response.json()["response"]
    # Add the assistant's reply to the history
    conversation_history.append(("assistant", reply))
    return reply

import re

def clean_text(text):
    # Remove emojis and unsupported punctuation (keep basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9 .,?!'\"]+", '', text)
    # Replace curly quotes with straight quotes
    text = text.replace('’', "'").replace('“', '"').replace('”', '"')
    return text.strip()

import threading
import sounddevice as sd
import soundfile as sf

playback_thread = None
stop_playback = threading.Event()

def play_audio_interruptible(filename):
    data, samplerate = sf.read(filename, dtype='float32')
    sd.play(data, samplerate)
    sd.wait()  # Wait until playback is finished

def stop_audio_playback():
    stop_playback.set()
    sd.stop()
    if playback_thread and playback_thread.is_alive():
        playback_thread.join(timeout=1)

def speak(text):
    text = clean_text(text)
    if len(text.split()) < 2:
        print("[TTS] Skipping very short or empty response.")
        return
    # Use a more natural TTS model (multi-speaker)
    global tts, tts_speaker
    if 'tts' not in globals():
        tts = TTS(model_name="tts_models/en/vctk/vits")
        tts_speaker = tts.speakers[0]  # Default to first speaker
        print(f"[TTS] Using speaker: {tts_speaker}")
    tts.tts_to_file(text=text, speaker=tts_speaker, file_path="output.wav")
    play_audio_interruptible("output.wav")

def main():
    print("Local Voice Chat with TinyLlama (Ollama)")
    while True:
        try:
            stop_audio_playback()  # Stop any ongoing playback before listening
            user_text = listen_and_transcribe()
            print(f"You: {user_text}")
            response = query_ollama(user_text)
            print(f"TinyLlama: {response}")
            speak(response)
        except KeyboardInterrupt:
            print("\nExiting...")
            stop_audio_playback()
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
