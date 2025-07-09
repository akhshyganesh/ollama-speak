import queue
import sounddevice as sd
import sys
import json
import requests
from TTS.api import TTS
import os
import numpy as np
import tempfile
import speech_recognition as sr

# Settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:1b"
q = queue.Queue()

# Initialize speech recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Adjust for ambient noise on startup
print("Calibrating microphone for ambient noise...")
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)
print("Microphone calibrated.")

def listen_and_transcribe_native():
    """Use macOS built-in speech recognition"""
    try:
        print("Listening... (speak now)")
        with microphone as source:
            # Listen for audio with a timeout
            audio = recognizer.listen(source, timeout=1, phrase_time_limit=10)
        
        print("Processing speech...")
        try:
            # Use macOS built-in speech recognition
            text = recognizer.recognize_whisper_api(audio, api_key=None)  # Uses local Whisper if available
        except sr.RequestError:
            # Fallback to macOS built-in recognition
            try:
                text = recognizer.recognize_sphinx(audio)
            except sr.RequestError:
                # Final fallback - you could also try Google's free tier
                print("Could not request results from speech recognition service")
                return ""
        
        return text.strip()
        
    except sr.WaitTimeoutError:
        print("No speech detected within timeout period")
        return ""
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""

def listen_and_transcribe_macos():
    """Use macOS native speech recognition via system commands"""
    try:
        print("Listening... (speak now)")
        
        # Create a temporary script for macOS speech recognition
        script = '''
        tell application "SpeechRecognitionServer"
            listen continuously with feedback
        end tell
        '''
        
        # Alternative: Use dictation via command line
        # This requires enabling dictation in System Preferences
        result = os.popen('osascript -e "tell application \\"System Events\\" to tell application process \\"VoiceOver Utility\\" to click"').read()
        
        # For now, let's use a simpler approach with speech_recognition library
        return listen_and_transcribe_native()
        
    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""

# Use the original Whisper function as well for comparison
def listen_and_transcribe_whisper():
    """Original Whisper-based transcription for comparison"""
    import collections
    import noisereduce as nr
    from faster_whisper import WhisperModel
    
    # Enhanced Whisper model for better accuracy
    WHISPER_MODEL_SIZE = "medium"
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    
    samplerate = 16000
    chunk_duration = 0.5  # seconds
    silence_limit = 2  # seconds of silence before stopping
    silence_threshold = 400
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
        # Noise reduction
        audio_float = audio.astype(np.float32).flatten()
        reduced_noise = nr.reduce_noise(y=audio_float, sr=samplerate)
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

# Choose which transcription method to use
def listen_and_transcribe():
    """Main transcription function - you can switch between methods here"""
    # Try macOS native first, fallback to Whisper if needed
    text = listen_and_transcribe_native()
    if not text:
        print("Falling back to Whisper transcription...")
        text = listen_and_transcribe_whisper()
    return text

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
    text = text.replace(chr(8217), "'").replace(chr(8220), '"').replace(chr(8221), '"')
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

# Speak function for TTS
def speak(text):
    text = clean_text(text)
    if len(text.split()) < 2:
        print("[TTS] Skipping very short or empty response.")
        return
    
    # Option 1: Use macOS built-in TTS (fastest)
    os.system(f'say {json.dumps(text)}')
    
    # Option 2: Use TTS library (better quality but slower)
    # global tts, tts_speaker
    # if 'tts' not in globals():
    #     tts = TTS(model_name="tts_models/en/vctk/vits")
    #     tts_speaker = tts.speakers[0]
    #     print(f"[TTS] Using speaker: {tts_speaker}")
    # tts.tts_to_file(text=text, speaker=tts_speaker, file_path="output.wav")
    # play_audio_interruptible("output.wav")

def main():
    print("Local Voice Chat with Ollama using macOS Speech Recognition")
    print("Speech recognition methods available:")
    print("1. macOS native speech recognition (primary)")
    print("2. Whisper fallback (if native fails)")
    print("3. macOS 'say' command for TTS")
    print()
    
    while True:
        try:
            stop_audio_playback()  # Stop any ongoing playback before listening
            user_text = listen_and_transcribe()
            
            if user_text:
                print(f"You: {user_text}")
                response = query_ollama(user_text)
                print(f"Assistant: {response}")
                speak(response)
            else:
                print("No speech detected. Try again...")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            stop_audio_playback()
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
