
import subprocess
import sounddevice as sd
import queue
import vosk
import json

def recognize_speech_mac():
    model_path = "vosk-model-small-en-us-0.15"
    q = queue.Queue()
    samplerate = 16000
    device = None  # Use default input device
    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, samplerate)
    print("Say something...")

    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device, dtype='int16', channels=1, callback=callback):
        print("Listening...")
        result = ""
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                result = res.get("text", "")
                break
        print(f"You said: {result}")
        return result if result else None

def speak_mac(text):
    if text:
        subprocess.run(["say", text])
    else:
        print("No text to speak.")

def main():
    text = recognize_speech_mac()
    speak_mac(text)

if __name__ == "__main__":
    main()
