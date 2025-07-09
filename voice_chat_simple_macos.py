import os
import json
import requests
import speech_recognition as sr
import tempfile
import subprocess
import time

# Settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:1b"

# Initialize speech recognition
recognizer = sr.Recognizer()

def get_microphone():
    """Get the default microphone, handling potential issues"""
    try:
        return sr.Microphone()
    except Exception as e:
        print(f"Error initializing microphone: {e}")
        print("Make sure you have given microphone permissions to your terminal/Python")
        return None

def listen_and_transcribe_macos():
    """Use macOS built-in speech recognition via SpeechRecognition library"""
    mic = get_microphone()
    if not mic:
        return ""
    
    try:
        print("Listening... (speak now)")
        with mic as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Listen for audio with timeout
            audio = recognizer.listen(source, timeout=2, phrase_time_limit=10)
        
        print("Processing speech...")
        
        # Try different recognition engines in order of preference
        try:
            # Try Whisper API if available (most accurate)
            text = recognizer.recognize_whisper_api(audio)
            print("[STT] Used Whisper API")
            return text.strip()
        except (sr.RequestError, sr.UnknownValueError):
            pass
        
        try:
            # Try Google Speech Recognition (requires internet)
            text = recognizer.recognize_google(audio)
            print("[STT] Used Google Speech Recognition")
            return text.strip()
        except (sr.RequestError, sr.UnknownValueError):
            pass
        
        try:
            # Try CMU Sphinx (offline, included with speech_recognition)
            text = recognizer.recognize_sphinx(audio)
            print("[STT] Used CMU Sphinx")
            return text.strip()
        except (sr.RequestError, sr.UnknownValueError):
            pass
        
        print("Could not understand audio with any recognition engine")
        return ""
        
    except sr.WaitTimeoutError:
        print("No speech detected within timeout period")
        return ""
    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""

def listen_and_transcribe_applescript():
    """Alternative approach using AppleScript for macOS speech recognition"""
    try:
        print("Starting dictation... (speak now)")
        
        # Use AppleScript to start dictation
        script = '''
        tell application "System Events"
            -- This would open dictation but requires user interaction
            key code 100 using {function down, function down}
        end tell
        '''
        
        # This approach requires more complex setup and user permissions
        # For now, we'll stick with the speech_recognition library
        return listen_and_transcribe_macos()
        
    except Exception as e:
        print(f"AppleScript STT Error: {e}")
        return ""

# Maintain conversation history for context
conversation_history = []

def query_ollama(prompt):
    """Send prompt to Ollama and get response"""
    conversation_history.append(("user", prompt))
    
    # Build context from conversation history
    context = ""
    for role, message in conversation_history:
        if role == "user":
            context += f"User: {message}\n"
        else:
            context += f"Assistant: {message}\n"
    
    try:
        payload = {"model": MODEL, "prompt": context, "stream": False}
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        reply = response.json()["response"]
        conversation_history.append(("assistant", reply))
        return reply
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return "Sorry, I encountered an error processing your request."

def speak_macos(text):
    """Use macOS built-in 'say' command for text-to-speech"""
    if not text or len(text.strip()) < 2:
        return
    
    # Clean text for TTS
    cleaned_text = text.replace('"', '\\"').replace("'", "\\'")
    
    try:
        # Use macOS say command with a pleasant voice
        # You can change the voice with -v parameter: say -v Samantha "text"
        os.system(f'say "{cleaned_text}"')
    except Exception as e:
        print(f"TTS Error: {e}")

def speak_advanced_macos(text):
    """Advanced TTS with voice selection"""
    if not text or len(text.strip()) < 2:
        return
    
    # Available voices: Samantha, Alex, Victoria, Fred, etc.
    # You can list all voices with: say -v "?"
    voice = "Samantha"  # Change this to your preferred voice
    rate = 200  # Words per minute (default is around 175)
    
    cleaned_text = text.replace('"', '\\"').replace("'", "\\'")
    
    try:
        subprocess.run(['say', '-v', voice, '-r', str(rate), cleaned_text], check=True)
    except Exception as e:
        print(f"Advanced TTS Error: {e}")
        # Fallback to simple say command
        speak_macos(text)

def test_microphone():
    """Test microphone access and speech recognition"""
    print("Testing microphone and speech recognition...")
    
    mic = get_microphone()
    if not mic:
        print("❌ Microphone initialization failed")
        return False
    
    try:
        with mic as source:
            print("📱 Say something to test the microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        
        print("🔄 Processing...")
        text = recognizer.recognize_google(audio)
        print(f"✅ Recognized: '{text}'")
        return True
        
    except sr.WaitTimeoutError:
        print("⏰ No speech detected")
        return False
    except sr.UnknownValueError:
        print("❓ Could not understand audio")
        return False
    except sr.RequestError as e:
        print(f"❌ Could not request results: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🎙️  macOS Voice Chat with Ollama")
    print("=" * 40)
    print("Features:")
    print("• macOS native speech recognition")
    print("• macOS 'say' command for TTS")  
    print("• Conversation history")
    print("• Multiple recognition engines")
    print()
    
    # Test microphone first
    if not test_microphone():
        print("\n⚠️  Microphone test failed. Please check:")
        print("1. Microphone permissions for Terminal/Python")
        print("2. System Preferences > Security & Privacy > Microphone")
        print("3. Make sure your microphone is working")
        return
    
    print("\n🚀 Starting voice chat...")
    print("💡 Tip: Speak clearly and pause when finished")
    print("⌨️  Press Ctrl+C to exit\n")
    
    while True:
        try:
            user_text = listen_and_transcribe_macos()
            
            if user_text:
                print(f"👤 You: {user_text}")
                
                # Handle exit commands
                if any(word in user_text.lower() for word in ['exit', 'quit', 'goodbye', 'bye']):
                    print("🤖 Assistant: Goodbye!")
                    speak_macos("Goodbye!")
                    break
                
                response = query_ollama(user_text)
                print(f"🤖 Assistant: {response}")
                speak_macos(response)
            else:
                print("🔇 No speech detected. Try again...")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            speak_macos("Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
