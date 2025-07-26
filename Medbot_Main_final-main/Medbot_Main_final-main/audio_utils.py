import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from gtts import gTTS
import os
from datetime import datetime
import numpy as np
import tempfile
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioHandler:
    def __init__(self, device=None):
        """Initialize AudioHandler with error recovery capabilities"""
        self.recognizer = sr.Recognizer()
        self.device = device
        self.setup_audio_device()
        
        # Create temporary directories for audio files
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Initialized temporary directory at {self.temp_dir}")

    def setup_audio_device(self) -> None:
        """Setup audio device with fallback options"""
        try:
            if self.device is None:
                devices = sd.query_devices()
                default_device = sd.default.device[0]
                self.device = default_device
                logger.info(f"Using default input device: {default_device}")
            
            # Test device
            sd.check_input_settings(device=self.device)
            
        except Exception as e:
            logger.error(f"Error setting up audio device: {e}")
            # Try to find any working input device
            try:
                devices = sd.query_devices()
                for device in devices:
                    try:
                        if device['max_input_channels'] > 0:
                            self.device = device['index']
                            sd.check_input_settings(device=self.device)
                            logger.info(f"Found alternative working device: {self.device}")
                            break
                    except:
                        continue
            except:
                logger.error("No working audio input devices found")
                self.device = None

    def record_audio(self, duration: int = 5, sample_rate: int = 44100) -> Optional[str]:
        """Record audio with error handling and recovery"""
        if self.device is None:
            logger.error("No valid input device available")
            return None

        try:
            logger.info("Starting recording...")
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                device=self.device
            )
            sd.wait()
            
            # Generate temporary file path
            temp_file = os.path.join(self.temp_dir, f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            
            # Normalize audio data
            recording = np.clip(recording, -1.0, 1.0)
            
            sf.write(temp_file, recording, sample_rate)
            logger.info(f"Recording saved to {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            # Try to reinitialize audio device
            self.setup_audio_device()
            return None

    def transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio with multiple fallback options"""
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return None

        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                
                # Try Google Speech Recognition first
                try:
                    text = self.recognizer.recognize_google(audio_data)
                    logger.info("Successfully transcribed audio using Google Speech Recognition")
                    return text
                except sr.RequestError:
                    # Fallback to offline Sphinx recognition if Google fails
                    try:
                        text = self.recognizer.recognize_sphinx(audio_data)
                        logger.info("Successfully transcribed audio using Sphinx")
                        return text
                    except:
                        logger.error("All transcription methods failed")
                        return None
                        
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    def text_to_speech(self, text: str) -> Optional[str]:
        """Convert text to speech with fallback options"""
        if not text:
            logger.error("No text provided for conversion")
            return None

        try:
            # Generate temporary file path
            temp_file = os.path.join(self.temp_dir, f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
            
            # Try gTTS first
            try:
                tts = gTTS(text=text, lang='en')
                tts.save(temp_file)
                logger.info("Successfully generated audio using gTTS")
                return temp_file
            except Exception as e:
                logger.error(f"gTTS failed: {e}")
                # Could implement fallback TTS here if needed
                return None
                
        except Exception as e:
            logger.error(f"Error converting text to speech: {e}")
            return None

    def play_audio(self, audio_file: str) -> bool:
        """Play audio with error handling"""
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return False

        try:
            data, samplerate = sf.read(audio_file)
            sd.play(data, samplerate)
            sd.wait()
            return True
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
