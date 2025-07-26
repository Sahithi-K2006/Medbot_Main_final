import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from medical_rag import MedicalRAGPipeline
import logging
from typing import Optional, Dict
import numpy as np
import time
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioHandler:
    def __init__(self):
        """Initialize AudioHandler with error handling"""
        self.recognizer = sr.Recognizer()
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Initialized temporary directory at {self.temp_dir}")

    def process_audio_data(self, audio_bytes: bytes) -> Optional[str]:
        """Process audio data from audio-recorder-streamlit or uploaded file"""
        if not audio_bytes:
            return None

        try:
            # Save audio bytes to temporary file
            temp_file = os.path.join(self.temp_dir, "temp_audio.wav")
            with open(temp_file, 'wb') as f:
                f.write(audio_bytes)

            # Transcribe the audio
            with sr.AudioFile(temp_file) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                logger.info("Successfully transcribed audio")
                return text

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech and return bytes"""
        if not text:
            return None

        try:
            temp_file = os.path.join(self.temp_dir, "temp_speech.mp3")
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file)

            with open(temp_file, 'rb') as f:
                audio_bytes = f.read()

            return audio_bytes

        except Exception as e:
            logger.error(f"Error converting text to speech: {e}")
            return None
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def cleanup(self):
        """Clean up temporary directory"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")

class MedicalAssistantApp:
    def __init__(self):
        self.initialize_session_state()
        self.setup_sidebar()
        self.setup_styles()

    def setup_styles(self):
        """Setup custom CSS styles"""
        st.markdown("""
        <style>
        .chat-container {
            max-width: 800px;
            margin: auto;
        }
        .chat-message {
            padding: 15px;
            border-radius: 8px;
            margin: 5px 0;
            font-size: 16px;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        .user-message {
            background-color: #4CAF50;
            color: white;
            text-align: right;
            justify-content: flex-end;
        }
        .assistant-message {
            background-color: #2196F3;
            color: white;
            text-align: left;
            justify-content: flex-start;
        }
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'audio_handler' not in st.session_state:
            st.session_state.audio_handler = AudioHandler()
        
        if 'pipeline' not in st.session_state:
            try:
                st.session_state.pipeline = MedicalRAGPipeline(
                    gemini_api_key= os.getenv("GEMINI_API_KEY")  # Replace with your API key
                )
                documents = st.session_state.pipeline.load_diseases_data("diseases.json")
                st.session_state.intents_data = st.session_state.pipeline.load_intents_data("intents.json")
                st.session_state.pipeline.create_index(documents)
            except Exception as e:
                logger.error(f"Error initializing pipeline: {e}")
                st.error("Error initializing the medical assistant. Please try refreshing the page.")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'recording_duration' not in st.session_state:
            st.session_state.recording_duration = 10

        if 'audio_responses' not in st.session_state:
            st.session_state.audio_responses = {}

        # Add state for audio processing
        if 'processing_audio' not in st.session_state:
            st.session_state.processing_audio = False
            
        if 'current_audio_data' not in st.session_state:
            st.session_state.current_audio_data = None

    def setup_sidebar(self):
        """Setup sidebar with chat history"""
        with st.sidebar:
            st.title("Chat History")
            for msg in st.session_state.chat_history:
                with st.container():
                    role_class = "user-message" if msg['role'] == "user" else "assistant-message"
                    avatar = "🧑‍💼" if msg['role'] == "user" else "🤖"
                    st.markdown(
                        f'<div class="chat-message {role_class}"><span class="avatar">{avatar}</span>{msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                    if 'audio' in msg:
                        st.audio(msg['audio'], format='audio/mp3')
                st.markdown("---")

    def add_to_chat_history(self, role: str, content: str, audio: bytes = None):
        """Add message to chat history"""
        message = {'role': role, 'content': content, 'timestamp': datetime.now()}
        if audio:
            message['audio'] = audio
        st.session_state.chat_history.append(message)

    def run(self):
        st.title("🩺 Medical Assistant")
        st.write("Ask medical questions through text, voice, or uploaded audio.")
        
        # Create tabs
        text_tab, audio_tab, upload_tab = st.tabs(["Text Input", "Audio Recording", "Audio Upload"])

        # Text Input Tab
        with text_tab:
            self.handle_text_input()

        # Audio Recording Tab
        with audio_tab:
            self.handle_audio_recording()

        # Audio Upload Tab
        with upload_tab:
            self.handle_audio_upload()

        # Clear chat button
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.audio_responses = {}
            st.rerun()  # Changed from experimental_rerun

        # Add disclaimer
        st.markdown("---")
        st.markdown("""
        **Disclaimer**: This medical assistant is for informational purposes only. 
        Always consult healthcare professionals for medical advice, diagnosis, or treatment.
        """)

    def handle_text_input(self):
        """Handle text input interactions"""
        text_query = st.text_input("Enter your medical question:", key="text_query")
        if st.button("Send", key="text_button"):
            if text_query:
                try:
                    # Add user message to chat
                    self.add_to_chat_history('user', text_query)
                    
                    # Generate response
                    response = st.session_state.pipeline.generate_response(
                        text_query, 
                        st.session_state.intents_data
                    )
                    
                    # Convert response to speech
                    audio_bytes = st.session_state.audio_handler.text_to_speech(response)
                    
                    # Add assistant response to chat
                    self.add_to_chat_history('assistant', response, audio_bytes)
                    
                    # Force refresh
                    st.rerun()  # Changed from experimental_rerun
                    
                except Exception as e:
                    logger.error(f"Error processing text input: {e}")
                    st.error("Error generating response. Please try again.")

    def handle_audio_recording(self):
        """Handle audio recording with timer and explicit processing button"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Click the microphone to start recording")
            audio_bytes = audio_recorder(
                key="audio_recorder",
                pause_threshold=st.session_state.recording_duration
            )

        with col2:
            # Recording duration selector
            st.session_state.recording_duration = st.number_input(
                "Recording duration (seconds)",
                min_value=5,
                max_value=60,
                value=st.session_state.recording_duration,
                step=5
            )

        # Store audio bytes in session state if new recording is made
        if audio_bytes:
            st.session_state.current_audio_data = audio_bytes
            st.audio(audio_bytes, format='audio/wav')
            st.write("Recording captured! Click 'Process Audio' to analyze.")

        # Add process button
        if st.button("Process Audio", key="process_recording") and st.session_state.current_audio_data:
            try:
                # Process the audio
                text = st.session_state.audio_handler.process_audio_data(st.session_state.current_audio_data)
                if text:
                    # Add user message to chat
                    self.add_to_chat_history('user', text, st.session_state.current_audio_data)
                    
                    # Generate response
                    response = st.session_state.pipeline.generate_response(
                        text,
                        st.session_state.intents_data
                    )
                    
                    # Convert response to speech
                    audio_response = st.session_state.audio_handler.text_to_speech(response)
                    
                    # Add assistant response to chat
                    self.add_to_chat_history('assistant', response, audio_response)
                    
                    # Clear the current audio data after processing
                    st.session_state.current_audio_data = None
                    st.rerun()
                else:
                    st.error("Could not understand the audio. Please try again.")
            except Exception as e:
                logger.error(f"Error processing audio input: {e}")
                st.error("Error processing audio. Please try again.")

    def handle_audio_upload(self):
        """Handle audio file upload"""
        uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])
        
        if uploaded_file is not None and st.button("Process Audio"):
            try:
                # Create temporary file
                temp_file = os.path.join(st.session_state.audio_handler.temp_dir, f"temp_{uploaded_file.name}")
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process the audio using speech recognition
                with sr.AudioFile(temp_file) as source:
                    audio_data = st.session_state.audio_handler.recognizer.record(source)
                    text = st.session_state.audio_handler.recognizer.recognize_google(audio_data)

                if text:
                    # Add user message to chat
                    with open(temp_file, 'rb') as f:
                        audio_bytes = f.read()
                    self.add_to_chat_history('user', text, audio_bytes)
                    
                    # Generate response
                    response = st.session_state.pipeline.generate_response(
                        text,
                        st.session_state.intents_data
                    )
                    
                    # Convert response to speech
                    audio_response = st.session_state.audio_handler.text_to_speech(response)
                    
                    # Add assistant response to chat
                    self.add_to_chat_history('assistant', response, audio_response)
                    
                    # Force refresh
                    st.rerun()  # Changed from experimental_rerun
                else:
                    st.error("Could not understand the audio. Please try again.")

            except Exception as e:
                logger.error(f"Error processing uploaded audio: {e}")
                st.error("Error processing audio file. Please try again.")
            finally:
                # Cleanup temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

if __name__ == "__main__":
    app = MedicalAssistantApp()
    app.run()
