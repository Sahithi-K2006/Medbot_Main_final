import streamlit as st
from audio_recorder_streamlit import audio_recorder

# Define CSS styles for the pulsating effect
css = """
<style>
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(255, 77, 77, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(255, 77, 77, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(255, 77, 77, 0);
        }
    }
    .pulse {
        animation: pulse 2s infinite;
        border-radius: 50%; /* Makes the border circular */
        height: 50px; /* Adjust as needed */
        width: 50px; /* Adjust as needed */
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .microphone-icon {
        font-size: 2em; /* Adjust the icon size */
        color: #6699ff; /* Blue color */
    }
    .pulsating-microphone {
        position: relative;
        display: inline-block;
    }
    .pulsating-effect {
        position: absolute;
        top: 0;
        left: 0;
    }
</style>
"""

# Display the CSS styles
st.markdown(css, unsafe_allow_html=True)

# Customize the styling parameters for the audio recorder
audio_bytes = audio_recorder(
    text="",
    recording_color="#ff4d4d",  # Red color when recording
    neutral_color="#6699ff",    # Blue color when not recording
    icon_name="microphone-alt",  # Microphone icon
    icon_size="5x"               # Larger icon size
)

# Apply the pulsating effect to the microphone icon
st.markdown("""
    <div class="pulsating-microphone">
        <i class="fas fa-microphone-alt microphone-icon"></i>
        <div class="pulse pulsating-effect"><i class="fas fa-microphone-alt microphone-icon pulse"></i></div>
    </div>
""", unsafe_allow_html=True)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    # Automatically save the audio file with a default name
    default_file_name = "audio.wav"
    with open(default_file_name, "wb") as f:
        f.write(audio_bytes)
    st.success(f"Audio saved as {default_file_name}")
