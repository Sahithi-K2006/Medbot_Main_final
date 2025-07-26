import streamlit as st
import numpy as np
from keras.models import load_model
import pickle
import nltk
import random
from PIL import Image
from nltk.stem import WordNetLemmatizer
from gtts import gTTS
import speech_recognition as sr
import time
import os
import json
from audio_recorder_streamlit import audio_recorder
print("logging")


# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# Load model and data
model = load_model('medical_chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.load(open('intents.json'))
diseases = json.load(open('diseases.json'))
diseases_names = [disease['tag'] for disease in diseases['diseases']]

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Functions for text processing and chatbot logic
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)               
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# def getResponse(ints, diseases_json):
#     if not ints:
#         return "No response found, Query is not related to the Medical Domain !!!"
    
#     tag = ints[0]['intent']
#     list_of_diseases = diseases_json['diseases']
#     for i in list_of_diseases:
#         if(i['tag'] == tag):
#             result = random.choice(i['responses'])
#             break
#     return result

def getResponse(ints, diseases_json):
    if not ints:
        return "No response found, Query is not related to the Medical Domain !!!"

    filter()
    # Initialize result with a default response
    result = "Sorry, I don't have information on that specific condition."

    # for i in list_of_diseases:
    #     if i['tag'] == tag:
    #         result = random.choice(i['responses'])
    #         break  # Exit the loop once a match is found
    

    return result

# Function for converting text to speech
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

# Function for converting speech to text
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")

# Define correct username and password
CORRECT_USERNAME = "medbot"
CORRECT_PASSWORD = "medbot@123"  # Change this to your desired password

# Set title and separator
st.title("Medical Assistance ChatBot")
st.markdown("---")

# Authentication
session_state = st.session_state
if "authenticated" not in session_state:
    session_state.authenticated = False

# Check if the user is authenticated
if not session_state.authenticated:
    # Login section
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == CORRECT_USERNAME and password == CORRECT_PASSWORD:
            session_state.authenticated = True
        else:
            st.error("Incorrect username or password")

# If authenticated, proceed with the application
if session_state.authenticated:
    # Logout button
    if st.button("Logout"):
        session_state.authenticated = False

    # Sidebar options
    option = st.sidebar.radio("Choose Interaction Type", ("Text to Text", "Text to Voice", "Voice to Voice"))

    # Load and display the image
    image = Image.open("image1.png")
    st.image(image, use_container_width=True)
    print("logging",option)
    # Text-to-Text interaction
    if option == "Text to Text":
        user_input = st.text_input("Enter your message:")
        if st.button("Submit"):
            # ints = predict_class(user_input, model)
            # print("logging  :",ints,diseases)
            response = getResponse(user_input.split(), diseases)
            st.success(f"MedBot: {response}")

    # Text-to-Voice interaction
    elif option == "Text to Voice":
        user_input = st.text_input("Enter your message:")
        if st.button("Submit"):
            ints = predict_class(user_input, model)
            response = getResponse(ints, diseases)
            text_to_speech(response, "voice_response.mp3")
            st.audio("voice_response.mp3", format="audio/mp3")

    # Voice-to-Voice interaction
    elif option == "Voice to Voice":
        st.subheader("Speak to the Medical Assistance ChatBot")
        audio_bytes = audio_recorder(
            text="",
            recording_color="#ff4d4d",  # Red color when recording
            neutral_color="#6699ff",    # Blue color when not recording
            icon_name="microphone-alt",  # Microphone icon
            icon_size="5x"               # Larger icon size
        )

        if audio_bytes:
            # Save audio input
            default_file_name = "audio_input.wav"
            with open(default_file_name, "wb") as f:
                f.write(audio_bytes)
            
            st.success(f"Audio saved as {default_file_name}")

            # Convert audio input to text
            user_input = speech_to_text(default_file_name)

            if user_input:
                ints = predict_class(user_input, model)
                response = getResponse(ints, diseases)

                # Generate response as speech
                text_to_speech(response, "voice_response.mp3")

                # Play the response
                st.audio("voice_response.mp3", format="audio/mp3")
