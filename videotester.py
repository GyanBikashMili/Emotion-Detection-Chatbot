import os
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from keras.layers import DepthwiseConv2D
from keras.utils import get_custom_objects
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from gtts import gTTS
from playsound import playsound
import tempfile
import threading
from typing import Dict
import google.generativeai as genai

# Custom DepthwiseConv2D to handle 'groups' parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

# Load model
try:
    model = load_model("best_model.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
except Exception as e:
    print(f"Error loading model: {e}")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Predefined messages for each emotion
emotion_messages = {
    'angry': 'You seem upset. How can I help?',
    'disgust': 'Is something bothering you?',
    'fear': 'It looks like you are scared. Everything okay?',
    'happy': 'You look happy! That’s great to see!',
    'sad': 'Why the sad face? Want to talk about it?',
    'surprise': 'You look surprised! What happened?',
    'neutral': 'You seem calm and neutral.'
}

# Initialize Tkinter
root = tk.Tk()
root.title("Emotion-Based Chatbot")
root.geometry("1200x700")
root.configure(bg='#2E3440')  # Dark gray background

# Style configuration
style = ttk.Style()
style.theme_use('clam')

# Configure button styles
style.configure('TButton',
                font=('Helvetica', 12),
                padding=10,
                background='#88C0D0',
                foreground='#2E3440',
                borderwidth=0,
                focusthickness=3,
                focuscolor='none')
style.map('TButton',
          background=[('active', '#81A1C1')],
          foreground=[('active', '#2E3440')],
          relief=[('pressed', 'groove'), ('!pressed', 'ridge')])

# Configure label styles
style.configure('TLabel', font=('Helvetica', 14), background='#2E3440', foreground='#ECEFF4')
style.configure('TFrame', background='#2E3440')
style.configure('TText', background='#3B4252', foreground='#ECEFF4', font=('Helvetica', 12))

# Create frames
video_frame = ttk.Frame(root, width=600, height=650, style='TFrame')
video_frame.pack(side=tk.LEFT, padx=10, pady=10)

chat_frame = ttk.Frame(root, width=600, height=700, style='TFrame')
chat_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Chatbot UI elements
chat_log = tk.Text(chat_frame, state='disabled', height=25, width=50, bg='#3B4252', fg='#ECEFF4', font=('Helvetica', 12), bd=0, highlightthickness=0, padx=10, pady=10)
chat_log.pack(pady=10)

input_emotion = tk.StringVar()
emotion_label = ttk.Label(chat_frame, text="Detected Emotion: None", style='TLabel')
emotion_label.pack(pady=5)

# Frame for user input and send button
input_frame = ttk.Frame(chat_frame, style='TFrame')
input_frame.pack(pady=5)

# Entry field for user input
user_input_field = ttk.Entry(input_frame, width=40, font=('Helvetica', 12))
user_input_field.pack(side=tk.LEFT, padx=5)

# Configure the SDK with your API key
genai.configure(api_key="YOUR GEMINI API")  # Replace with your actual Gemini API key

def play_sound(audio_file: str):
    playsound(audio_file)

def synthesize_speech(text):
    tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_file = fp.name
        tts.save(temp_file)
    play_sound(temp_file)

# Define send_message function
def send_message():
    user_input = user_input_field.get().strip().lower()
    user_input_field.delete(0, tk.END)
    chat_log.config(state='normal')
    chat_log.insert(tk.END, f"You: {user_input}\n")

    # Use detected emotion to create a context-aware prompt
    emotion = detected_emotion
    prompt = f"User is feeling {emotion}. They said: {user_input}. Respond appropriately."

    # Use Gemini API to get bot response
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    bot_response = response.text

    chat_log.insert(tk.END, f"Bot: {bot_response}\n")
    chat_log.config(state='disabled')

    # Synthesize and play the speech
    threading.Thread(target=synthesize_speech, args=(bot_response,), daemon=True).start()

# Send button to send message
send_button = ttk.Button(input_frame, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT, padx=5)

def capture_emotion():
    global detected_emotion
    # Use detected emotion as the prompt
    prompt = f"User is feeling {detected_emotion}. Respond appropriately."

    # Use Gemini API to get bot response
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    bot_response = response.text

    chat_log.config(state='normal')
    chat_log.insert(tk.END, f"Bot: {bot_response}\n")
    chat_log.config(state='disabled')

    # Synthesize and play the speech
    threading.Thread(target=synthesize_speech, args=(bot_response,), daemon=True).start()

# Video capture and display
cap = cv2.VideoCapture(0)

def show_frame():
    global detected_emotion
    ret, frame = cap.read()
    if not ret:
        return
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), thickness=7)  # Changed to white
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

        img_pixels = image.img_to_array(roi_rgb)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        detected_emotion = emotions[max_index]
        emotion_label.config(text=f"Detected Emotion: {detected_emotion}")

        # Add text for the detected emotion above the rectangle
        cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, show_frame)

video_label = ttk.Label(video_frame)
video_label.pack(pady=5)

# Place the button below the video frame
capture_emotion_button = ttk.Button(video_frame, text="Capture Emotion", command=capture_emotion)
capture_emotion_button.pack(side=tk.BOTTOM, pady=10)

show_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
