import pyaudio
import numpy as np
import speech_recognition as sr
import tkinter as tk
from tkinter import Button, Label, Text
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

# Audio parameters
CHUNK = 1024  # Number of samples per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate (samples per second)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Speech recognition setup
recognizer = sr.Recognizer()

# Flag to control audio capture
is_listening = False

# Lists to store audio data and FFT results
audio_data_list = []
fft_result_list = []

# Function to capture audio and convert to text
def capture_audio():
    global is_listening, audio_data_list, fft_result_list
    frames = []
    is_listening = True
    print("Listening...")
    try:
        while is_listening:
            data = stream.read(CHUNK)
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            fft_result = np.fft.fft(audio_data)
            audio_data_list.append(audio_data)
            fft_result_list.append(fft_result)
    except Exception as e:
        print(f"Error during audio capture: {e}")

    # Convert captured audio to text
    audio_bytes = b''.join(frames)
    audio_data = sr.AudioData(audio_bytes, RATE, 2)
    try:
        text = recognizer.recognize_google(audio_data)
        print("Recognized Text:", text)
        text_display.insert(tk.END, text + '\n')
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

# Function to stop audio capture
def stop_capture():
    global is_listening
    is_listening = False
    print("Stopped listening.")

# Visualization setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
line1, = ax1.plot([], [], lw=2)
line2, = ax2.plot([], [], lw=2)
ax1.set_title('Audio Signal')
ax1.set_xlim(0, CHUNK)
ax1.set_ylim(-32768, 32767)
ax2.set_title('Frequency Spectrum')
ax2.set_xlim(0, RATE / 2)
ax2.set_ylim(0, 1)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def update_visualization(frame):
    if audio_data_list:
        audio_data_full = np.concatenate(audio_data_list)
        fft_result_full = np.concatenate(fft_result_list)
        line1.set_data(np.arange(len(audio_data_full)), audio_data_full)
        freqs = np.fft.fftfreq(len(audio_data_full), 1/RATE)
        line2.set_data(freqs[:len(audio_data_full)//2], np.abs(fft_result_full[:len(audio_data_full)//2]) / (len(audio_data_full)/2))
        ax1.set_xlim(0, len(audio_data_full))
    return line1, line2

# GUI setup
root = tk.Tk()
root.title("Audio to Text Converter")

start_button = Button(root, text="Start Listening", command=lambda: threading.Thread(target=capture_audio).start())
start_button.pack()

stop_button = Button(root, text="Stop Listening", command=stop_capture)
stop_button.pack()

text_label = Label(root, text="Recognized Text:")
text_label.pack()

text_display = Text(root, height=10, width=50)
text_display.pack()

# Create a separate window for the visualization
vis_window = tk.Toplevel(root)
vis_window.title("Audio Visualization")

# Embed Matplotlib figure in the separate window
canvas = FigureCanvasTkAgg(fig, master=vis_window)
canvas.draw()
canvas.get_tk_widget().pack()

# Start the Matplotlib animation
ani = FuncAnimation(fig, update_visualization, init_func=init, blit=True, interval=100)

root.mainloop()

# Clean up
stream.stop_stream()
stream.close()
p.terminate()
