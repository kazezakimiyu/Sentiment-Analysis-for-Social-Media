import tkinter as tk
from tkinter import Label, Button, Entry
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load model and tokenizer
model = load_model('sentiment_lstm_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_sentiment():
    text = text_entry.get()
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=200, padding='post')
    pred = model.predict(padded)
    labels = ['Negative', 'Neutral', 'Positive']
    result_label.config(text=f"Sentiment: {labels[np.argmax(pred)]}")

# Create the main window
root = tk.Tk()
root.title("Sentiment Analysis App")
root.geometry('500x200')

# Layout using grid
label = Label(root, text="Enter Text:")
label.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=10)

text_entry = Entry(root, width=40)
text_entry.grid(row=0, column=1, sticky='we', padx=(0, 10), pady=10)

predict_button = Button(root, text="Predict Sentiment", command=predict_sentiment)
predict_button.grid(row=1, column=1, pady=10)

result_label = Label(root, text="Sentiment will appear here", font=('Helvetica', 10))
result_label.grid(row=2, column=1, pady=10)

# Start the GUI event loop
root.mainloop()