import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import Scrollbar, Text, Entry, Button, END

# Load intents from JSON file
json_url = "intents.json"
with open(json_url, "r") as file:
    intents_data = json.load(file)

# Extract intent questions and responses
questions = []
responses = []

for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        questions.append(pattern)
        responses.append(intent['responses'][0])  # Assuming taking the first response

# Create DataFrame
df = pd.DataFrame({'Question': questions, 'Answer': responses})

# Text Processing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Question'].values.astype('U'))

# Function to get response
def get_response(user_input):
    user_input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vec, X)
    idx = np.argmax(similarities)
    return df.iloc[idx]['Answer']

# GUI Setup
def send():
    user_input = entry.get()
    if user_input.strip() == '':
        return
    response = get_response(user_input)
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {user_input}\n", 'user')
    chat_history.insert(tk.END, f"Chatbot: {response}\n", 'bot')
    chat_history.config(state=tk.DISABLED)
    chat_history.see(tk.END)
    entry.delete(0, tk.END)

root = tk.Tk()
root.title("Simple Chatbot")

frame = tk.Frame(root)
frame.pack(pady=10)

scrollbar = Scrollbar(frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

chat_history = Text(frame, wrap=tk.WORD, width=50, height=20, yscrollcommand=scrollbar.set)
chat_history.pack()

scrollbar.config(command=chat_history.yview)

entry = Entry(root, width=50)
entry.pack(pady=10)

send_button = Button(root, text="Send", command=send)
send_button.pack()

# Style configuration
chat_history.tag_configure('user', foreground='blue')
chat_history.tag_configure('bot', foreground='green')

# Initialize chat with a greeting
chat_history.config(state=tk.NORMAL)
chat_history.insert(tk.END, "Chatbot: Hello! Ask me a question or type 'exit' to end.\n", 'bot')
chat_history.config(state=tk.DISABLED)

root.mainloop()
