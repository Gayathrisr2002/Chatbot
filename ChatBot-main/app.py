# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import json
# import pickle
# import random  # Add this import for random response selection

# app = Flask(__name__)

# # Load trained model
# model = tf.keras.models.load_model('model.h5')

# # Load label encoder
# with open('D:\SYSTEM DATA\Downloads\ChatBot-main\ChatBot-main\label_encoder.pkl', 'rb') as le_file:
#     le = pickle.load(le_file)

# # Load intents data
# with open('D:\SYSTEM DATA\Downloads\ChatBot-main\ChatBot-main\intents.json', encoding='utf-8') as content:
#     intents = json.load(content)

# # Tokenize function
# def tokenize_text(text):
#     # Tokenize the text
#     tokenizer = Tokenizer(num_words=2000)
#     tokenizer.fit_on_texts([text])
#     sequences = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(sequences, maxlen=69, padding='post')  # Update maxlen to 69
#     return padded

# # Predict function
# def predict_intent(text):
#     # Tokenize input text
#     padded_text = tokenize_text(text)
#     # Predict using the loaded model
#     predictions = model.predict(padded_text)
#     # Get the predicted tag index
#     predicted_tag_index = predictions.argmax(axis=-1)[0]
#     # Get the corresponding tag
#     predicted_tag = le.inverse_transform([predicted_tag_index])[0]
#     return predicted_tag

# # Get response for a given tag
# def get_response(tag):
#     for intent in intents['intents']:
#         if intent['tag'] == tag:
#             responses = intent['responses']
#             return random.choice(responses)  # Select a random response from the list
#     return "Sorry, I didn't understand that."


# # Route for home page
# @app.route('/')
# def home():
#     return render_template('chat.html')

# # Route for getting responses from the bot
# @app.route('/get', methods=['POST'])
# def chatbot_response():
#     user_message = request.form['msg']
#     # Call predict_intent function to get the predicted tag
#     predicted_tag = predict_intent(user_message)
#     # Get response for the predicted tag
#     response = get_response(predicted_tag)
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pickle
import random

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model('model.h5')

# Load label encoder
with open('D:/SYSTEM DATA/Downloads/ChatBot-main/ChatBot-main/label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

# Load tokenizer
with open('D:/SYSTEM DATA/Downloads/ChatBot-main/ChatBot-main/tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load intents data
with open('D:/SYSTEM DATA/Downloads/ChatBot-main/ChatBot-main/intents.json', encoding='utf-8') as content:
    intents = json.load(content)

# Tokenize function
def tokenize_text(text):
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=69, padding='post')  # Update maxlen to 69
    return padded

# Predict function
def predict_intent(text):
    # Tokenize input text
    padded_text = tokenize_text(text)
    # Predict using the loaded model
    predictions = model.predict(padded_text)
    # Get the predicted tag index
    predicted_tag_index = predictions.argmax(axis=-1)[0]
    # Get the corresponding tag
    predicted_tag = le.inverse_transform([predicted_tag_index])[0]
    return predicted_tag

# Get response for a given tag
def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']
            return random.choice(responses)  # Select a random response from the list
    return "Sorry, I didn't understand that."

# Route for home page
@app.route('/')
def home():
    return render_template('chat.html')

# Route for getting responses from the bot
@app.route('/get', methods=['POST'])
def chatbot_response():
    user_message = request.form['msg']
    # Call predict_intent function to get the predicted tag
    predicted_tag = predict_intent(user_message)
    # Get response for the predicted tag
    response = get_response(predicted_tag)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
