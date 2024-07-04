# # from tensorflow.keras.preprocessing.text import Tokenizer
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # import pandas as pd
# # import json
# # import string
# # from nltk.corpus import stopwords
# # from sklearn.preprocessing import LabelEncoder
# # import tensorflow as tf
# # import pickle
# # import random  # Import random for random response selection
# # from tensorflow.keras.layers import LSTM, Dropout

# # # Load data
# # with open('D:\SYSTEM DATA\Downloads\ChatBot-main\ChatBot-main\intents.json', encoding='utf-8') as content:
# #     data = json.load(content)

# # # Process data
# # patterns = []
# # tags = []
# # for intent in data['intents']:
# #     for line in intent['patterns']:
# #         patterns.append(line)
# #         tags.append(intent['tag'])

# # data_df = pd.DataFrame({"patterns": patterns, "tags": tags})

# # # Preprocessing
# # stop_words = set(stopwords.words('english'))

# # def clean_data(text):
# #     text = text.lower()
# #     text = ''.join([word for word in text if word not in string.punctuation])
# #     words = [w for w in text.split() if w not in stop_words]
# #     return ' '.join(words)

# # data_df['patterns'] = data_df['patterns'].apply(clean_data)

# # # Tokenization
# # tokenizer = Tokenizer(num_words=2000)
# # tokenizer.fit_on_texts(data_df['patterns'])
# # train_sequences = tokenizer.texts_to_sequences(data_df['patterns'])
# # train_padded = pad_sequences(train_sequences)

# # # Encoding labels
# # le = LabelEncoder()
# # labels = le.fit_transform(data_df['tags'])




# # # Define the model architecture
# # model = tf.keras.models.Sequential([
# #     tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10, input_length=train_padded.shape[1]),
# #     tf.keras.layers.LSTM(64, return_sequences=True),  # Increase the number of units in the LSTM layer
# #     tf.keras.layers.Dropout(0.2),  # Add dropout regularization
# #     tf.keras.layers.LSTM(32, return_sequences=True),  # Add another LSTM layer
# #     tf.keras.layers.Dropout(0.2),  # Add dropout regularization
# #     tf.keras.layers.Flatten(),
# #     tf.keras.layers.Dense(len(set(labels)), activation='softmax')
# # ])

# # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # # Train the model
# # model.fit(train_padded, labels, epochs=240)

# # # Save the model and label encoder
# # model.save('model.h5')
# # with open('label_encoder.pkl', 'wb') as le_file:
# #     pickle.dump(le, le_file)

# # # Get response for a given tag
# # def get_response(tag):
# #     for intent in data['intents']:
# #         if intent['tag'] == tag:
# #             responses = intent['responses']
# #             return random.choice(responses)  # Select a random response from the list
# #     return "Sorry, I didn't understand that."

# # # Test the chatbot
# # def test_chatbot():
# #     while True:
# #         user_input = input("You: ")
# #         if user_input.lower() == 'quit':
# #             break
# #         else:
# #             # Preprocess user input
# #             processed_input = clean_data(user_input)
# #             # Tokenize and pad user input
# #             tokenized_input = tokenizer.texts_to_sequences([processed_input])
# #             padded_input = pad_sequences(tokenized_input, maxlen=train_padded.shape[1])
# #             # Predict intent
# #             predictions = model.predict(padded_input)
# #             predicted_tag_index = tf.argmax(predictions, axis=1).numpy()[0]
# #             predicted_tag = le.inverse_transform([predicted_tag_index])[0]
# #             # Get response
# #             response = get_response(predicted_tag)
# #             print("Bot:", response)
            
# # # Test the chatbot
# # test_chatbot()

# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import json
# import pickle
# import random

# app = Flask(__name__)

# # Load trained model
# model = tf.keras.models.load_model('model.h5')

# # Load label encoder
# with open('label_encoder.pkl', 'rb') as le_file:
#     le = pickle.load(le_file)

# # Load tokenizer
# with open('tokenizer.pkl', 'rb') as tokenizer_file:
#     tokenizer = pickle.load(tokenizer_file)

# # Load intents data
# with open('D:\SYSTEM DATA\Downloads\ChatBot-main\ChatBot-main\intents.json', encoding='utf-8') as content:
#     intents = json.load(content)

# # Helper function to clean input text
# stop_words = set(stopwords.words('english'))
# def clean_data(text):
#     text = text.lower()
#     text = ''.join([word for word in text if word not in string.punctuation])
#     words = [w for w in text.split() if w not in stop_words]
#     return ' '.join(words)

# # Tokenize function
# def tokenize_text(text):
#     # Tokenize the text
#     text = clean_data(text)
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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import json
import string
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pickle
import random  # Import random for random response selection
from tensorflow.keras.layers import LSTM, Dropout

# Load data
with open('D:\SYSTEM DATA\Downloads\ChatBot-main\ChatBot-main\intents.json', encoding='utf-8') as content:
    data = json.load(content)

# Process data
patterns = []
tags = []
for intent in data['intents']:
    for line in intent['patterns']:
        patterns.append(line)
        tags.append(intent['tag'])

data_df = pd.DataFrame({"patterns": patterns, "tags": tags})

# Preprocessing
stop_words = set(stopwords.words('english'))

def clean_data(text):
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

data_df['patterns'] = data_df['patterns'].apply(clean_data)

# Tokenization
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data_df['patterns'])
train_sequences = tokenizer.texts_to_sequences(data_df['patterns'])
train_padded = pad_sequences(train_sequences)

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Encoding labels
le = LabelEncoder()
labels = le.fit_transform(data_df['tags'])

# Save the label encoder
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10, input_length=train_padded.shape[1]),
    tf.keras.layers.LSTM(64, return_sequences=True),  # Increase the number of units in the LSTM layer
    tf.keras.layers.Dropout(0.2),  # Add dropout regularization
    tf.keras.layers.LSTM(32, return_sequences=True),  # Add another LSTM layer
    tf.keras.layers.Dropout(0.2),  # Add dropout regularization
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_padded, labels, epochs=240)

# Save the model
model.save('model.h5')

# Get response for a given tag
def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']
            return random.choice(responses)  # Select a random response from the list
    return "Sorry, I didn't understand that."

# Test the chatbot
def test_chatbot():
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        else:
            # Preprocess user input
            processed_input = clean_data(user_input)
            # Tokenize and pad user input
            tokenized_input = tokenizer.texts_to_sequences([processed_input])
            padded_input = pad_sequences(tokenized_input, maxlen=train_padded.shape[1])
            # Predict intent
            predictions = model.predict(padded_input)
            predicted_tag_index = tf.argmax(predictions, axis=1).numpy()[0]
            predicted_tag = le.inverse_transform([predicted_tag_index])[0]
            # Get response
            response = get_response(predicted_tag)
            print("Bot:", response)

# Test the chatbot
test_chatbot()
