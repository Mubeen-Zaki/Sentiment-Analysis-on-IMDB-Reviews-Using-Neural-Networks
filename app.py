# Library imports
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from numpy import array
import requests
from bs4 import BeautifulSoup

from keras.preprocessing.text import one_hot, Tokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
from keras_preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from keras_preprocessing.text import tokenizer_from_json
import io
import json

from preprocessing_function import CustomPreprocess


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
nltk.download('punkt')




stopwords_list = set(stopwords.words('english'))
maxlen = 100

# Load model
model_path ='Models\\lstm_model_acc_0.867.h5'
pretrained_lstm_model = load_model(model_path)

# Loading
with open('b3_tokenizer.json') as f:
    data = json.load(f)
    loaded_tokenizer = tokenizer_from_json(data)

custom = CustomPreprocess()


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form['text_input']
    link_input = request.form['link_input']

    # print(text_input)
    # print(type(text_input))
    # print(link_input)
    # print(type(link_input))
    
    if link_input == "":
        sentiment = predictForText(text_input)
        summary = summarize_text([text_input])
    else:
        sentiment, reviews = predictForLink(link_input)
        summary = summarize_text(reviews)

    
    if sentiment >= 0.5:
        return render_template('index.html', prediction_text=f"Positive Review with probable IMDb rating as: {np.round(sentiment*10,1)} . Summarized Review : {summary}")
    else:
        return render_template('index.html', prediction_text=f"Negative Review with probable IMDb rating as: {np.round(sentiment*10,1)} . Summarized Review : {summary}")

def predictForText(text_input):
    processed_text = custom.preprocess_text(text_input)
    # print(processed_text)
    tokenized_seq = loaded_tokenizer.texts_to_sequences([processed_text])
    # print(tokenized_seq)
    padded_seq = pad_sequences(tokenized_seq, padding='post', maxlen=100)
    # print(padded_seq)
    sentiment = pretrained_lstm_model.predict(padded_seq)
    return sentiment[0][0]

def predictForLink(link_input):
    unprocessed_reviews = []

    # Navigate to the user reviews page
    if '/title/' in link_input:
        reviews_link = link_input.split('?')[0] + 'reviews?ref_=tt_urv'
    else:
        raise ValueError("Invalid IMDb movie link")

    # Send a GET request to the user reviews page
    response = requests.get(reviews_link)
    if response.status_code != 200:
        raise Exception("Failed to load page")

    # Parse the HTML content of the reviews page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the review bodies
    reviews = soup.find_all('div', class_='text show-more__control')
    for review in reviews:
        unprocessed_reviews.append(review.get_text())

    # print(unprocessed_reviews)
    processed = []
    for review in unprocessed_reviews:
        review = custom.preprocess_text(review)
        processed.append(review)

    tokenized = loaded_tokenizer.texts_to_sequences(processed)
    padded = pad_sequences(tokenized, padding='post', maxlen=100)
    sentiments = pretrained_lstm_model.predict(padded)
    flat_sentiments = [sentiment[0] for sentiment in sentiments]
    avg_sent = 0
    for i in flat_sentiments:
        avg_sent += i
    return avg_sent / len(flat_sentiments), unprocessed_reviews

def summarize_text(reviews):

    text = ". ".join(reviews)

    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # Create an LSA summarizer
    summarizer = LsaSummarizer()

    # Generate summary (specify the number of sentences)
    summary = summarizer(parser.document, 2)

    summary = [str(s) for s in summary]
    return ". ".join(summary)


if __name__ == '__main__':
    app.run(debug=True)
