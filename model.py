import io
import os
import re
import base64
import nltk
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# YouTube Data API setup
load_dotenv(dotenv_path=".env")
api_key = os.getenv("API_KEY") 
youtube = build('youtube', 'v3', developerKey=api_key)


# Function to clean text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"<[^>]*>", "", text)  # Remove HTML tags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = text.lower().strip().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Train LSTM model
# Load IMDB Dataset
data = pd.read_csv('Dataset\sentiment_reviews.csv')
data['review'] = data['review'].apply(preprocess_text)

# Encode labels
encoder = LabelEncoder()
data['sentiment'] = encoder.fit_transform(data['sentiment'])
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    data['review'], data['sentiment'], stratify=data['sentiment'], test_size=0.2
)

# Hyperparameters
vocab_size = 3000
embedding_dim = 100
max_length = 200
oov_tok = "<OOV>"

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Build LSTM model
lstm_model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile LSTM model
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.summary()

# Train LSTM model
num_epochs = 5
lstm_model.fit(
    train_padded, train_labels,
    epochs=num_epochs,
    validation_split=0.1,
    verbose=1
)

# Extract comments from a YouTube video
def get_youtube_comments(link):
    match = re.search(r"(?:v=|\/)([A-Za-z0-9_-]{11})", link)
    if not match:
        raise ValueError("Invalid YouTube link format.")
    video_id = match.group(1)

    comments = []
    request = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText', maxResults=100)
    while request:
        response = request.execute()
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)
    
    return comments


# Predict sentiment for YouTube videos using the LSTM model
def predict_sentiment(comments):
    preprocessed_comments = [preprocess_text(comment) for comment in comments]
    sequences = tokenizer.texts_to_sequences(preprocessed_comments)
    padded = pad_sequences(sequences, maxlen=200, padding='post')
    predictions = lstm_model.predict(padded)
    return ["Positive" if pred >= 0.5 else "Negative" for pred in predictions]


# Visualize sentiments
def visualize_sentiments(sentiment_counts):
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    colors = ['green', 'red']

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title("Sentiment Analysis Results for YouTube video")
    plt.axis('equal')

    buffer = io.BytesIO()
    plt.savefig("static/plot.png")

    buffer.seek(0)
    chart = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return chart

