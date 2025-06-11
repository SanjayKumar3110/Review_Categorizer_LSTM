import re
import os
import nltk
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tf_keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load saved model and tokenizer
# model = load_model("saved_model//lstm_sentiment_model.h5")
model_path = os.path.join(os.path.dirname(__file__),"saved_model", "lstm_sentiment_model.h5")
model = load_model(model_path)

tokenizer_path = os.path.join(os.path.dirname(__file__), "saved_model", "tokenizer.pkl")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# Load API key
load_dotenv(dotenv_path="..//.env")
api_key = os.getenv("API_KEY")
youtube = build('youtube', 'v3', developerKey=api_key)

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.lower().strip().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Get comments
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

# Predict sentiment
def predict_sentiment(comments):
    processed = [preprocess_text(c) for c in comments]
    sequences = tokenizer.texts_to_sequences(processed)
    padded = pad_sequences(sequences, maxlen=200, padding='post')
    preds = model.predict(padded)
    return ["Positive" if p >= 0.5 else "Negative" for p in preds]

# Plot results
def visualize_sentiments(predictions):
    from collections import Counter
    counts = Counter(predictions)
    labels = counts.keys()
    sizes = counts.values()
    colors = ['green', 'red']

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title("Sentiment Analysis Results for YouTube video")
    plt.axis('equal')

    # buffer = io.BytesIO()
    # plt.savefig("..//static//plot.png")
    # buffer.seek(0)
    # chart = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # buffer.close()
    # return chart

    output_path = "static/plot.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

# Continuous loop
if __name__ == "__main__":
    while True:
        link = input("\n Enter YouTube video link (or type 'exit' to quit): ").strip()
        if link.lower() in ["exit", "quit"]:
            print("👋 Exiting the sentiment analyzer.")
            break

        try:
            comments = get_youtube_comments(link)
            if not comments:
                print("⚠️ No comments found for this video.")
                continue

            predictions = predict_sentiment(comments)

            from collections import Counter
            counts = Counter(predictions)
            print("📊 Sentiment Counts:", counts)

            visualize_sentiments(predictions)
            print("Sentiment chart saved successfully")

        except Exception as e:
            print("❌ Error:", e)