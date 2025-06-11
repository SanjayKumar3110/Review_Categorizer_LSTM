import re
import nltk
import pandas as pd
import numpy as np
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.lower().strip().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load and preprocess data
data = pd.read_csv("..//Dataset//sentiment_reviews.csv")
data['review'] = data['review'].apply(preprocess_text)

# Label encoding
encoder = LabelEncoder()
data['sentiment'] = encoder.fit_transform(data['sentiment'])

# Split data
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    data['review'], data['sentiment'], stratify=data['sentiment'], test_size=0.2
)

# Tokenization & padding
vocab_size = 3000
embedding_dim = 100
max_length = 200
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Model architecture
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_padded, train_labels, epochs=5, validation_split=0.1)

# Save model and tokenizer
model.save("saved_model/lstm_sentiment_model.h5")
joblib.dump(tokenizer, "saved_model/tokenizer.pkl")
joblib.dump(encoder, "saved_model/label_encoder.pkl")
