from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences

# Flask app
app = Flask(__name__)

# Load the saved model and tokenizer
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Preprocessing function
def preprocess_text(text):
    TAG_RE = re.compile(r'<[^>]+>')
    stop_words = set(stopwords.words('english'))

    # Remove HTML tags
    text = TAG_RE.sub('', text)
    # Lowercase text
    text = text.lower()
    # Remove punctuations and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data.get('review', '')

    if not review:
        return jsonify({"error": "No review provided"}), 400

    # Preprocess the review
    processed_review = preprocess_text(review)
    sequences = tokenizer.texts_to_sequences([processed_review])
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

    # Predict sentiment
    prediction = model.predict(padded)
    sentiment = "Positive" if prediction[0] >= 0.5 else "Negative"

    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
