from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.svm import SVC

app = Flask(__name__)

# Load the hybrid model
hybrid_model = joblib.load('hybrid_model.pkl')

# Load tokenizer and TF-IDF vectorizer
tokenizer = joblib.load('tokenizer.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Set the max sequence length
max_sequence_length = 100

# Emotion mapping
emotion_mapping = {
    0: 'joy',
    1: 'fear',
    2: 'anger',
    3: 'sadness',
    4: 'disgust',
    5: 'shame',
    6: 'guilt'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    print(f"Received text: {text}")  # Debug output

    try:
        # Preprocess text input
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = np.zeros((1, 0))  # Set padded sequences to zero length
        print(f"Padded sequences shape: {padded_sequences.shape}")  # Debug output

        # Get TF-IDF features for the text
        text_tfidf = tfidf_vectorizer.transform([text]).toarray()
        print(f"TF-IDF shape: {text_tfidf.shape}")  # Debug output

        # Use only TF-IDF features
        combined_features = text_tfidf
        print(f"Combined features shape: {combined_features.shape}")  # Debug output

        # Predict emotion using the hybrid model
        predicted_emotion = hybrid_model.predict(combined_features)
        print(f"Predicted emotion: {predicted_emotion}")  # Debug output

        return jsonify({'emotion': emotion_mapping[predicted_emotion[0]]})

    except Exception as e:
        print(f"Error: {e}")  # Print the error for debugging
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
