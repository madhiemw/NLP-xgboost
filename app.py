import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import pickle

from utils import preprocess_text, remove_stopwords, lemmatize_tokens, untokenize, stem_tokens

app = Flask(__name__)

with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            df = pd.read_csv(file, usecols=['review'])
            
            df.dropna(inplace=True)

            df['review'] = df['review'].apply(preprocess_text)
            df['review'] = df['review'].apply(remove_stopwords)
            df['review'] = df['review'].apply(stem_tokens)
            df['review'] = df['review'].apply(lemmatize_tokens)
            df['review'] = df['review'].apply(untokenize)

            input_vectorized = vectorizer.transform(df['review'])
            predictions = model.predict(input_vectorized)
            
            prediction_labels = ['Positive' if pred == 1 else 'Negative' for pred in predictions]

            df['prediction'] = prediction_labels

            predictions_json = df.to_json(orient='records')

            return predictions_json

if __name__ == '__main__':
    app.run(debug=True, port=5000)