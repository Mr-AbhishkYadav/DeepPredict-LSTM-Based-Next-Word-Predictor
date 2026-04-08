import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# 1. Load your saved files
model = load_model('lstm_model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('max_len.pkl', 'rb') as handle:
    max_sequence_len = pickle.load(handle)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    seed_text = data.get('text', '')
    next_words = int(data.get('count', 3)) # Number of words to predict

    if not seed_text:
        return jsonify({"error": "No text provided"})

    output_text = seed_text

    # Loop to predict multiple words
    for _ in range(next_words):
        # Tokenize and pad the current text
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predict the next word index
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        
        # Convert index back to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        if not output_word:
            break
            
        output_text += " " + output_word

    return jsonify({"result": output_text})

if __name__ == '__main__':
    app.run(debug=True)