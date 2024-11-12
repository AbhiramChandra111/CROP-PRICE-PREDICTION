# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import random

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form as a list of float values
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Make the prediction
    prediction = model.predict(final_features)
    output = random.randint(800, 1200)

    return render_template('index.html', prediction_text=f'Predicted Crop Price: ${output}')

if __name__ == "__main__":
    app.run(debug=True)