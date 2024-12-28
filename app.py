# app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    with open('iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

@app.route('/')
def home():
    return "Welcome to the Iris Prediction Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the server.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json()

        # Ensure the incoming data contains the 'features' key
        if 'features' not in data:
            return jsonify({'error': 'Missing features in request body'}), 400

        # Extract features from the incoming JSON data
        features = np.array(data['features']).reshape(1, -1)

        # Validate the shape of features to match the model input
        if features.shape[1] != 4:
            return jsonify({'error': 'Invalid number of features. Expected 4 features.'}), 400

        # Make a prediction using the loaded model
        prediction = model.predict(features)

        # Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
