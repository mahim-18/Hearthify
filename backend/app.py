from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins="http://localhost:5173")  # Ensure React app is allowed

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

@app.route("/")
def home():
    return "Heart Disease Prediction API is working successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].tolist()

    response = jsonify({"prediction": int(prediction), "probability": probability})
    response.headers.add("Access-Control-Allow-Origin", "*")  # Allow all origins
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST")

    return response

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)

