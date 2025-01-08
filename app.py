from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model & scaler
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return "House Price Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    return jsonify({
        "predicted_price": round(float(prediction[0]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
