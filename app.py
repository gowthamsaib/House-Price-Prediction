from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("housepred.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ${round(prediction[0], 2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)