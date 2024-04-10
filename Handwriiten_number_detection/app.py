import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import cv2

app = Flask(__name__)
model = load_model("model/my_model.h5")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get uploaded image
        file = request.files["file"]
        
        # Read the uploaded image as a JPEG file
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.reshape(-1, 28, 28, 1) / 255.0
        
        # Make prediction
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        return str(predicted_digit)

if __name__ == "__main__":
    app.run(debug=True)
