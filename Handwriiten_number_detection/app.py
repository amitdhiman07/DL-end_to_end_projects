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

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (28, 28))
        gray_img = gray_img.reshape(-1, 28, 28, 1)
        gray_img = gray_img.astype('float32') / 255
        prediction = model.predict(gray_img)
        predicted_digit = np.argmax(prediction)
        return str(predicted_digit).encode('utf-8')  # Encode prediction as UTF-8

if __name__ == "__main__":
    app.run(debug=True)
