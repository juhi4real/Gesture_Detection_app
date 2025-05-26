import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load model architecture
json_path = r"C:\Users\KIIT0001\Documents\REAL_TIME_DETECTION_ASSIGNMENT\signlanguagedetectionmodel48x48.json"
weights_path = r"C:\Users\KIIT0001\Documents\REAL_TIME_DETECTION_ASSIGNMENT\signlanguagedetectionmodel48x48.h5"

with open(json_path, "r") as json_file:
    model = model_from_json(json_file.read())

# Load weights
model.load_weights(weights_path)

# Class labels
labels = ['A', 'M', 'N', 'S', 'T', 'blank']

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    if file:
        static_folder = os.path.join(os.getcwd(), 'static')
        os.makedirs(static_folder, exist_ok=True)
        filepath = os.path.join(static_folder, file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        pred_class = labels[np.argmax(prediction)]
        accuracy = "{:.2f}%".format(np.max(prediction) * 100)

        return render_template('result.html',
                               prediction=pred_class,
                               accuracy=accuracy,
                               image_path='static/' + file.filename)

if __name__ == '__main__':
    app.run(debug=True)