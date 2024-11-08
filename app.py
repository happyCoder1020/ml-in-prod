from flask import Flask, render_template, request, jsonify
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys 
import os
import base64
sys.path.append(os.path.abspath("./model"))
from model.load import * 


global graph, model
SAVE_FOLDER = "C:\\Users\\AIUSER\\ShareAI\\deploying-web-app\\flask"

model, graph = init()

app = Flask(__name__)


@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/drawings/')
def drawings_view():
    return render_template('draw.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the base64 image data from the request
    img_data = request.json.get('imgData')  # This is the base64 data from the frontend

    # Extract the base64 string (remove the "data:image/png;base64," part)
    img_data = img_data.split(',')[1]

    # Decode the base64 string to bytes
    img_bytes = base64.b64decode(img_data)

    # Save the image to a file
    img_filename = os.path.join(SAVE_FOLDER, 'output.png')
    with open(img_filename, 'wb') as f:
        f.write(img_bytes)

    # Load the saved image and preprocess it
    x = imread(img_filename, mode='L')
    x = np.invert(x)  # Invert colors (optional)
    x = imresize(x, (28, 28))  # Resize image to match model input size (e.g., 28x28)
    x = x.reshape(1, 28, 28, 1)  # Reshape for model input (e.g., adding batch dimension)

    # Make a prediction using the model
    with graph.as_default():
        out = model.predict(x)

    # Get the predicted class (digit) and return the result
    prediction = np.argmax(out, axis=1)[0]  # Get the predicted class (e.g., 0-9)
    response = f"The number you drew is: {prediction}"

    return response

if __name__ == '__main__':
    app.run(debug=True, port=8000)