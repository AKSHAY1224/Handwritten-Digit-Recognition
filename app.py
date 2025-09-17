# app.py
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load the trained model (make sure mnist_cnn.h5 is in the same folder)
model = tf.keras.models.load_model('mnist_cnn.h5')

# Choose a resampling filter that's compatible across Pillow versions
try:
    # Pillow >= 9.1
    resample_mode = Image.Resampling.LANCZOS
except AttributeError:
    # Older Pillow versions
    try:
        resample_mode = Image.LANCZOS
    except AttributeError:
        # Fallback
        resample_mode = Image.BICUBIC


def preprocess_pil_image(pil_img):
    """
    Convert a PIL image into a (1,28,28,1) float32 numpy array
    matching MNIST-style inputs.
    """
    img = pil_img.convert('L')  # ensure grayscale

    # If background is light, invert so the digit is white on black (MNIST-like)
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    # Crop to content (remove empty borders)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Resize while keeping aspect ratio to fit into a 20x20 box
    # Use the compatibility resample_mode chosen above
    img.thumbnail((20, 20), resample=resample_mode)

    # Paste the resized image centered on a 28x28 black canvas
    new_img = Image.new('L', (28, 28), 0)
    left = (28 - img.size[0]) // 2
    top = (28 - img.size[1]) // 2
    new_img.paste(img, (left, top))

    # Normalize to [0,1] and reshape for the model
    arr = np.array(new_img).astype('float32') / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))

        x = preprocess_pil_image(pil_img)
        preds = model.predict(x)
        pred = int(np.argmax(preds))
        probs = [float(p) for p in preds[0]]

        return jsonify({'prediction': pred, 'probabilities': probs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
