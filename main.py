from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# 1. Load the model using the filename from your notebook
# Note: Use compile=False to avoid errors if custom optimizers were used
MODEL_PATH = 'model.h5' 
model = load_model(MODEL_PATH, compile=False)

# 2. Updated class names to match the EMOTIONS_LIST in your notebook
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Emotion detection function
def detect_emotion(img_path):
    # Load image in grayscale and 48x48 as per notebook img_size
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    
    # IMPORTANT: In your notebook, you didn't rescale (1./255). 
    # The ImageDataGenerator only used horizontal_flip=True.
    # We just need to expand dimensions to (1, 48, 48, 1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = round(np.max(prediction) * 100, 2)

    return predicted_class, confidence

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded!'
        file = request.files['file']
        if file.filename == '':
            return 'No file selected!'

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Detect emotion
            emotion, confidence = detect_emotion(file_path)

            # Ensure image_path is relative for HTML display
            display_path = file_path.replace('\\', '/')
            
            return render_template('index.html', 
                                   image_path=display_path, 
                                   emotion=emotion, 
                                   confidence=confidence)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)