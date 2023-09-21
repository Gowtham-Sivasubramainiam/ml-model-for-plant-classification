from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model('plant_classification_modele_new_plant_1.h5')
output_layer = model.output
train_path = 'Dataset1/Train'
train_subfolders = [folder for folder in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, folder))]

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def predict_plant(image_path):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    class_index = np.argmax(predictions)
    class_labels = train_subfolders
    predicted_class = class_labels[class_index]
    return predicted_class

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']
        image_path = 'temp_image.jpg'
        image_file.save(image_path)
                predicted_class = predict_plant(image_path)
        
        response = {'prediction': predicted_class}
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
