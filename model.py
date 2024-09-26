from keras.models import load_model
from keras_preprocessing import image
import numpy as np

# Load the saved model
# model = load_model('D:/visual atudio code/AI_PROJECTS/Flower_Recognition/flower_classifier1.keras') #3 classes
model = load_model('D:/visual atudio code/AI_PROJECTS/Flower_Recognition/flower_recognition_model.keras') #5 classes

# Define class names manually (adjust according to your training data)
class_indices = {'daisy':0, 'Dandelion': 1, 'Rose': 2, 'Sunflower': 3, 'tulip': 4}
class_names = list(class_indices.keys())

# Prediction function for new images
def predict_flower(model, img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_tensor)

    # Get the predicted class
    predicted_class = class_names[np.argmax(prediction)]
    print(f"\n\n\033[95mFlower in Image is a\033[0m", f"\033[93m' {predicted_class} ' \033[0m\n\n")
    print(f"\nFlower is : {predicted_class}\n")

# Example usage:

flower_name = input("\nEnter flower path (s1.jpg): ")

if flower_name != '':
    path = 'D:/visual atudio code/AI_PROJECTS/Flower_Recognition/img/' + flower_name
    predict_flower(model, path)

input("")

