import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Paths to your dataset

validation_dir = 'D:/visual atudio code/AI_PROJECTS/Flower_Recognition/dataset/validation'
train_dir = 'D:/visual atudio code/AI_PROJECTS/Flower_Recognition/dataset/train'


# train_dir = '/dataset/train'
# validation_dir = '/dataset/validation'


# Data augmentation for training data
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=40, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   fill_mode='nearest')

# Rescaling validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Creating training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# Creating validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

# Building the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 classes, use softmax
])

# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)


model.save('flower_recognition_model.keras')


# Prediction function for new images
def predict_flower(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)

    prediction = model.predict(img_tensor)
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())

    predicted_class = class_names[np.argmax(prediction)]
    print(f"It's a {predicted_class}!")

# Example prediction
predict_flower(model, 'D:/visual atudio code/AI_PROJECTS/Flower_Recognition/img/r1.jpg')