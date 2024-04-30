import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import glob
import random
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# TensorFlow does not need manual seed setting for reproducibility in this snippet

'''
Reading in JSON file and creating key for ASL Video dataset
'''
main_path = "/app/rundir/CPSC542-FinalProject/archive-3/"
wlasl_df = pd.read_json(main_path + "WLASL_v0.3.json")

# Reading in the JSON file
with open(main_path + 'WLASL_v0.3.json', 'r') as data_file:
    json_data = json.load(data_file)

bbox_data = []
for entry in json_data:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        bbox = instance['bbox'] if 'bbox' in instance else None
        bbox_data.append({
            'video_id': video_id,
            'gloss': gloss,
            'bbox': bbox,
        })

bbox_df = pd.DataFrame(bbox_data)
bbox_df.set_index('video_id', inplace=True)

# Helper function to load images
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    return image

# Helper function to apply bounding box cropping
def apply_bbox(image, video_id):
    if video_id in bbox_df.index and bbox_df.loc[video_id]['bbox'] is not None:
        bbox = bbox_df.loc[video_id]['bbox']
        image = tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], bbox[3], bbox[2])
    return image

# Define the CNN-LSTM model
def create_model(input_shape, num_classes):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.RepeatVector(16)(x)  # Repeat vector for sequence input to LSTM
    x = layers.LSTM(256, return_sequences=True)(x)
    x = layers.LSTM(256)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=x)
    return model

model = create_model((256, 256, 3), 2000)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training and validation
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Assuming `train_ds` and `val_ds` are defined somewhere above as datasets
train_generator = train_datagen.flow_from_directory(
    directory='/app/rundir/CPSC542-FinalProject/images/',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    directory='/app/rundir/CPSC542-FinalProject/images/',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary')

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Number of steps per epoch
    epochs=50,
    validation_data=val_generator,
    validation_steps=50,
    callbacks=[reduce_lr, early_stopping]
)

# Save the model
model.save('model.h5')

# Plotting training and validation loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.savefig('training_plot.png')
