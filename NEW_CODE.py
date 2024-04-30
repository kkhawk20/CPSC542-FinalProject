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
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch

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
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

class CNNLSTMHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.RepeatVector(16),
            layers.LSTM(
                units=hp.Int('units', min_value=128, max_value=512, step=128),
                return_sequences=True
            ),
            layers.LSTM(
                units=hp.Int('units', min_value=128, max_value=512, step=128),
            ),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

hypermodel = CNNLSTMHyperModel((256, 256, 3), 2000)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='keras_tuner_dir',
    project_name='cnn_lstm_tuning'
)

# Training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Adjust brightness
    channel_shift_range=150.0  # Adjust color channels
)
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

retrain = False
if retrain:
    tuner.search(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stopping])
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save('best_model.h5')
    history = best_model.history

    # Plot training and validation loss and accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.savefig('training_plot.png')

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize

# Grad-CAM Function
def apply_grad_cam(model, img_array, category_index, layer_name='conv5_block3_out'):
    gradcam = Gradcam(model, model_modifier=None, clone=False)
    cam = gradcam(loss=lambda output: output[category_index], seed_input=img_array, penultimate_layer=-1)
    cam = normalize(cam)
    cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_array
    return superimposed_img



model = tf.keras.models.load_model('model.h5')

# Mapping labels from label dictionary file
labels_dict = {}
inverse_labels_dict = {}
with open('labels_dict.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split(': ')
        value = int(value)
        labels_dict[key] = value
        inverse_labels_dict[value] = key

def predict_and_visualize(video_path, model, bbox_df, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    gradcam_output_dir = output_dir + "_GRADCAM"
    if not os.path.exists(gradcam_output_dir):
        os.makedirs(gradcam_output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    print("Video successfully opened.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read or error in fetching frame.")
            break

        print(f"Processing frame {frame_count + 1}")

        # Convert frame to PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video_id = os.path.basename(video_path).split('.')[0]
        bbox = None
        
        if video_id in bbox_df.index and bbox_df.loc[video_id]['bbox'] is not None:
            bbox = bbox_df.loc[video_id]['bbox']
            print(f"Original BBox for video_id {video_id}: {bbox}")

            # Calculate the reduced bounding box
            reduction_ratio = 0.8  # Example reduction ratio
            new_width = bbox[2] * reduction_ratio
            new_height = bbox[3] * reduction_ratio
            new_x = bbox[0] + (bbox[2] - new_width) / 2
            new_y = bbox[1] + (bbox[3] - new_height) / 2
            bbox = [int(new_x), int(new_y), int(new_width), int(new_height)]

            frame_pil = frame_pil.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Process image for the model
        frame_processed = keras_image.img_to_array(frame_pil)
        frame_processed = tf.image.resize(frame_processed, [256, 256])
        frame_processed = np.expand_dims(frame_processed, axis=0)  # Add batch dimension
        frame_processed = np.copy(frame_processed)  # Ensure the array is writable
        frame_processed /= 255.0  # Normalize to [0,1]

        # Predict using the model
        prediction = model.predict(frame_processed)
        predicted_label = np.argmax(prediction, axis=1)
        predicted_label_name = inverse_labels_dict[predicted_label[0]]
        print(f"Predicted label: {predicted_label_name}")
        
        grad_cam_img = apply_grad_cam(model, frame_processed[0].astype(np.uint8), 
                                    predicted_label, 'last_conv_layer_name')


        # Annotate and save frame
        if bbox:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, predicted_label_name, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the frame to a file
        frame_output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_output_path, frame)

        # Save the same frame with gradCAM to a file
        gradcam_output_path = os.path.join(gradcam_output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(gradcam_output_path, grad_cam_img)

        print(f"Frame {frame_count + 1} saved at {frame_output_path}")
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed. Frames saved to:", output_dir)

# Ensure the video path and output directory are correctly specified
video_path = './videos/a/01610.mp4'  # Make sure the file extension is specified if needed
output_dir = './output_frames_a_test'
predict_and_visualize(video_path, model, bbox_df, output_dir)