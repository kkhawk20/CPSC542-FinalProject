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
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import matplotlib.cm as cm
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize

def assess_model(history):

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

    # Grad-CAM Function
    def apply_grad_cam(model, img_array, category_index, layer_name):
        # Ensure image is float32, normalized (as expected by most models)
        if img_array.dtype != np.float32:
            img_array = img_array.astype('float32') / 255.0
        gradcam = Gradcam(model,
                        model_modifier=ReplaceToLinear(),
                        clone=True)
        def score_function(output):
            return output[:, category_index]
        cam = gradcam(score_function, 
                    seed_input=img_array,
                    penultimate_layer=layer_name)  # Use the name of the last convolutional layer
        cam = normalize(cam)
        heatmap = np.uint8(255 * cam[0])
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img_with_heatmap = cv2.addWeighted(img_array[0].astype('uint8'), 0.5, heatmap, 0.5, 0)

        return img_with_heatmap

    # preprocess images correctly before applying Grad-CAM
    def preprocess_image_for_gradcam(img, size=(256, 256)):
        img = img.resize(size, Image.ANTIALIAS)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    model = tf.keras.models.load_model('model.h5')

    # Saving the model architecture to a file
    with open('model_summary.txt', 'w') as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    # print(model.summary())

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
            frame_processed = preprocess_image_for_gradcam(frame_pil)

            # Predict using the model
            prediction = model.predict(frame_processed)
            predicted_label = np.argmax(prediction, axis=1)
            predicted_label_name = inverse_labels_dict[predicted_label[0]]
            print(f"Predicted label: {predicted_label_name}")

            layer_name_for_gradcam = 'conv5_block3_3_conv'
            grad_cam_img = apply_grad_cam(model, frame_processed, predicted_label[0], layer_name_for_gradcam)

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

    video_path = './videos/audiology/04137.mp4' 
    output_dir = './output_frames_audiology_test'
    predict_and_visualize(video_path, model, bbox_df, output_dir)

