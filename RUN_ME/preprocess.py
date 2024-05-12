'''
IMPORTS
'''
import json
import os
import shutil
import numpy as np
import pandas as pd
import torch
import copy
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import random_split

def data_preprocess():
        
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

    return train_generator, val_generator, tuner