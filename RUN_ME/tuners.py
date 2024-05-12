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
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader, Subset
import glob
import random
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score

    
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

def ASL_model(train_generator, val_generator, tuner, retrain=False):
    hypermodel = CNNLSTMHyperModel((256, 256, 3), 2000)
    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    if retrain:
        tuner.search(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stopping])
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.save('best_model.h5')
        history = best_model.history
        # print(best_model.summary())

    return history


