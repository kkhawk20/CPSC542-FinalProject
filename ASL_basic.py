# I am using the ASL MNIST dataset

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from keras.datasets import mnist
import keras as kb
from keras.models import Sequential
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from sklearn.preprocessing import LabelBinarizer

from plotnine import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import matplotlib.pyplot as plt
import os

save_dir = os.path.join(os.path.dirname(__file__), 'outputs')
data_dir = os.path.join(os.path.dirname(__file__), 'Data')

train_df = pd.read_csv(os.path.join(data_dir, 'sign_mnist_train/sign_mnist_train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'sign_mnist_test/sign_mnist_test.csv'))

# Separating X and Y
y_train = train_df['label']
y_test = test_df['label']

del train_df['label']
del test_df['label']

# rescale data to be 0-1 instead of 0-255
trainX = train_df.astype("float32") / 255.0
testX = test_df.astype("float32") / 255.0

# change the labels to be in the correct format
lb = LabelBinarizer()
trainY = lb.fit_transform(y_train)
testY = lb.transform(y_test)

# Visualize some images!!!
# I used different names cuz i wanted to reshape them without
# Changing the original data put into the model :)
x_train = train_df.values
x_test = test_df.values
x_train_vis = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
f, ax = plt.subplots(2,5)
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_train_vis[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()
    plt.show()

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# build structure of the model!! DCNN
# The images are naturally 28*28*1 shape, so input will be 784 (columns)
model = Sequential([
    # Convolutional layer: 32 filters, kernel size of 3x3, activation 'relu'
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Pooling layer: Max pooling with a 2x2 window
    MaxPooling2D((2, 2)),
    # Adding more convolutional layers to capture complex patterns
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    # Flatten the output to feed into a DNN
    Flatten(),
    # Dense layer
    Dense(64, activation='relu'),
    # Output layer
    Dense(24, activation='softmax')  # Adjust the number of units to match the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early = [kb.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)]

history = model.fit(x_train, trainY, epochs = 100, validation_data=(x_test, testY), callbacks = early)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig(os.path.join(save_dir, 'ASL_MNIST_Loss_Accuracy.png'))

# Evaluate the model
key = {0:'a', 1:'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f',
       6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k',
       11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p',
       16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u',
       21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'
       }

train_pred = model.predict(x_train)
train_pred_labels = np.argmax(train_pred, axis=1)

# Calculate accuracy for the training set
acc_train = accuracy_score(y_train, train_pred_labels)
f1_train = f1_score(y_train, train_pred_labels, average = 'micro')

# For the testing set, do the prediction and then calculate accuracy
test_pred = model.predict(x_test)
test_pred_labels = np.argmax(test_pred, axis=1)
acc_test = accuracy_score(y_test, test_pred_labels)
f1_test = f1_score(y_test, test_pred_labels, average = 'micro')

print(f"Training accuracy: {acc_train:.4f}")
print(f"Training F1: {f1_train:.4f}")
print(f"Testing accuracy: {acc_test:.4f}")
print(f"Testing F1: {f1_test:.4f}")

# Visualize some predictions!

# Predicting on test dataset
test_pred = model.predict(x_test)
test_pred_labels = np.argmax(test_pred, axis=1)

# Actual labels in numeric form
true_labels = np.argmax(testY, axis=1)

# Mapping numeric labels back to letters using 'key' dictionary
mapped_true_labels = [key[label] for label in true_labels]
mapped_pred_labels = [key[label] for label in test_pred_labels]

# Select a few random images from the test set to display
num_images = 5
random_indices = np.random.choice(x_test.shape[0], num_images, replace=False)

# Setup for a 1x5 grid
fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 15))

for ax, idx in zip(axes.flat, random_indices):
    # Reshape the image for display
    image = x_test.iloc[idx].values.reshape(28, 28)  # Reshape back to 28x28 for display
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Actual: {mapped_true_labels[idx]}\nPredicted: {mapped_pred_labels[idx]}')
    ax.axis('off')

plt.tight_layout()
plt.show()
plt.savefig('app/rundir/test_rundir/CPSC542-FinalProject/Results/ASL_MNIST_Predictions.png')