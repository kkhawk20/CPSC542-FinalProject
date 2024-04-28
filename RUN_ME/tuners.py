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

def ASL_model():
    '''
    NOW FOR THE MODEL STUFF!!!!
    '''
    print("MODEL BUILDING!!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class CNNLSTM(nn.Module):
        def __init__(self, num_classes):
            super(CNNLSTM, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1),  # Larger kernel and stride
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Standard pooling
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),  # Increase to 32 filters, larger stride
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Another pooling layer
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Further increase in channels, and stride
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Additional pooling
                nn.Flatten()
            )
            # Linear layer will now need adjustment based on new output size from the CNN
            # Assuming the input images are 224x224, calculate the output size after the last layer
            self.fc1 = nn.Linear(self._get_conv_output((3, 224, 224)), 256)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
            self.fc2 = nn.Linear(256, num_classes)

        def _get_conv_output(self, shape):
            input = torch.rand(1, *shape)
            output_feat = self.cnn(input)
            n_size = output_feat.data.view(1, -1).size(1)
            return n_size

        def forward(self, x):
            batch_size, timesteps, C, H, W = x.size()
            c_out = []
            for t in range(timesteps):
                c_in = x[:, t, :, :, :]
                c_out_t = self.cnn(c_in)
                c_out_t = self.fc1(c_out_t)
                c_out_t = self.relu(c_out_t)
                c_out.append(c_out_t)

            c_out = torch.stack(c_out, dim=1)
            r_out, (h_n, c_n) = self.lstm(c_out)
            out = self.fc2(r_out[:, -1, :])
            return out



    def my_collate(batch):
        # Extract frames and labels from the batch
        frames, labels = zip(*batch)
        # Pad frames to have the same sequence length
        frames_padded = pad_sequence(frames, batch_first=True, padding_value=0)
        # Stack labels into a tensor
        labels = torch.tensor(labels)
        return frames_padded, labels

    # Use custom collate function in DataLoader
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, collate_fn=my_collate)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=my_collate)

    model = CNNLSTM(num_classes=2000).to(device)  # Assuming num_classes is 2000
    criterion = nn.CrossEntropyLoss()  # Assuming a classification task
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_losses = []
    training_accuracies = []
    validation_accuracies = []

    print("MODEL BUILT - STARTING TRAINING!")
    num_epochs = 10  # Number of epochs
    for epoch in range(num_epochs):
        print("Training Epoch: ", epoch+1)
        model.train()  # Set model to training mode
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct / total
        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # Evaluate the model on the test set
        model.eval()  # Set model to evaluation mode
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        validation_accuracy = accuracy_score(all_labels, all_predictions) * 100
        validation_accuracies.append(validation_accuracy)
        print(f'Test Accuracy of the model on the test images: {validation_accuracy:.2f}%')

    # Plotting the results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('results.png')