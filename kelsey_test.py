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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50
import torch.nn.functional as F
import torchvision.transforms as transforms


'''
Reading in JSON file and creating key for ASL Video dataset
'''
main_path = "/app/rundir/CPSC542-FinalProject/archive-3/"
wlasl_df = pd.read_json(main_path + "WLASL_v0.3.json")
# print(wlasl_df.head())

# Function to get the video ids from the json file
def get_videos_ids(json_list):
    video_ids = []
    for ins in json_list:
        if 'video_id' in ins:
            video_id = ins['video_id']
            if os.path.exists(f'{main_path}videos_raw/{video_id}.mp4'):
                video_ids.append(video_id)
    return video_ids

# Reading in the JSON file
with open(main_path + 'WLASL_v0.3.json', 'r') as data_file:
    json_data = json.load(data_file)

# Extracting bounding box data
bbox_data = []
for entry in json_data:
    gloss = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        bbox = instance['bbox'] if 'bbox' in instance else None
        frame_start = instance['frame_start']
        frame_end = instance['frame_end']
        signer_id = instance['signer_id']
        source = instance['source']
        split = instance['split']
        url = instance['url']
        variation_id = instance['variation_id']
        
        bbox_data.append({
            'video_id': video_id,
            'gloss': gloss,
            'bbox': bbox,
            'frame_start': frame_start,
            'frame_end': frame_end,
            'signer_id': signer_id,
            'source': source,
            'split': split,
            'url': url,
            'variation_id': variation_id
        })

# Creating a DataFrame from the extracted data
bbox_df = pd.DataFrame(bbox_data)
bbox_df.set_index('video_id', inplace=True)

# Now bbox_df is a DataFrame where each video_id is mapped to its bounding box and other metadata

# RUN FROM HERE DOWN IF YOU ARE NOT KELSEY :)

# Creating a function that gets the videos from the dataset of videos
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_vids(path2ajpgs):
    listOfCats = os.listdir(path2ajpgs)
    ids = []
    labels = []
    for catg in listOfCats:
        path2catg = os.path.join(path2ajpgs, catg)
        listOfSubCats = os.listdir(path2catg)
        path2subCats= [os.path.join(path2catg,los) for los in listOfSubCats]
        ids.extend(path2subCats)
        labels.extend([catg]*len(listOfSubCats))
    return ids, labels, listOfCats 

# Creating a dictionary to hold all 2000 labels and indexes for references
sub_folder_jpg = 'images'
path2ajpgs = sub_folder_jpg

all_vids, all_labels, all_cats = get_vids(path2ajpgs)
# print(len(all_vids), len(all_labels), len(all_cats))
# print(all_vids[:5], all_labels[:5], all_cats[:5])

labels_dict = {}
ind = 0
for label in all_cats:
    labels_dict[label] = ind
    ind += 1

with open('labels_dict.txt', 'w') as file:
    for key in labels_dict.keys():
        file.write(f"{key}: {labels_dict[key]}\n")
# print("Saved to labels file!")

num_classes = 2000
unique_ids = [id_ for id_, label in zip(all_vids, all_labels) if labels_dict[label] < num_classes]
unique_labels = [label for id_, label in zip(all_vids, all_labels) if labels_dict[label] < num_classes]
# print(len(unique_ids), len(unique_labels))

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
train_indx, test_indx = next(sss.split(unique_ids, unique_labels))

train_ids = [unique_ids[ind] for ind in train_indx]
train_labels = [unique_labels[ind] for ind in train_indx]
# print(len(train_ids), len(train_labels)) 

test_ids = [unique_ids[ind] for ind in test_indx]
test_labels = [unique_labels[ind] for ind in test_indx]
# print(len(test_ids), len(test_labels))
# print(train_ids[:5], train_labels[:5])

np.random.seed(2020)
random.seed(2020)
torch.manual_seed(2020)

# This is creating a dataset of the videos utilizing a sliding window sampling approach
# This crops the video images into the bounding boxes
class VideoDataset(Dataset):
    def __init__(self, ids, labels, transform, bbox_df=None, sequence_length=16, step=8):
        self.transform = transform
        self.bbox_df = bbox_df
        self.data = []
        self.labels = []

        # Prepare data with sliding window
        for idx, video_id in enumerate(ids):
            label_idx = labels_dict.get(labels[idx], -1)  # Use get to avoid KeyError
            if label_idx == -1:
                continue  # Skip if label is not found in labels_dict
            frame_paths = sorted(glob.glob(f"{video_id}/*.jpg"))
            for start in range(0, len(frame_paths) - sequence_length + 1, step):
                sequence_frames = frame_paths[start:start + sequence_length]
                self.data.append(sequence_frames)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame_paths = self.data[idx]
        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            video_id = os.path.basename(os.path.dirname(frame_path))
            bbox = self.bbox_df.loc[video_id]['bbox'] if self.bbox_df is not None and video_id in self.bbox_df.index else None
            if bbox:
                frame = frame.crop([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            frame = self.transform(frame)
            frames.append(frame)

        frames_tr = torch.stack(frames)
        return frames_tr, self.labels[idx]


model_type = "rnn"    

timesteps =16
if model_type == "rnn":
    h, w =224, 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
else:
    h, w = 112, 112
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

train_transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


train_ds = VideoDataset(ids=train_ids, labels=train_labels, transform=train_transformer, bbox_df=bbox_df, sequence_length=16, step=8)
# print(len(train_ds))
imgs, label = train_ds[10]
# print(imgs.shape, label, torch.min(imgs), torch.max(imgs))

test_transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ]) 
test_ds = VideoDataset(ids= test_ids, labels= test_labels, transform= test_transformer)
# print(len(test_ds))
imgs, label = test_ds[5]
# print(imgs.shape, label, torch.min(imgs), torch.max(imgs))

'''
NOW FOR THE MODEL STUFF!!!!
'''
print("MODEL BUILDING!!")

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.5):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_out = self.resnet(x.view(batch_size * timesteps, C, H, W))
        c_out = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, c_n) = self.lstm(c_out)
        out = self.fc(r_out[:, -1, :])
        return out

    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)
        output_feat = self.cnn(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

def my_collate(batch):
    # Extract frames and labels from the batch
    frames, labels = zip(*batch)
    # Pad frames to have the same sequence length
    frames_padded = pad_sequence(frames, batch_first=True, padding_value=0)
    # Stack labels into a tensor
    labels = torch.tensor(labels)
    return frames_padded, labels

# Use custom collate function in DataLoader
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=my_collate, num_workers=8)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=8, collate_fn=my_collate)

# Setting the GPU utilization ;)
model = CNNLSTM(num_classes=2000)
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

'''
Early Stopping Class
'''
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        if self.verbose:
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')


print("MODEL BUILT - STARTING TRAINING!")

scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
early_stopping = EarlyStopping(patience=10, verbose=True)

training_losses = []
training_accuracies = []
validation_accuracies = []
validation_losses = []

num_epochs = 100  # Number of epochs
for epoch in range(num_epochs):
    print("Training Epoch: ", epoch+1)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    training_losses.append(running_loss / len(train_loader.dataset))
    training_accuracies.append(100 * correct / total)

    # Validation phase
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    validation_losses.append(validation_loss / len(test_loader.dataset))
    validation_accuracies.append(100 * correct / total)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {training_losses[-1]:.4f}, Train Acc: {training_accuracies[-1]:.2f}%, Val Loss: {validation_losses[-1]:.4f}, Val Acc: {validation_accuracies[-1]:.2f}%')

    # Reduce learning rate if needed
    scheduler.step(validation_losses[-1])
    # Check for early stopping
    early_stopping(validation_losses[-1], model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# Save the model & weights
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")
print("Training completed!")

# Plotting the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
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

import cv2

def predict_and_visualize(video_path, model, bbox_df):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the captured frame to PIL Image and then to Tensor
        frame_pil = Image.fromarray(frame)
        # Assume the bounding box data is available in bbox_df for this frame
        video_id = os.path.basename(video_path).split('.')[0]
        if video_id in bbox_df.index:
            bbox = bbox_df.loc[video_id]['bbox']
            # Crop and transform the frame using the bounding box
            frame_pil = frame_pil.crop(bbox)
        
        # Transform the frame to tensor
        frame_tensor = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])(frame_pil).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(frame_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label_name = labels_dict.inverse_transform([predicted.item()])[0]

        # Draw the bounding box and label on the frame
        if bbox:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            color = (0, 255, 0) # Green color
            thickness = 2
            frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, predicted_label_name, (bbox[0], bbox[1] - 10), font, 0.9, color, thickness)
        
        # Display the frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode
predict_and_visualize('./input_video.mp4', model, bbox_df)
print("Prediction and visualization completed!")