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

class VideoDataset(Dataset):
    def __init__(self, ids, labels, transform, bbox_df=None):      
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.bbox_df = bbox_df
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        video_id = self.ids[idx]
        label = labels_dict[self.labels[idx]]
        
        # Retrieve bbox information if available
        if self.bbox_df is not None and video_id in self.bbox_df.index:
            bbox_info = self.bbox_df.loc[video_id]
            bbox = bbox_info['bbox']  # This assumes that 'bbox' is a list [x, y, width, height]
            # Convert bbox from [x, y, width, height] to [x_min, y_min, x_max, y_max]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        else:
            bbox = None
        
        frame_paths = sorted(glob.glob(f"{video_id}/*.jpg"))[:timesteps]
        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            # Crop the frame using the bounding box if available
            if bbox:
                frame = frame.crop(bbox)
            frame = self.transform(frame)
            frames.append(frame)

        frames_tr = torch.stack(frames) if frames else torch.empty(0)
        return frames_tr, label

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

import torchvision.transforms as transforms

train_transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])  

train_ds = VideoDataset(ids= train_ids, labels= train_labels, transform= train_transformer)
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
validation_losses = []

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

    validation_loss = criterion(outputs, labels)
    validation_losses.append(validation_loss)

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

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")

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