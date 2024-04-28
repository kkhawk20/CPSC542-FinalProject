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

'''
Reading in JSON file and creating key for ASL Video dataset
'''
main_path = "/app/rundir/CPSC542-FinalProject/archive-3/"
wlasl_df = pd.read_json(main_path + "WLASL_v0.3.json")
print(wlasl_df.head())

# Function to get the video ids from the json file
def get_videos_ids(json_list):
    video_ids = []
    for ins in json_list:
        if 'video_id' in ins:
            video_id = ins['video_id']
            if os.path.exists(f'{main_path}videos_raw/{video_id}.mp4'):
                video_ids.append(video_id)
    return video_ids

with open(main_path+'WLASL_v0.3.json', 'r') as data_file:
    json_data = data_file.read()
instance_json = json.loads(json_data)

# Creating a dataframe to store the features from the key file
features_df = pd.DataFrame(columns=['gloss', 'video_id'])
for row in wlasl_df.iterrows():
    ids = get_videos_ids(row[1][1])
    word = [row[1][0]] * len(ids)
    df = pd.DataFrame(list(zip(word, ids)), columns=features_df.columns)
    features_df = pd.concat([features_df, df], ignore_index=True)

print(features_df.head())

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
print(len(all_vids), len(all_labels), len(all_cats))
print(all_vids[:5], all_labels[:5], all_cats[:5])

labels_dict = {}
ind = 0
for label in all_cats:
    labels_dict[label] = ind
    ind += 1

with open('labels_dict.txt', 'w') as file:
    for key in labels_dict.keys():
        file.write(f"{key}: {labels_dict[key]}\n")
print("Saved to labels file!")



num_classes = 2000
unique_ids = [id_ for id_, label in zip(all_vids, all_labels) if labels_dict[label] < num_classes]
unique_labels = [label for id_, label in zip(all_vids, all_labels) if labels_dict[label] < num_classes]
print(len(unique_ids), len(unique_labels))

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
train_indx, test_indx = next(sss.split(unique_ids, unique_labels))

train_ids = [unique_ids[ind] for ind in train_indx]
train_labels = [unique_labels[ind] for ind in train_indx]
print(len(train_ids), len(train_labels)) 

test_ids = [unique_ids[ind] for ind in test_indx]
test_labels = [unique_labels[ind] for ind in test_indx]
print(len(test_ids), len(test_labels))
print(train_ids[:5], train_labels[:5])

from torch.utils.data import Dataset, DataLoader, Subset
import glob
from PIL import Image
import torch
import numpy as np
import random

np.random.seed(2020)
random.seed(2020)
torch.manual_seed(2020)

class VideoDataset(Dataset):
    def __init__(self, ids, labels, transform):      
        self.transform = transform
        self.ids = ids
        self.labels = labels
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        path2imgs=glob.glob(self.ids[idx]+"/*.jpg")
        path2imgs = path2imgs[:timesteps]
        label = labels_dict[self.labels[idx]]
        frames = []
        for p2i in path2imgs:
            frame = Image.open(p2i)
            frames.append(frame)
        
        seed = np.random.randint(1e9)        
        frames_tr = []
        for frame in frames:
            random.seed(seed)
            np.random.seed(seed)
            frame = self.transform(frame)
            frames_tr.append(frame)
        if len(frames_tr)>0:
            frames_tr = torch.stack(frames_tr)
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
print(len(train_ds))
imgs, label = train_ds[10]
imgs.shape, label, torch.min(imgs), torch.max(imgs)


test_transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ]) 
test_ds = VideoDataset(ids= test_ids, labels= test_labels, transform= test_transformer)
print(len(test_ds))
imgs, label = test_ds[5]
imgs.shape, label, torch.min(imgs), torch.max(imgs)

def create_dataset_arrays(data_loader):
    X = []
    Y = []
    for imgs, labels in data_loader:
        X.append(imgs)
        Y.append(labels)
    X = torch.cat(X)  # Concatenating list of tensors to a single tensor
    Y = torch.tensor(Y)
    return X, Y

# Assuming you have data loaders `train_loader` and `val_loader` set up:
X_train, Y_train = create_dataset_arrays(train_loader)
X_val, y_val = create_dataset_arrays(val_loader)

'''
NOW FOR THE MODEL STUFF!!!!
'''


