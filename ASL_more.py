import json
import os
import shutil
import numpy as np
import pandas as pd

main_path = "/app/rundir/CPSC542-FinalProject/archive-3/"
wlasl_df = pd.read_json(main_path + "WLASL_v0.3.json")

print(wlasl_df.head())

def get_videos_ids(json_list):
    """
    check if the video id is available in the dataset
    and return the viedos ids of the current instance
    
    Args:
        json_list: Instance of video metadata.
        
    Returns:
        List of video ids. 
    """
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

# get_videos_ids(instance_json[0]['instances'])

# len(get_videos_ids(instance_json[0]['instances']))

# wlasl_df["video_ids"] = wlasl_df["instances"].apply(get_videos_ids)

features_df = pd.DataFrame(columns=['gloss', 'video_id'])
for row in wlasl_df.iterrows():
    ids = get_videos_ids(row[1][1])
    word = [row[1][0]] * len(ids)
    df = pd.DataFrame(list(zip(word, ids)), columns=features_df.columns)
    features_df = pd.concat([features_df, df], ignore_index=True)

# print(features_df)

# def move_videos_to_subdir(dataframe):
#     for label in dataframe["gloss"].unique():
#         dst_path = f'videos/{label}'
#         os.makedirs(dst_path, exist_ok=True)
        
#         for video in dataframe.loc[dataframe["gloss"] == label]["video_id"]:
#             src = f'{main_path}videos_raw/{video}.mp4'
#             dst = f'{dst_path}/{video}.mp4'
            
#             try:
#                 shutil.copyfile(src, dst)
#             except IOError as e:
#                 print(f"Could not copy file {src} to {dst}. Error: {e}")
#             except Exception as e:
#                 print(f"Unexpected error occurred while copying file {src} to {dst}. Error: {e}")

# move_videos_to_subdir(features_df)

# os.listdir('videos/')

# def create_empty_subdirs(dataframe, dst_root):
#     for label in dataframe["gloss"].unique():
#         dst_path = os.path.join(dst_root, label)
#         os.makedirs(dst_path, exist_ok=True)

# # Example usage:
# dst_root = 'images'
# create_empty_subdirs(features_df, dst_root)

import os
import torch
import copy
from tqdm import tqdm_notebook
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
from tqdm import tqdm_notebook

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

def denormalize(x_, mean, std):
    x = x_.clone()
    for i in range(3):
        x[i] = x[i]*std[i]+mean[i]
    x = to_pil_image(x)        
    return x

def train_val(model, params):
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    
    loss_history={
        "train": [],
        "val": [],
    }
    
    metric_history={
        "train": [],
        "val": [],
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
        
        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" %(train_loss,val_loss,100*val_metric))
        print("-"*10) 
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

# get learning rate 
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b
    

def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in tqdm_notebook(dataset_dl):
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb)
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        running_loss+=loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric


def plot_loss(loss_hist, metric_hist):

    num_epochs= len(loss_hist["train"])

    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()

    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1), metric_hist["train"],label="train")
    plt.plot(range(1,num_epochs+1), metric_hist["val"],label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.show()
    plt.savefig('loss.png')
    
from torch import nn
class Resnt18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnt18Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate= params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]
        
        baseModel = models.resnet18(pretrained=pretrained)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:,ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x    

from torchvision import models
from torch import nn

def get_model(num_classes, model_type="rnn"):
    if model_type == "rnn":
        params_model={
            "num_classes": num_classes,
            "dr_rate": 0.1,
            "pretrained" : True,
            "rnn_num_layers": 1,
            "rnn_hidden_size": 100,}
        model = Resnt18Rnn(params_model)        
    else:
        model = models.video.r3d_18(pretrained=True, progress=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)    
    return model

import cv2
import numpy as np
def get_frames(filename, n_frames=1, log_file='failed_videos.txt'):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    if not v_cap.isOpened():
        with open(log_file, 'a') as f:
            f.write(f"Error opening video file: {filename}\n")
        print(f"Error opening video file {filename}")
        return frames, 0
    
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len < n_frames:  # Check if there are enough frames in the video
        with open(log_file, 'a') as f:
            f.write(f"Not enough frames in video file: {filename}\n")
        print(f"Not enough frames in {filename}")
        v_cap.release()
        return frames, 0

    frame_list = np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    
    for fn in frame_list:
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        success, frame = v_cap.read()
        if not success:
            with open(log_file, 'a') as f:
                f.write(f"Failed to get frame {fn} from video file: {filename}\n")
            print(f"Failed to get the frame {fn} from {filename}")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frames.append(frame)
    v_cap.release()
    return frames, v_len

import torchvision.transforms as transforms
from PIL import Image
def transform_frames(frames, model_type="rnn"):
    if model_type == "rnn":
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]

    test_transformer = transforms.Compose([
                transforms.Resize((h,w)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]) 

    frames_tr = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame_tr = test_transformer(frame)
        frames_tr.append(frame_tr)
    imgs_tensor = torch.stack(frames_tr)    

    if model_type=="3dcnn":
        imgs_tensor = torch.transpose(imgs_tensor, 1, 0)
    imgs_tensor = imgs_tensor.unsqueeze(0)
    return imgs_tensor

def store_frames(frames, path2store):
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        path2img = os.path.join(path2store, "frame"+str(ii)+".jpg")
        cv2.imwrite(path2img, frame)

import os

sub_folder = "videos"
sub_folder_jpg = "images"
path2aCatgs = sub_folder

listOfCategories = os.listdir(path2aCatgs)
len(listOfCategories)

for cat in listOfCategories:
    path2acat = os.path.join(path2aCatgs, cat)
    listOfSubs = os.listdir(path2acat)

extension = ".mp4"
n_frames = 16

# # This is used for data engineering 
# for root, dirs, files in os.walk(path2aCatgs, topdown=False):
#     for name in files:
#         if extension not in name:
#             continue
#         path2vid = os.path.join(root, name)
#         frames, vlen = get_frames(path2vid, n_frames= n_frames, log_file = 'app/rundir/CPSC542-FinalProject/failed_videos.txt')
#         if vlen == 0:  # Indicates that the video file couldn't be processed
#             print(f"Skipping video {path2vid} due to errors.")
#             continue
#         path2store = path2vid.replace(sub_folder, sub_folder_jpg)
#         path2store = path2store.replace(extension, "")
#         print(path2store)
#         os.makedirs(path2store, exist_ok= True)
#         store_frames(frames, path2store)
#     print("-"*50)   

# This is going to run the model and such
path2ajpgs = sub_folder_jpg
ids, labels, listOfCategories = get_vids(path2ajpgs)
len(ids), len(labels), len(listOfCategories)

from sklearn.model_selection import train_test_split
ids_train, ids_val, labels_train, labels_val = train_test_split(ids, labels, test_size=0.1, random_state=42)
len(ids_train), len(ids_val)

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, ids, labels, transform=None):
        self.ids = ids
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        path2img = self.ids[idx]
        img = Image.open(path2img)
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
    
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transformer = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_ds = CustomDataset(ids_train, labels_train, transform=train_transformer)
val_ds = CustomDataset(ids_val, labels_val, transform=val_transformer)

from torch.utils.data import DataLoader
batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from torchvision import models
import torch

model = get_model(num_classes=len(listOfCategories), model_type="rnn")
model = model.to(device)

loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

params_train={
    "num_epochs": 10,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": "app/rundir/CPSC542-FinalProject/weights.pt",
}

model, loss_hist, metric_hist = train_val(model, params_train)
plot_loss(loss_hist, metric_hist)

