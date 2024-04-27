import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
import optuna
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

# Dataset class for loading video frames as training data
class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.instances = self._load_instances()

    def _load_instances(self):
        instances = []
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            for video_dir in os.listdir(label_dir):
                video_path = os.path.join(label_dir, video_dir)
                frames = sorted(os.listdir(video_path))
                if len(frames) == 16:
                    instances.append((label, [os.path.join(video_path, frame) for frame in frames]))
        return instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        label, frame_paths = self.instances[idx]
        frames = [self.transform(Image.open(fp).convert('RGB')) for fp in frame_paths]
        video_tensor = torch.stack(frames)
        return video_tensor, label

# ConvLSTM cell and network definition
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)  # Correct padding calculation for a tuple
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.layers = nn.ModuleList([
            ConvLSTMCell(input_dim=self.input_dim if i == 0 else self.hidden_dim[i - 1],
                         hidden_dim=self.hidden_dim[i],
                         kernel_size=self.kernel_size[i],
                         bias=self.bias)
            for i in range(self.num_layers)
        ])

    def forward(self, input_tensor, hidden_state=None):
        current_input = input_tensor
        next_hidden = []
        seq_len = current_input.size(1)

        for layer_idx, layer in enumerate(self.layers):
            hidden_c = hidden_state[layer_idx] if hidden_state is not None else None
            output_inner = []
            for t in range(seq_len):
                hidden_c = layer(current_input[:, t, :, :, :], hidden_c)
                output_inner.append(hidden_c[0])
            current_input = torch.stack(output_inner, dim=1)
            next_hidden.append(hidden_c)

        return current_input, next_hidden

# Model integrating ConvLSTM with CNN for feature extraction
class VideoSignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(VideoSignLanguageModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer

        self.conv_lstm = ConvLSTM(input_dim=512, hidden_dim=[128, 64],
                                  kernel_size=[(3, 3), (3, 3)], num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, num_classes)  # Final classification layer

    def forward(self, x):
        bsz, seq_len, C, H, W = x.shape
        cnn_embeddings = [self.resnet(x[:, i]) for i in range(seq_len)]  # Apply CNN to each frame
        cnn_embeddings = torch.stack(cnn_embeddings, dim=1)  # Stack along sequence dimension
        
        # Apply ConvLSTM
        lstm_output, _ = self.conv_lstm(cnn_embeddings)
        # Use the output of the last time step
        last_time_step_output = lstm_output[:, -1, :, :, :]
        # Average pooling and flattening
        pooled_output = torch.mean(last_time_step_output, dim=[2, 3])
        output = self.fc(pooled_output)
        return output

# Function to setup data loaders
def setup_data_loaders(dataset_path):
    dataset = SignLanguageDataset(root_dir=dataset_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader

# Function to run Optuna for tuning
def run_optuna_tuning():
    train_loader, val_loader = setup_data_loaders('/app/rundir/CPSC542-FinalProject/images')
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=20)
    print("Best trial:", study.best_trial)
    final_model = train_final_model(train_loader, val_loader, study.best_trial.params)
    return final_model

# Define the Optuna objective function
def objective(trial, train_loader, val_loader):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    model = VideoSignLanguageModel(num_classes=2000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return train_and_evaluate(model, train_loader, val_loader, criterion, optimizer)

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer):
    model.train()
    for epoch in range(3):  # Short training for hyperparameter tuning
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    model.eval()
    accuracy = 0
    total = 0
    for inputs, labels in val_loader:
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    return accuracy / total

def objective(trial, train_loader, val_loader):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    model = VideoSignLanguageModel(num_classes=2000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return train_and_evaluate(model, train_loader, val_loader, criterion, optimizer)

def run_optuna_tuning():
    train_loader, val_loader = setup_data_loaders('/app/rundir/CPSC542-FinalProject/images')
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=20)
    print("Best trial:", study.best_trial)
    model = VideoSignLanguageModel(num_classes=2000)
    optimizer = optim.SGD(model.parameters(), lr=study.best_trial.params['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer)
    return model

def setup_data_loaders(dataset_path):
    dataset = SignLanguageDataset(root_dir=dataset_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader

def get_frames(video_path, num_frames=16, log_file='failed_videos.txt'):
    frames = []
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        with open(log_file, 'a') as f:
            f.write(f"Failed to open video file: {video_path}\n")
        return []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    for idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            with open(log_file, 'a') as f:
                f.write(f"Failed to read frame at index {idx} from {video_path}\n")
    video.release()
    return frames

def predict_from_video(video_path, model):
    frames = get_frames(video_path, num_frames=16)
    if not frames:
        return "Error processing video."
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame_tensors = torch.stack([transform(Image.fromarray(frame)) for frame in frames])
    frame_tensors = frame_tensors.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(frame_tensors)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Running everything
model = run_optuna_tuning()
video_path = '/app/rundir/CPSC542-FinalProject/input_video.mp4'
predicted_label = predict_from_video(video_path, model)
print(f'Predicted Sign Language Gloss: {predicted_label}')
