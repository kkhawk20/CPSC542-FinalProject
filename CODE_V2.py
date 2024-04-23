import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

def get_labels():
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

    features_df = pd.DataFrame(columns=['gloss', 'video_id'])
    for row in wlasl_df.iterrows():
        ids = get_videos_ids(row[1][1])
        word = [row[1][0]] * len(ids)
        df = pd.DataFrame(list(zip(word, ids)), columns=features_df.columns)
        features_df = pd.concat([features_df, df], ignore_index=True)

    return features_df

# Utility functions
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

# Dataset class
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

# Model definition
class ResNet18RNN(nn.Module):
    def __init__(self, num_classes, pretrained=True, hidden_size=256, num_layers=1):
        super().__init__()
        self.base_model = models.resnet18(pretrained=pretrained)
        self.base_model.fc = nn.Identity()
        self.rnn = nn.LSTM(self.base_model.fc.in_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        bsz, seq_len, C, H, W = x.size()
        cnn_emb = torch.cat([self.base_model(x[:, i]) for i in range(seq_len)], dim=1)
        _, (hn, _) = self.rnn(cnn_emb.view(bsz, seq_len, -1))
        out = self.fc(hn[-1])
        return out
    
import optuna
import torch.optim as optim
from torch import nn

def objective(trial):
    # Hyperparameters to be tuned by Optuna
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    rnn_hidden_size = trial.suggest_categorical('rnn_hidden_size', [128, 256, 512])
    rnn_num_layers = trial.suggest_int('rnn_num_layers', 1, 3)

    # Model initialization
    model = ResNet18RNN(num_classes=25, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Training logic
    for epoch in range(3):  # using fewer epochs for the tuning phase
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Simple evaluation logic for tuning
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


# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return model

# Prediction function
def predict_from_video(video_path, model, frame_count=16):
    frames = get_frames(video_path, num_frames=frame_count)
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
        predicted_label = index_to_label[predicted.item()]  # Ensure index_to_label is defined
    return predicted_label

# Main execution function
def run_model():
    dataset_path = '/app/rundir/CPSC542 - FINALPROJECT/images'
    num_classes = 2000  # Adjust according to actual dataset
    dataset = SignLanguageDataset(root_dir=dataset_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader}
    model = ResNet18RNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=20)
    return model

# Running everything
model = run_model()
video_path = '/app/rundir/CPSC542 - FINALPROJECT/input_video.mp4'
predicted_label = predict_from_video(video_path, model)
print(f'Predicted Sign Language Gloss: {predicted_label}')

# Grabbing the features_df to look at classes
features_df = get_labels()
index_to_label = {idx: label for idx, label in enumerate(features_df['gloss'].unique())}
print(index_to_label)
