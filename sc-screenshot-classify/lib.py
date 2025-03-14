from dataclasses import dataclass
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


@dataclass
class Rect:
    """Specifies the region of the image we're interested in. 
    Resolution is a width:height of the image."""
    top: float
    left: float
    width: float
    height: float
    resolution: tuple = (2560, 1440)

    @property
    def right(self):
        return self.left + self.resolution[0] - self.width

    @property
    def bottom(self):
        return self.top + self.resolution[1] - self.height



class CustomDataset(Dataset):
    """Takes care of loading images from disk & transforming them.
    Makes use of predefined classes: 'other' and 'starmap'.
    """
    def __init__(self, root_dir, crop_rect: Rect, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['other', 'starmap']
        self.file_paths = []
        self.crop_rect = crop_rect
        
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for file in os.listdir(class_path):
                self.file_paths.append(
                    (os.path.join(class_path, file), self.classes.index(class_name))
                )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path, label = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        rect = self.crop_rect
        image = image.crop((rect.left, rect.top, rect.right, rect.bottom))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

@dataclass
class Session:
    transform: torchvision.transforms.transforms.Compose
    crop_region: Rect
    optimizer: object
    criterion: object
    device: object

@dataclass
class Data:
    dataset: CustomDataset
    batch_size: int
    shuffle: bool

    @property
    def train_size(self):
        return int(0.8 * len(self.dataset))

    @property
    def val_size(self):
        return len(self.dataset) - self.train_size

    def dataset_split(self):
         return torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size])

@dataclass
class TrainingResult:
    model: object
    loader_val: object

def train(sess: Session, data: Data, model, num_epochs=5):
    dataset_train, dataset_val = data.dataset_split()
    loader_train = DataLoader(dataset_train, batch_size=data.batch_size, shuffle=data.shuffle)
    loader_val = DataLoader(dataset_val, batch_size=data.batch_size, shuffle=data.shuffle)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2) # 2 output classes
    device = sess.device
    device_model = model.to(device)

    optimizer = sess.optimizer(device_model)
    criterion = sess.criterion()
    
    for epoch in range(num_epochs):
        device_model.train()
        running_loss = 0.0

        for inputs, labels, in loader_train:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = device_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset_train)
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}')

    return TrainingResult(model=device_model, loader_val=loader_val)

def evaluate(result: TrainingResult, device, classes: [str]):
    model, val_loader = result.model, result.loader_val
    
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=classes))
    
