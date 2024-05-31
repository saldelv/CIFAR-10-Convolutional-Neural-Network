import os
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(255),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 1

class CustomImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform, target_transform):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
training_data = CustomImageDataset(
    annotation_file='labels.csv',
    img_dir='dataset',
    transform = transform,
    target_transform = None
)

test_data = CustomImageDataset(
    annotation_file='labels.csv',
    img_dir='dataset',
    transform = transform,
    target_transform = None
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img.permute(1, 2, 0))
plt.show()
print(f"Label: {label}")