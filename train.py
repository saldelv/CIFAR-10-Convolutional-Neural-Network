import os
import pandas as pd
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision.transforms as transforms 
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt

# Transformer
transform = transforms.Compose(
    [transforms.ToPILImage('RGB'),
     transforms.Resize([255, 255]),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Batch size
batch_size = 4

# Images classes
classes = ('Abra', 'Aerodactyl', 'Alakazam', 'Arbok', 'Arcanine', 'Articuno', 'Beedril', 'Bellsprout', 'Blastoise', 'Bulbasaur')

# Custom dataset class
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
        image = read_image(img_path, ImageReadMode.RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# Neural network class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(57600, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == "__main__":

    # Creating dataset
    training_data = CustomImageDataset(
        annotation_file='labels.csv',
        img_dir='data_merged',
        transform = transform,
        target_transform = None
    )

    # Creating dataloader
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        
    net = Net()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training network
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Saving model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)