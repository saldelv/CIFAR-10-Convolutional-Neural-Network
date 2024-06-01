import os
from torch import load, max
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from train import CustomImageDataset, transform, batch_size, Net

# Images classes
classes = ('Abra', 'Aerodactyl', 'Alakazam', 'Arbok', 'Arcanine', 'Articuno', 'Beedril', 'Bellsprout', 'Blastoise', 'Bulbasaur')

# Creating dataset
test_data = CustomImageDataset(
    annotation_file='labels.csv',
    img_dir='datasettest',
    transform = transform,
    target_transform = None
)

# Creating dataloader
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Showing test images
dataiter = iter(test_dataloader)
images, labels = next(dataiter)

grid_img = make_grid(images)
grid_img.shape
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Loading trained model
net = Net()
net.load_state_dict(load('cifar_net.pth'))
outputs = net(images)
_, predicted = max(outputs, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))