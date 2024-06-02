import os
from torch import load, max, no_grad
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from train import CustomImageDataset, transform, batch_size, Net, classes

# tests batch of images from dataset
def test_batch():
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

    outputs = net(images)
    _, predicted = max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))

# tests overall accuracy
def test_accuracy():
    correct = 0
    total = 0

    with no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on test dataset: {100 * correct // total} %')

# tests accuracy of each class
def test_accuracy_classes():
    correct = {classname: 0 for classname in classes}
    total = {classname: 0 for classname in classes}

    with no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = max(outputs, 1)
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct[classes[label]] += 1
                total[classes[label]] += 1
    for classname, correct_count in correct.items():
        accuracy = 100 * float(correct_count) / total[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

if __name__ == "__main__":

    # Creating dataset
    test_data = CustomImageDataset(
        annotation_file='labels.csv',
        img_dir='data_merged',
        transform = transform,
        target_transform = None
    )

    # Creating dataloader
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Loading trained model
    net = Net()
    net.load_state_dict(load('cifar_net.pth'))

    # Calls functions, can comment out for select tests
    test_batch()
    test_accuracy()
    test_accuracy_classes()