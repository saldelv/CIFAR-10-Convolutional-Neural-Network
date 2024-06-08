# Neural-Network-Pokemon-Identification
Nerual network to identify images of pokemon. Dataset used is found [here](https://www.kaggle.com/datasets/echometerhhwl/pokemon-gen-1-38914).

# Results
![image](https://github.com/saldelv/Neural-Network-Pokemon-Identification/assets/96501610/506d5988-480e-4298-9b67-cb9d21f5ccc2)
The highest test accuracy recieved was 56%. The training was done using 20 epochs and the non-augmented dataset was used as the test dataset
![10](https://github.com/saldelv/Neural-Network-Pokemon-Identification/assets/96501610/a4bcdf44-5290-4379-9f8f-06d37889c39d)
The highest test accuracy for only the first 10 classes was 87%, done in the same way

# How to Use
* Download the dataset and put the `data` folder in this directory
* Run `prepare_dataset.py` to format images and create labels
* Run `train.py` to train the neural network on the dataset
* Run `main.py` to easily upload your own image to identify with a confidence score, or run `test.py` to test on random dataset images or for overall accuracy
<img width="300" alt="p" src="https://github.com/saldelv/Neural-Network-Pokemon-Identification/assets/96501610/66f6cdd3-11b0-4275-a138-dce36da8f239">
