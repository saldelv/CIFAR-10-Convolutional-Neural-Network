import tkinter
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import torch
from torch import nn as nn
from torch import load, max
from torchvision.io import read_image, ImageReadMode
import numpy
from math import trunc
from train import Net
from constants import *

# Function for uploading image
def UploadAction(event=None):
    filepath = filedialog.askopenfilename()
    if filepath.endswith((".png", ".jpg")):
        image = Image.open(filepath)
        image = image.resize((400, 400), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(image)
        image_label.image = img
        image_label.configure(image=img)
        identify(filepath)
    else:
        prediction_label.configure(text="Invalid file type, please use .png or .jpg", fg="red")
        image_label.configure(image="")

# Identifies image
def identify(filepath):
    # Adding image
    image = read_image(filepath, ImageReadMode.RGB)
    image = transform_test(image)
    image = image.float().unsqueeze(0)

    # Loading model
    net = Net()
    net.eval()
    net.load_state_dict(load('cifar_net.pth'))
    outputs = net(image)
    soft_out = torch.softmax(outputs, dim=1)
    conf, predicted = max(soft_out, 1)
    conf_number = conf[0].detach().numpy() * 100
    conf_number = trunc(float(conf_number) * 10 ** 3) / 10 ** 3
    message = 'Prediction: ' + ''.join(f'{classes[predicted[0]]:5s}') + '\nConfidence: ' + str(conf_number) + '%'
    prediction_label.configure(text=message, fg="black")

# Creating gui
root = tkinter.Tk()

label = Label(root, text='Upload Image of a Generation 1 Pokemon:', font=(32))
label.pack()
button = Button(root, text='Upload', command=UploadAction, height=1, width=20)
button.pack()
image_label = Label(image="")
image_label.pack()
prediction_label = Label(root, text="", font=(32))
prediction_label.pack()

root.mainloop()