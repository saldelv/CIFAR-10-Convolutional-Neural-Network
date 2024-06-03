import tkinter
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from torch import nn as nn
from torch import load, max
from torchvision.io import read_image, ImageReadMode
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
    image = transform(image)
    image = image.float().unsqueeze(0)

    # Loading model
    net = Net()
    net.eval()
    net.load_state_dict(load('cifar_net.pth'))
    outputs = net(image)
    _, predicted = max(outputs, 1)
    prediction_label.configure(text='Prediction: ' + ''.join(f'{classes[predicted[0]]:5s}'), fg="black")

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