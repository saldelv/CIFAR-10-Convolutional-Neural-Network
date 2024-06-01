import os
import csv
from PIL import Image

dir = 'datasetfolders'
targetdir = 'dataset'
rows = []
index = 0

for fname in os.listdir(dir):
    path = os.path.join(dir, fname)
    if os.path.isdir(path):
        for filename in os.listdir(path):

            # Turns images to 8 bit depth for pytorch
            image = Image.open(path + "/" + filename)
            if image.format == 'PNG':
                image = image.quantize(colors=256, method=2)
            image.save(targetdir + "/" + filename)

            rows.append([filename, index])
    index += 1

with open("labels.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)