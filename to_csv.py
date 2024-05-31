import os
import csv
import pandas as pd

dir = 'datasetfolders'
rows = []
index = 0

for fname in os.listdir(dir):
    path = os.path.join(dir, fname)
    if os.path.isdir(path):
        for filename in os.listdir(path):
            rows.append([filename, index])
    index += 1

with open("labels.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)