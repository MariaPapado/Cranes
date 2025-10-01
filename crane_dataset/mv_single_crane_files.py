####XWRIZE TIS EIKONES POU PERIEXOUN MONO ENA CRANE

import cv2
import os
import numpy as np
from ultralytics.data.split_dota import split_test, split_trainval
import math
import random
import subprocess

# Function to read annotations
def read_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            # Extract coordinates (as float)
            coords = list(map(float, parts[1:]))
            annotations.append((class_id, coords))
    return annotations

############@@@@@@@@@@@@@@@@@@@@@@@@STILL at one point the bbox is outside the patch

# Define paths
image_dir = "./IMAGES/"  # Change to your images directory
label_dir = "./LABELS/"  # Change to your labels directory

cnt=0
# Example usage
ids = os.listdir(label_dir)
print('lll', len(ids))
for id in ids:
    #print(id)
#    img = cv2.imread('./IMAGES/{}'.format(id[:-4]+'.png'))
#    print('imgshape', img.shape)
    annotations = read_annotations('./LABELS/{}'.format(id))
    print(len(annotations))
    if len(annotations)==1:
        cnt=cnt+1

        subprocess.run(['mv', './IMAGES/{}.png'.format(id[:-4]), './images_single_crane/'])
        subprocess.run(['mv', './LABELS/{}.txt'.format(id[:-4]), './labels_single_crane/'])

print(cnt)
