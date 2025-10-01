import os
import subprocess
import random

ids = os.listdir('./CRANE_dataset/images/train/')
#ids = os.listdir('./images_single_crane/')

print(len(ids))

random.shuffle(ids)

test_ids = ids[:8]
#test_ids = ids[:16]

print(len(test_ids))

for id in test_ids:
    subprocess.run(['mv', './CRANE_dataset/images/train/{}.png'.format(id[:-4]), './TEST/images/'])
    subprocess.run(['mv', './CRANE_dataset/labels/train/{}.txt'.format(id[:-4]), './TEST/labels/'])

#    subprocess.run(['mv', './images_single_crane/{}.png'.format(id[:-4]), './TEST/images/'])
#    subprocess.run(['mv', './labels_single_crane/{}.txt'.format(id[:-4]), './TEST/labels/'])


