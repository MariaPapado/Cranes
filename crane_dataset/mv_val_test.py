###split some train image patches for validation

import os
import subprocess
import random

ids = os.listdir('./CRANE_dataset-split/images/train')

print(len(ids))

random.shuffle(ids)

val_ids = ids[:140]
for id in val_ids:
    subprocess.run(['mv', './CRANE_dataset-split/images/train/{}.png'.format(id[:-4]), './CRANE_dataset-split/images/val/'])
    subprocess.run(['mv', './CRANE_dataset-split/labels/train/{}.txt'.format(id[:-4]), './CRANE_dataset-split/labels/val/'])
