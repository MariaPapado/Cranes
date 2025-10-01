##to split ap to multiple crane images vgazei patches se jpg gi auto

import os
from PIL import Image
import subprocess
import numpy as np


split='train'
ids = os.listdir('./CRANE_dataset-split/images/{}/'.format(split))

for id in ids:
    print(id)
    if 'jpg' in id:
        img = Image.open('./CRANE_dataset-split/images/{}/{}.jpg'.format(split,id[:-4]))
        img.save('./CRANE_dataset-split/images/{}/{}.png'.format(split,id[:-4]))
        subprocess.run(['rm', './CRANE_dataset-split/images/{}/{}.jpg'.format(split,id[:-4])])
        
        


