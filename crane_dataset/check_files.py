###KANW DELETE OLES TIS EIKONES POU DEN PERIEXOUN CRANE

import os
import subprocess

ids = os.listdir('./IMAGES/')

for id in ids:
    if os.path.exists('./LABELS/{}.txt'.format(id[:-4])):
        print('./LABELS/{}.txt'.format(id[:-4]))
    else:
        subprocess.run(['rm', './IMAGES/{}.png'.format(id[:-4])])

        
