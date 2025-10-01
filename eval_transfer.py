##evaluation of trained model on certain files

from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np


#ids = os.listdir('./DOTAv1-split/images/test/')
repo = './TEST/images/'
ids = [os.path.join(repo, f) for f in os.listdir(repo) if os.path.isfile(os.path.join(repo, f))]

model = YOLO('./runs/obb/train4/weights/best.pt')

out_dir = "submission"
os.makedirs(out_dir, exist_ok=True)

#results = model.predict(ids, stream=True)


ids = ['img.png']
#ids = ['./TEST/images/38_54188_20240914.png']



for _, id in enumerate(tqdm(ids)):
    results = model.predict(id, stream=True, imgsz=1024)

#######here it saves the model result in a visualized image with boxes##############################################################################
#    for i, r in enumerate(results):
#        # Plot results image
#        im_bgr = r.plot()  # BGR-order numpy array
#        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

#        r.show()

#        # Save results to disk
#        r.save("./submission/{}".format(os.path.basename(id)))
#####################################################################################
    #print((results[0].boxes))

####below it writes the model predictions in txt
    with open('./preds_TEST/{}.txt'.format(os.path.basename(id)[:-4]), "w") as f:
      for r in results:
          #print(r.obb.conf.shape)
          for i,box in enumerate(r.obb.xyxyxyxyn.cpu().numpy()):  # normalized 4-point boxes
            box = box.flatten().tolist()
#              if r.obb.conf[i]>0.2:
#              box.insert(0, 0)

            line = f"{0} " + " ".join(f"{c:.2f}" for c in box)  + "\n"
            f.write(line)


