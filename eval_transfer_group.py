####vriskei gia ola ta rotations ta model predictions, vlepei poia kanoun overlap sumfwna me ena iou threshold kai pairnei to meso oro tous

from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
from combine_rotated_preds import *
from group_polys import *

#ids = os.listdir('./DOTAv1-split/images/test/')
repo = './TEST/images/'
ids = [os.path.join(repo, f) for f in os.listdir(repo) if os.path.isfile(os.path.join(repo, f))]

model = YOLO('./runs/obb/train4/weights/best.pt')

out_dir = "submission"
os.makedirs(out_dir, exist_ok=True)

#results = model.predict(ids, stream=True)


ids = ['img.png']
ids = ['38_28914_20240322.png']
#os.mkdir('outs')

conf_thres = 0.1

for _, id in enumerate(tqdm(ids)):


    results = model.predict(repo + id, stream=True, imgsz=1024)

#######################


    img = Image.open(repo + id) #.convert("RGB")
    W0, H0 = img.size

    all_quads = []   # list of (class_id, conf, 4x2 normalized in original)
    all_confs = []


    for k in (0, 1, 2, 3):  # 0,90,180,270 CCW
        im_k = rot_image(np.array(img), k)
        im_k = Image.fromarray(im_k)
        im_k.save('im_k.png')
        preds = model.predict('im_k.png', stream=True, imgsz=1024, verbose=False)
        for r in preds:
            if r.obb is None or len(r.obb) == 0:
                continue
            conf = r.obb.conf.cpu().numpy()
            cls  = r.obb.cls.cpu().numpy().astype(int)
            quads = r.obb.xyxyxyxyn.cpu().numpy().reshape(-1, 4, 2)

            # filter by conf first
            keep = conf >= conf_thres
            if not np.any(keep):
                continue

            for q, c, cf in zip(quads[keep], cls[keep], conf[keep]):
                q_orig = map_back_quad(q, k, W0, H0)  # (4,2) normalized in original
                all_quads.append((c, cf, q_orig))
                all_confs.append(cf)

    final_items = simple_dedup(all_quads)
    print('lennnnnnn', len(final_items))

    out_txt = "./outs/{}.txt".format(id[:-4])
    merge_overlaps_yolo_obb(final_items, out_txt, iou_thr=0.5)
    print(f"Saved merged boxes to {out_txt}")



