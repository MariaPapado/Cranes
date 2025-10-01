#############CHATGP      !!!!!!!!!!!!!!!!!!!!!!! @@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

def rot_image(img, k):
    # k in {0,1,2,3} for 0/90/180/270 CCW
    return np.rot90(img, k=k).copy()

def map_back_quad(quad_n, k, W0, H0):
    """
    quad_n: (4,2) normalized points from the *rotated* image
    k: rotation applied to image before predict (0/1/2/3 CCW)
    W0,H0: original image size (before rotation)
    returns normalized points in the original image frame (shape (4,2))
    """
    # rotated image size after rot90
    if k % 2 == 0:
        Wr, Hr = W0, H0
    else:
        Wr, Hr = H0, W0

    # to pixels in rotated frame
    P = quad_n.copy()
    P[:, 0] *= Wr
    P[:, 1] *= Hr

    # inverse mapping from rotated->original (integer grid formulas)
    if k == 0:  # identity
        X = P[:, 0]
        Y = P[:, 1]
    elif k == 1:  # 90° CCW applied -> inverse is 90° CW
        # original x = W0 - 1 - y_r ; original y = x_r
        X = W0 - 1 - P[:, 1]
        Y = P[:, 0]
    elif k == 2:  # 180°
        X = W0 - 1 - P[:, 0]
        Y = H0 - 1 - P[:, 1]
    else:  # k == 3, 270° CCW applied -> inverse is 270° CW
        # original x = y_r ; original y = H0 - 1 - x_r
        X = P[:, 1]
        Y = H0 - 1 - P[:, 0]

    # back to normalized in original frame
    X = np.clip(X / W0, 0, 1)
    Y = np.clip(Y / H0, 0, 1)
    return np.stack([X, Y], axis=1)

id = './TEST/images/38_28914_20240322.png'



conf_thres = 0.1
img = Image.open(id) #.convert("RGB")
W0, H0 = img.size
print(np.array(img).shape)

all_quads = []   # list of (class_id, conf, 4x2 normalized in original)
all_confs = []

model = YOLO('./runs/obb/train4/weights/best.pt')

for k in (0, 1, 2, 3):  # 0,90,180,270 CCW
    im_k = rot_image(np.array(img), k)
    im_k = Image.fromarray(im_k)
    im_k.save('im_k.png')
    preds = model.predict('im_k.png', stream=True, imgsz=1024)
    for r in preds:
        #if r.obb is None or len(r.obb) == 0:
        #    continue
        conf = r.obb.conf.cpu().numpy()
        cls  = r.obb.cls.cpu().numpy().astype(int)
        quads = r.obb.xyxyxyxyn.cpu().numpy().reshape(-1, 4, 2)
        print('k', k)
        print(quads)

        # filter by conf first
        keep = conf >= conf_thres
        if not np.any(keep):
            continue
        print(keep)

        for q, c, cf in zip(quads[keep], cls[keep], conf[keep]):
            q_orig = map_back_quad(q, k, W0, H0)  # (4,2) normalized in original
            all_quads.append((c, cf, q_orig))
            all_confs.append(cf)

# (Optional) very simple de-dup: keep highest-conf only if same class and
# quad points are almost identical. Replace with proper polygon NMS if needed.
def simple_dedup(items, tol=2e-3):
    out = []
    for c, cf, q in sorted(items, key=lambda x: -x[1]):
        dup = False
        for c2, cf2, q2 in out:
            if c == c2 and np.allclose(q, q2, atol=tol):
                dup = True
                break
        if not dup:
            out.append((c, cf, q))
    return out

final_items = simple_dedup(all_quads)

# write results (class + 8 coords)
os.makedirs('./combrot_preds_TEST', exist_ok=True)
with open(f'./combrot_preds_TEST/{os.path.basename(id)[:-4]}.txt', 'w') as f:
    for c, cf, q in final_items:
        flat = q.reshape(-1)  # x1,y1,x2,y2,x3,y3,x4,y4 (normalized)
        line = f"{c} " + " ".join(f"{v:.4f}" for v in flat) + "\n"
        f.write(line)
