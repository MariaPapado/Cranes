#!/usr/bin/env python3
import os
import numpy as np

# --- Try Shapely; else use OpenCV raster fallback for IoU/union ---
try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    HAVE_SHAPELY = True
except Exception:
    import cv2
    HAVE_SHAPELY = False

def parse_yolo_obb_line(line):
    p = line.strip().split()
    if not p or len(p) < 9:
        return None
    cls = int(p[0])
    coords = list(map(float, p[1:9]))
    pts = np.array(coords, dtype=np.float32).reshape(4, 2)  # [[x1,y1],...]
    return cls, pts

def load_yolo_obb_txt(path):
    boxes = []
    with open(path, 'r') as f:
        for ln in f:
            rec = parse_yolo_obb_line(ln)
            if rec is None: 
                continue
            cls, pts = rec
            # skip degenerate
            if np.any(~np.isfinite(pts)): 
                continue
            boxes.append((cls, pts))
    return boxes

def polygon_area(pts):
    x, y = pts[:,0], pts[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

def iou_polygon(a, b, raster_res=1024):
    if HAVE_SHAPELY:
        pa, pb = Polygon(a), Polygon(b)
        if (not pa.is_valid) or (not pb.is_valid) or pa.area == 0 or pb.area == 0:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return float(inter / union) if union > 0 else 0.0
    # Fallback: rasterize to estimate IoU
    canv = np.zeros((raster_res, raster_res), np.uint8)
    def to_pix(pts):
        # pts are normalized [0,1]; map to raster grid
        return np.round(pts * (raster_res - 1)).astype(np.int32)
    A = np.zeros_like(canv); B = np.zeros_like(canv)
    cv2.fillPoly(A, [to_pix(a)], 1)
    cv2.fillPoly(B, [to_pix(b)], 1)
    inter = np.sum((A & B) != 0)
    union = np.sum((A | B) != 0)
    return float(inter / union) if union > 0 else 0.0

def min_rot_rect_of_union(polys):
    if HAVE_SHAPELY:
        u = unary_union([Polygon(p) for p in polys if Polygon(p).is_valid])
        if u.is_empty:
            return None
        mrr = u.minimum_rotated_rectangle
        pts = np.array(mrr.exterior.coords)[:-1]  # 4 points
        return order_clockwise(pts)
    # Fallback: average via PCA of all vertices
    all_pts = np.concatenate(polys, axis=0)
    c = all_pts.mean(0)
    X = all_pts - c
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  # Vt rows are eigenvectors
    axes = Vt[:2]
    proj = X @ axes.T
    mins = proj.min(0); maxs = proj.max(0)
    rect = np.array([
        mins[0], mins[1],
        maxs[0], mins[1],
        maxs[0], maxs[1],
        mins[0], maxs[1],
    ], dtype=np.float32).reshape(4,2)
    pts = rect @ axes + c
    return order_clockwise(pts)

def order_clockwise(pts):
    # ensure 4x2, order clockwise starting from the point with smallest (y, then x)
    c = pts.mean(0)
    angles = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    order = np.argsort(-angles)  # clockwise
    pts = pts[order]
    # rotate so first point is top-most (then left-most)
    top_idx = np.lexsort((pts[:,0], pts[:,1]))[0]
    pts = np.roll(pts, -top_idx, axis=0)
    return pts

def cluster_by_iou(polys, thr=0.5):
    n = len(polys)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for i in range(n):
        for j in range(i+1, n):
            if iou_polygon(polys[i], polys[j]) >= thr:
                union(i, j)
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())

def save_yolo_obb_txt(path, records):
    with open(path, 'w') as f:
        for cls, pts in records:
            flat = pts.reshape(-1)
            f.write(f"{cls} " + " ".join(f"{v:.6f}" for v in flat) + "\n")

def merge_overlaps_yolo_obb(boxes, out_txt, iou_thr=0.5):
#    boxes = load_yolo_obb_txt(in_txt)  # [(cls, 4x2)]
    print(boxes[0])
#    if not boxes:
#        open(out_txt, 'w').close(); return
    classes = [c for c,_,_ in boxes]
    polys = [p for _,_,p in boxes]

    # filter degenerate
    valid = []
    for i,p in enumerate(polys):
        if polygon_area(p) > 1e-8 and np.isfinite(p).all():
            valid.append(i)
    boxes = [boxes[i] for i in valid]
    classes = [classes[i] for i in valid]
    polys = [polys[i] for i in valid]

    groups = cluster_by_iou(polys, thr=iou_thr)
    merged = []
    for g in groups:
        gp = [polys[i] for i in g]
        mrr = min_rot_rect_of_union(gp)
        if mrr is None:
            continue
        # keep class id of first in group (or vote if you have multiple classes)
        merged.append((classes[g[0]], mrr.astype(np.float32)))

    save_yolo_obb_txt(out_txt, merged)

#if __name__ == "__main__":
#    # Example usage
#    in_txt  = "/home/maria/YOLO_dota/combrot_preds_TEST/38_28914_20240322.txt"   # your file with YOLO-OBB boxes
#    out_txt = "merged_obb.txt"
#    merge_overlaps_yolo_obb(in_txt, out_txt, iou_thr=0.5)
#    print(f"Saved merged boxes to {out_txt}")
