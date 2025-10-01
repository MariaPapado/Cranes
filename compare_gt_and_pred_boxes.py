import os
import glob
import numpy as np
from shapely.geometry import Polygon

def read_labels(file_path):
    """Return list[(cls, Polygon)] from a YOLO-OBB txt. Empty if file missing."""
    if not os.path.exists(file_path):
        return []
    boxes = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:  # cls + 8 coords
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            poly = Polygon(np.array(coords, dtype=float).reshape(-1, 2))
            if poly.is_valid and poly.area > 0:
                boxes.append((cls, poly))
    return boxes

def polygon_iou(a: Polygon, b: Polygon) -> float:
    if not (a.is_valid and b.is_valid):
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0

def eval_per_image(pred_dir, gt_dir, iou_threshold=0.5):
    """
    Returns:
      per_image: dict[filename] = {'TP': int, 'FP': int, 'FN': int, 'num_gt': int, 'num_pred': int}
      totals: {'TP':..., 'FP':..., 'FN':..., 'num_gt':..., 'num_pred':...}
    """
    per_image = {}
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.txt")))
    # also include images that only have GTs
    gt_only = set(os.listdir(gt_dir)) if os.path.isdir(gt_dir) else set()
    for pf in pred_files:
        gt_only.discard(os.path.basename(pf))
    all_files = sorted(set([os.path.basename(p) for p in pred_files]) | gt_only)

    totals = {'TP': 0, 'FP': 0, 'FN': 0, 'num_gt': 0, 'num_pred': 0}

    for filename in all_files:
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)

        preds = read_labels(pred_path)
        gts = read_labels(gt_path)

        matched_gt = set()
        TP = 0
        for cls_pred, poly_pred in preds:
            # find best GT match (greedy one-to-one)
            best_j, best_iou = -1, 0.0
            for j, (cls_gt, poly_gt) in enumerate(gts):
                if j in matched_gt or cls_gt != cls_pred:
                    continue
                iou = polygon_iou(poly_pred, poly_gt)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                TP += 1
                matched_gt.add(best_j)

        FP = len(preds) - TP
        FN = len(gts) - TP

        per_image[filename] = {
            'TP': TP, 'FP': FP, 'FN': FN,
            'num_gt': len(gts), 'num_pred': len(preds)
        }

        totals['TP'] += TP
        totals['FP'] += FP
        totals['FN'] += FN
        totals['num_gt'] += len(gts)
        totals['num_pred'] += len(preds)

    return per_image, totals


# Example usage
if __name__ == "__main__":
    pred_dir = "./preds_TEST/"  # folder with prediction txts
    gt_dir = "./TEST/labels/"   # folder with ground truth txts
    per_image, totals = eval_per_image(pred_dir, gt_dir, iou_threshold=0.2)

    # Print per-image summary
    for k, v in per_image.items():
        print(f"{k}: TP={v['TP']} FP={v['FP']} FN={v['FN']} (gt={v['num_gt']}, pred={v['num_pred']})")
    print("--- totals ---")
    print(f"TP={totals['TP']} FP={totals['FP']} FN={totals['FN']} (gt={totals['num_gt']}, pred={totals['num_pred']})")
