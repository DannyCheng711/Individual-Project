# Standard library
import os
# Third-party
import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics import auc
import numpy as np
# Local application
from config import VOC_CLASSES

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")

def calculate_iou(ground_box, pred_box):
    """
    Compute IoU between two boxes in [cx, cy, w, h] format.
    All coordinates should be in the same unit (e.g., grid cells).
    """

    cx1, cy1, w1, h1 = ground_box
    cx2, cy2, w2, h2 = pred_box

    # convert center format to corner format 
    x1_min = cx1 - w1 / 2
    y1_min = cy1 - h1 / 2
    x1_max = cx1 + w1 / 2
    y1_max = cy1 + h1 / 2
    
    x2_min = cx2 - w2 / 2
    y2_min = cy2 - h2 / 2
    x2_max = cx2 + w2 / 2
    y2_max = cy2 + h2 / 2

    # Compute intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Compute union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def compute_ap_per_class(pred_entries, all_gt_boxes, class_id, iou_threshold=0.5):
    """
    Compute Average Precision for a specific class
    """

    # Filter predictions and ground truths for specific class
    class_pred_entries = [(img_id, conf, box) for img_id, conf, box, cls in pred_entries if cls == class_id]
    class_gt_boxes = {}

    for img_id, gt_boxes in all_gt_boxes.items():
        filtered_gt_boxes = []
        for box, cls in gt_boxes:
            if cls == class_id:
                filtered_gt_boxes.append(box)
        if len(filtered_gt_boxes) > 0:
            class_gt_boxes[img_id] = filtered_gt_boxes

    if len(class_gt_boxes) == 0 or len(class_pred_entries) == 0:
        return 0.0, [], []
    
    # Sort predictions by confidence
    class_pred_entries = sorted(class_pred_entries, key=lambda x:x[1], reverse = True)

    tp_list = []
    fp_list = []
    used_gt = {k: np.zeros(len(v)) for k, v in class_gt_boxes.items()}
    total_gt = sum(len(v) for v in class_gt_boxes.values())

    for img_id, conf, pred_box in class_pred_entries:
        if img_id not in class_gt_boxes:
            fp_list.append(1)
            tp_list.append(0)
            continue
        
        gt_list = class_gt_boxes[img_id]
        ious = []

        for gt_box in gt_list:
            ious.append(calculate_iou(gt_box, pred_box))
        
        if len(ious) == 0:
            fp_list.append(1)
            tp_list.append(0)
            continue
            
        best_iou = max(ious)
        best_idx = np.argmax(ious)

        if best_iou >= iou_threshold and used_gt[img_id][best_idx] == 0:
            tp_list.append(1)
            fp_list.append(0)
            used_gt[img_id][best_idx] = 1
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    recalls = tp_cum / (total_gt + 1e-6)  # recall = TP / (TP + FN)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6) # precision = TP / (TP + FP)

    recall_vals = np.concatenate([[0.0], recalls, [1.0]])
    precision_vals = np.concatenate([[0.0], precisions, [0.0]])
    
    # Monotonic non-increasing smoothing
    for i in range(len(precision_vals)-2, -1, -1):
        precision_vals[i] = max(precision_vals[i], precision_vals[i+1])

    return auc(recall_vals, precision_vals), recall_vals, precision_vals

def compute_map(pred_entries, all_gt_boxes, num_classes, iou_threshold = 0.5, epoch = None, save_dir=None):
    """
    Compute mean Average Precision (mAP) across all classes
    """
    aps = []
    class_aps = {}

    for class_id in range(num_classes):
        ap, recall_vals, precision_vals = compute_ap_per_class(
            pred_entries, all_gt_boxes, class_id, iou_threshold
        )
        aps.append(ap)
        class_aps[class_id] = ap

        # save the pr graph
        class_name = VOC_CLASSES[class_id]
        title = f"Precision–Recall Curve [{class_name}] (IoU ≥ 0.5)"
        suffix = epoch if epoch else "final"
        plot_pr_curve(recall_vals, precision_vals, ap=ap, title=title,
            save_dir=os.path.join(save_dir, f"pr_curve_class_{class_id}_{suffix}.png"))
    
    map_score = np.mean(aps)
    return map_score, class_aps

def plot_pr_curve(recall_vals, precision_vals, ap=None, save_dir=None, title="Precision–Recall Curve (IoU ≥ 0.5)"):
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, marker='o', linestyle='-', label=f"AP@0.5 = {ap:.4f}" if ap is not None else "PR Curve")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.legend(loc="lower left")

    if save_dir:
        plt.savefig(save_dir)
        print(f"PR curve saved to {save_dir}")
