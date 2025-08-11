import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import DEVICE, DATASET_ROOT, VOC_ROOT
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np 
from sklearn.metrics import auc

import preprocess
import model

# Setting
num_classes = 1 # COCO 80, but we only detect human 
image_size = 160 # mcunet
S = 5 

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



def decode_pred(pred, anchors, num_classes, image_size = 160, conf_thresh = 0.5):
    """
    Decode YOLO predictions into bounding boxes.
    Args:
        pred: Tensor of shape [B, S, S, A*(5+C)]
        anchors: list of [w, h] in grid units (e.g. [1.0, 2.0])
        num_classes: number of classes
        image_size: pixel width/height of input image
        conf_thresh: objectness threshold

    Returns:
        List[List[box]]: for each batch element, a list of [xmin, ymin, xmax, ymax, confidence]
    """

    batch_size, S, _, pred_dim = pred.shape
    A = len(anchors)

    # Reshape and permute to [B, A, S, S, 5+C]
    pred = pred.reshape(batch_size, S, S, A, 5 + num_classes)
    pred = pred.permute(0, 3, 1, 2, 4).contiguous()  # → [B, A, S, S, 5 + C] and make it continuous in memory location
    
    # Step 2: extract tx, ty, tw, th, obj_score, class_probs 
    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    obj_score = torch.sigmoid(pred[..., 4])

    all_decoded = []

    for b in range(batch_size):
        decoded_boxes = [] 
        for a in range(A):
            for i in range(S):
                for j in range(S):
                    conf = obj_score[b, a, i, j]

                    if conf < conf_thresh: continue
                    
                    # grid unit -> image unit 
                    cx = (j + torch.sigmoid(tx[b, a, i, j])) / S
                    cy = (i + torch.sigmoid(ty[b, a, i, j])) / S
                    bw = anchors[a][0] * torch.exp(tw[b, a, i, j]) / S
                    bh = anchors[a][1] * torch.exp(th[b, a, i, j]) / S

                    # convert to box 
                    xmin = (cx - bw/2) * image_size
                    ymin = (cy - bh/2) * image_size
                    xmax = (cx + bw/2) * image_size
                    ymax = (cy + bh/2) * image_size
                    decoded_boxes.append(
                        [xmin.item(), ymin.item(), xmax.item(), ymax.item(), conf.item()])

        all_decoded.append(decoded_boxes)

    return all_decoded
    

def compute_ap_05(pred_entries, all_gt_boxes, iou_threshold = 0.5):
    
    pred_entries = sorted(pred_entries, key=lambda x:x[1], reverse = True)
    tp_list = []
    fp_list = []

    used_gt = {k: np.zeros(len(v)) for k, v in all_gt_boxes.items()}
    total_gt = sum(len(v) for v in all_gt_boxes.values())

    for img_id, conf, pred_box in pred_entries:
        if img_id not in all_gt_boxes:
            fp_list.append(1)
            tp_list.append(0)
            continue
        
        gt_list = all_gt_boxes[img_id]
        ious = []
        for gt_box in gt_list:
            # print(gt_box)
            # print(pred_box)
            # print(calculate_iou(gt_box, pred_box))
            # print("---------------------")
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

    # print(tp_list)
    # print(fp_list)

    recalls = tp_cum / (total_gt + 1e-6) # recall = TP / (TP + FN)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6) # precision = TP / (TP + FP)

    recall_vals = np.concatenate([[0.0], recalls, [1.0]])
    precision_vals = np.concatenate([[0.0], precisions, [0.0]])
    # It is a monotonic non-increasing smoothing of the precision curve, done in reverse.
    for i in range(len(precision_vals)-2, -1, -1):
        precision_vals[i] = max(precision_vals[i], precision_vals[i+1])

    return auc(recall_vals, precision_vals), recall_vals, precision_vals

def plot_pr_curve(recall_vals, precision_vals, ap=None, save_path=None, title="Precision–Recall Curve (IoU ≥ 0.5)"):
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, marker='o', linestyle='-', label=f"AP@0.5 = {ap:.4f}" if ap is not None else "PR Curve")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.legend(loc="lower left")

    if save_path:
        plt.savefig(save_path)
        print(f"PR curve saved to {save_path}")



def eval_metrics(val_loader, model, anchors, epoch = None, iou_threshold = 0.5, image_size = 160):

    model.eval()

    all_pred_entries = [] # list of (image_id, score, [x1, y1, x2, y2])
    all_gt_boxes_by_img = {} # dict: image_id -> list of [x1, y1, x2, y2]

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(tqdm(val_loader, desc="Validating", ncols=80)):

            imgs = imgs.to(DEVICE)
            preds = model(imgs) # [B, A, S, S, 5+C]
            decoded_preds = decode_pred(preds, anchors = anchors, num_classes = 1)

            for b in range(len(imgs)):

                img_id = f"{i}_{b}"
                # record all ground truth
                gt_boxes_img = []
                gt_boxes = targets[b].cpu() # [xc, yc, w, h, class_id]
                for gt in gt_boxes:
                    gt_boxes_img.append(gt[:4].tolist())

                all_gt_boxes_by_img[img_id] = gt_boxes_img

                # record all prediction boxes
                pred_boxes = decoded_preds[b] # [[xmin, ymin, xmax, ymax, conf]]
                for pred in pred_boxes:
                    xmin, ymin, xmax, ymax, conf = pred
                    cx_pred = (xmin + xmax) / 2 / image_size
                    cy_pred = (ymin + ymax) / 2 / image_size
                    w_pred = (xmax - xmin) / image_size
                    h_pred = (ymax - ymin) / image_size
                                        
                    all_pred_entries.append(
                        (img_id, conf, [cx_pred, cy_pred, w_pred, h_pred]))

                    
        
    # compute ap@0.5
    ap50, recall_vals, precision_vals = compute_ap_05(all_pred_entries, all_gt_boxes_by_img)

    print(f"\n[Validation Result]")
    print(f"AP@0.5: {ap50}")
    # print("\n[PR Curve Points]")
    # for i, (r, p) in enumerate(zip(recall_vals, precision_vals)):
    #    print(f"Point {i:02d}: Recall = {r:.4f}, Precision = {p:.4f}")

    if epoch is None:
        plot_pr_curve(recall_vals, precision_vals, ap=ap50, save_path=f"./images_voc/pr_curve_ap50_150_aug_final.png")
    else:
        plot_pr_curve(recall_vals, precision_vals, ap=ap50, save_path=f"./images_voc/pr_curve_ap50_150_aug_{epoch}.png")
   

