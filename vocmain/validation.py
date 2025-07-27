import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import DEVICE, VOC_ROOT, VOC_CLASSES, VOC_CLASS_TO_IDX
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np 
from sklearn.metrics import auc

import preprocess
import model

# Setting
num_classes = 20  # VOC has 20 classes,
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



def decode_pred(pred, anchors, num_classes, image_size, conf_thresh = 0.5):
    """
    Decode YOLO predictions into bounding boxes.
    Args:
        pred: Tensor of shape [B, A*(5+C), S, S] 
        anchors: tensor of shape [A, 2] in grid units 
        num_classes: number of classes
        image_size: pixel width/height of input image
        conf_thresh: objectness threshold

    Returns:
        List[List[box]]: for each batch element, a list of [xmin, ymin, xmax, ymax, confidence]
    """

    batch_size, pred_dim, S, _ = pred.shape  # [B, A*(5+C), S, S]
    A = len(anchors)
    C = num_classes

    # Reshape to [B, A, 5+C, S, S] then permute to [B, S, S, A, 5+C]
    pred = pred.reshape(batch_size, A, 5 + C, S, S)
    pred = pred.permute(0, 3, 4, 1, 2)  # [B, S, S, A, 5+C]
    
    # Extract components
    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    obj_score = torch.sigmoid(pred[..., 4])
    class_scores = torch.softmax(pred[..., 5:], dim=-1)  # Class probabilities

    all_decoded = []

    for b in range(batch_size):
        decoded_boxes = [] 
        for i in range(S):
            for j in range(S):
                for a in range(A): 

                    conf = obj_score[b, i, j, a]

                    if conf < conf_thresh: 
                        continue
                    
                    # grid unit -> image unit 
                    cx_grid = j + torch.sigmoid(tx[b, i, j, a])
                    cy_grid = i + torch.sigmoid(ty[b, i, j, a])
                    bw_grid = anchors[a, 0] * torch.exp(tw[b, i, j, a]) # anchor is tensor
                    bh_grid = anchors[a, 1] * torch.exp(th[b, i, j, a])
                    
                    cx = cx_grid / S
                    cy = cy_grid / S
                    bw = bw_grid / S 
                    bh = bh_grid / S 

                    # convert to box 
                    xmin = (cx - bw/2) * image_size
                    ymin = (cy - bh/2) * image_size
                    xmax = (cx + bw/2) * image_size
                    ymax = (cy + bh/2) * image_size

                    # Clamp to image bounds
                    xmin = max(0, min(xmin.item(), image_size))
                    ymin = max(0, min(ymin.item(), image_size))
                    xmax = max(0, min(xmax.item(), image_size))
                    ymax = max(0, min(ymax.item(), image_size))

                    class_probs = class_scores[b, i, j, a]
                    best_class_conf, best_class_id = torch.max(class_probs, dim=0)
                    final_conf = conf * best_class_conf  # Combined confidence
                            
                    decoded_boxes.append(
                        [xmin, ymin, xmax, ymax, final_conf.item(), best_class_id.item()])

        all_decoded.append(decoded_boxes)

    return all_decoded

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
    used_gt = {k: np.zeros(len(v)) for k, v in all_gt_boxes.items()}
    total_gt = sum(len(v) for v in all_gt_boxes.values())

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

def compute_map(pred_entries, all_gt_boxes, num_classes, iou_threshold = 0.5):
    
    """
    Compute mean Average Precision (mAP) across all classes
    """
    aps = []
    class_aps = {}

    for class_id in range(num_classes):
        ap, _, _ = compute_ap_per_class(pred_entries, all_gt_boxes, class_id, iou_threshold)
        aps.append(ap)
        class_aps[class_id] = ap

    map_score = np.mean(aps)
    return map_score, class_aps

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

def visualize_predictions(val_loader, model, anchors, num_samples=2, save_path="./images_voc/"):
    """
    Visualize model predictions vs ground truth on sample images
    """

    model.eval()
    
    # Convert anchors to tensor if needed
    if isinstance(anchors, list):    
        anchors = torch.tensor(anchors, dtype=torch.float32).to(DEVICE)
    
    
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(val_loader):
            # only check batch 0
            if batch_idx >= 10:
                break
            
            fig, axes = plt.subplots(2, 8, figsize=(24, 12))
            axes = axes.flatten()
            
            imgs = imgs.to(DEVICE)
            preds = model(imgs)
            decoded_preds = decode_pred(preds, anchors=anchors, num_classes=num_classes, image_size=image_size)
            
            batch_size = imgs.shape[0]

            for img_idx in range(len(imgs)):
                
                ax = axes[img_idx]
                
                # Convert image tensor to PIL for visualization
                img_pil = transforms.functional.to_pil_image(imgs[img_idx].cpu())
                ax.imshow(img_pil)
                
                # Draw ground truth boxes (green)
                target_tensor = targets[img_idx]
                
                for i_grid in range(target_tensor.shape[0]):
                    for j_grid in range(target_tensor.shape[1]):
                        for a_idx in range(target_tensor.shape[2]):
                            if target_tensor[i_grid, j_grid, a_idx, 4] == 1:
                                tx = target_tensor[i_grid, j_grid, a_idx, 0].item()
                                ty = target_tensor[i_grid, j_grid, a_idx, 1].item()
                                tw = target_tensor[i_grid, j_grid, a_idx, 2].item()
                                th = target_tensor[i_grid, j_grid, a_idx, 3].item()
                                
                                class_vector = target_tensor[i_grid, j_grid, a_idx, 5:]
                                class_id = torch.argmax(class_vector).item()                               
                                
                                # Convert to pixel coordinates
                                cx = (j_grid + tx) / target_tensor.shape[1] * image_size
                                cy = (i_grid + ty) / target_tensor.shape[0] * image_size
                                w = tw / target_tensor.shape[1] * image_size
                                h = th / target_tensor.shape[0] * image_size
                                
                                x1 = cx - w/2
                                y1 = cy - h/2
                                
                                # Draw GT box in green
                                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                                       edgecolor='green', facecolor='none')
                                ax.add_patch(rect)
                                
                                class_name = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else f"C{class_id}"
                                
                                
                                ax.text(x1, y1-5, f'GT: {class_name}', color='green', fontweight='bold')
                
                # Draw prediction boxes (red)
                
                pred_boxes = decoded_preds[img_idx]
                for pred in pred_boxes[:5]:  # Limit to top 5 predictions
                    xmin, ymin, xmax, ymax, conf, class_id = pred
                    w = xmax - xmin
                    h = ymax - ymin
                    
                    # Draw pred box in red
                    rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, 
                                           edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    
                    class_name = VOC_CLASSES[int(class_id)] if int(class_id) < len(VOC_CLASSES) else f"C{int(class_id)}"
                    ax.text(xmin, ymin-5, f'Pred: {class_name} ({conf:.2f})', color='red', fontweight='bold')
                
                ax.set_title(f'Image {img_idx}')
                ax.axis('off')
                
    
            plt.tight_layout()
            batch_save_path = os.path.join(save_path, f"batch_{batch_idx:02d}_predictions.png")
            plt.savefig(batch_save_path, dpi=300, bbox_inches='tight')
            plt.close()

    print(f"Prediction visualization saved to {batch_save_path}")



def eval_metrics(val_loader, model, anchors, epoch = None, iou_threshold = 0.5, target_classes=None):

    """
    Evaluate model performance using mAP or class-specific AP
    Args:
        target_classes: list of class IDs to evaluate (None for all classes)
        class_names: list of class names for display
    """
    
    model.eval()
    
    all_pred_entries = [] # list of (image_id, score, [x1, y1, x2, y2])
    all_gt_boxes_by_img = {} # dict: image_id -> list of [x1, y1, x2, y2]

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(tqdm(val_loader, desc="Validating", ncols=80)):

            imgs = imgs.to(DEVICE)
            preds = model(imgs) # [B, A*(5+C), S, S]
            
            decoded_preds = decode_pred(preds, anchors=anchors, num_classes=num_classes, image_size=image_size)

            for b in range(len(imgs)):
                img_id = f"{i}_{b}"

                # Extract ground truth boxes from targets [S, S, A, 5+C]
                gt_boxes_img = []
                target_tensor = targets[b] # [S, S, A, 5+C]
                
                for i_grid in range(target_tensor.shape[0]):
                    for j_grid in range(target_tensor.shape[1]):
                        for a_idx in range(target_tensor.shape[2]):
                            # object detected
                            if target_tensor[i_grid, j_grid, a_idx, 4] == 1:
                                # Extract normalized coordinates
                                tx = target_tensor[i_grid, j_grid, a_idx, 0].item()
                                ty = target_tensor[i_grid, j_grid, a_idx, 1].item()
                                tw = target_tensor[i_grid, j_grid, a_idx, 2].item()
                                th = target_tensor[i_grid, j_grid, a_idx, 3].item()
                                
                                # Get class from one-hot encoding
                                class_vector = target_tensor[i_grid, j_grid, a_idx, 5:]
                                class_id = torch.argmax(class_vector).item()
                                
                                # Convert to normalized image coordinates
                                cx = (j_grid + tx) / target_tensor.shape[1]
                                cy = (i_grid + ty) / target_tensor.shape[0]
                                w = tw / target_tensor.shape[1]
                                h = th / target_tensor.shape[0]

                                gt_boxes_img.append(([cx, cy, w, h], class_id))

                all_gt_boxes_by_img[img_id] = gt_boxes_img

                # record all prediction boxes
                pred_boxes = decoded_preds[b] # [[xmin, ymin, xmax, ymax, conf]]
                for pred in pred_boxes:
                    xmin, ymin, xmax, ymax, conf, class_id = pred
                    cx_pred = (xmin + xmax) / 2 / image_size
                    cy_pred = (ymin + ymax) / 2 / image_size
                    w_pred = (xmax - xmin) / image_size
                    h_pred = (ymax - ymin) / image_size
                                        
                    all_pred_entries.append(
                        (img_id, conf, [cx_pred, cy_pred, w_pred, h_pred], int(class_id)))
    
    # Compute all classes mAP
    map_score, class_aps =compute_map(
        all_pred_entries, all_gt_boxes_by_img, num_classes, iou_threshold)

    print(f"[Validation Result]")
    print(f"mAP@{iou_threshold}: {map_score:.4f}")
    # print("\nPer-class AP:")
    # for class_id, ap in class_aps.items():
    #    print(f"  Class {class_id}: {ap:.4f}")
    
    if target_classes:
        results = {}
        for class_id in target_classes:
            ap, recall_vals, precision_vals = compute_ap_per_class(
                all_pred_entries, all_gt_boxes_by_img, class_id, iou_threshold)
            
            results[class_id] = ap
            
            print(f"\n[Validation Result for Class {class_id}]")
            print(f"AP@{iou_threshold}: {ap:.4f}")
            
            # Plot PR curve
            if epoch is None:
                plot_pr_curve(recall_vals, precision_vals, ap=ap, 
                             save_path=f"./images_voc/pr_curve_class{class_id}_final.png")
            else:
                plot_pr_curve(recall_vals, precision_vals, ap=ap, 
                             save_path=f"./images_voc/pr_curve_class{class_id}_{epoch}.png")


