import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from config import DEVICE, VOC_CLASSES
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np 
from sklearn.metrics import auc
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms
import cv2

# Setting
num_classes = 20  # VOC has 20 classes,
image_size = 160 # mcunet
S = 5 

def calculate_iou(ground_box, pred_box):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """

    x1_min, y1_min, x1_max, y1_max = ground_box
    x2_min, y2_min, x2_max, y2_max = pred_box

    # Compute intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def evaluate_pred_gt_handcraft(all_preds, all_gt_boxes, iou_threshold=0.5):
    """
    Compute Average Precision for a specific class
    """
    pred_entries = []
    for img_id, pred_boxes in enumerate(all_preds):
        if not pred_boxes: 
            continue

        nms_pred_boxes = apply_classwise_nms(pred_boxes=pred_boxes, iou_thresh=0.5)
        
        for pred_box in nms_pred_boxes:
            x1, y1, x2, y2, conf, classid = pred_box
            pred_entries.append((img_id, conf, [x1, y1, x2, y2,]))

    # Sort predictions by confidence
    pred_entries = sorted(pred_entries, key=lambda x:x[1], reverse = True)

    gt_boxes_entries = {}
    for img_id, gt_boxes in enumerate(all_gt_boxes):
        converted_gt = [] # only use coordinates
        for gt_box in gt_boxes:
            if len(gt_box) >= 4:
                x1, y1, x2, y2 = gt_box[:4]
                converted_gt.append([x1, y1, x2, y2])
        gt_boxes_entries[img_id] = converted_gt

    tp_list = []
    fp_list = []
    used_gt = {k: np.zeros(len(v)) for k, v in gt_boxes_entries.items()}
    total_gt = sum(len(v) for v in gt_boxes_entries.values())

    for img_id, conf, pred_box in pred_entries:
        if img_id not in gt_boxes_entries:
            fp_list.append(1)
            tp_list.append(0)
            continue
        
        gt_list = gt_boxes_entries[img_id]
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

    ap = auc(recall_vals, precision_vals)

    return ap, recall_vals, precision_vals

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

def plot_pr_curves_comp(recall_vals_list, precision_vals_list, ap_list,
                        save_path=None, title="Precision-Recall Curves Comparison"):

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']

    for i, (recall_vals, precision_vals, ap) in enumerate(zip(recall_vals_list, precision_vals_list, ap_list)):
        color = colors[i % len(colors)]
        if i == 0:
            plt.plot(recall_vals, precision_vals, marker='o', linestyle='-', color=color, 
                    label=f"AP@0.5 = {ap:.4f}", linewidth=2, markersize=4)
        if i == 1:
            plt.plot(recall_vals, precision_vals, marker='o', linestyle='-', color=color, 
                    label=f"WBF-AP@0.5 = {ap:.4f}", linewidth=2, markersize=4)

    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Comparison PR curve saved to {save_path}")

# surpass the bbox whihc iou_thresh > 0.5
def apply_classwise_nms(pred_boxes, iou_thresh=0.5):
    boxes = torch.tensor([[p[0], p[1], p[2], p[3]] for p in pred_boxes], dtype=torch.float32)
    scores = torch.tensor([p[4] for p in pred_boxes], dtype=torch.float32)
    labels = torch.tensor([p[5] for p in pred_boxes], dtype=torch.long)

    keep = []
    for cls in labels.unique():
        # select the class index for this class
        class_mask = (labels == cls)
        # Shape: (N, 1) -> as_tuple -> tuple of one tensors (with one row) 
        class_indices = torch.nonzero(class_mask, as_tuple=True)[0]

        # Run NMS for this class subset
        keep_idxs = nms(boxes[class_indices], scores[class_indices], iou_thresh)
        keep.append(class_indices[keep_idxs])

    # [tensor([1, 2]), tensor([5, 6, 7]), tensor([9])]
    # -> tensor([1, 2, 5, 6, 7, 9])
    keep = torch.cat(keep)
    return [pred_boxes[i] for i in keep]

def evaluate_pred_gt(predictions, ground_truths, device=None):
    metric = MeanAveragePrecision(box_format = "xyxy", iou_type="bbox", class_metrics=True)

    preds_list = []
    targets_list = []
    
    for pred_boxes, gt_boxes in zip(predictions, ground_truths):
        nms_pred_boxes = apply_classwise_nms(pred_boxes=pred_boxes, iou_thresh=0.5)

        # Format predictions
        if len(nms_pred_boxes) > 0: 
            preds_list.append({
                "boxes": torch.tensor([[p[0], p[1], p[2], p[3]] for p in nms_pred_boxes], device = device),
                "scores": torch.tensor([p[4] for p in nms_pred_boxes], device=device),
                "labels": torch.tensor([p[5] for p in nms_pred_boxes], device=device, dtype=torch.long)
            })
        else:
            preds_list.append({
                "boxes": torch.empty((0, 4), device=device, dtype=torch.float32),
                "scores": torch.empty((0,), device=device, dtype=torch.float32),
                "labels": torch.empty((0,), device=device, dtype=torch.long) 
            })

        # Format ground truth 
        gt_box_list = []
        gt_label_list = []

        for gt_box in gt_boxes:
            gt_box_list.append([gt_box[0], gt_box[1], gt_box[2], gt_box[3]])
            gt_label_list.append(gt_box[4])

        targets_list.append({
            "boxes": torch.tensor(gt_box_list, device=device) if gt_box_list else torch.empty((0,4), device=device, dtype=torch.float32),
            "labels": torch.tensor(gt_label_list, device=device, dtype=torch.long) if gt_label_list else torch.empty((0, ), device=device, dtype=torch.long)
        })
    
    metric.update(preds_list, targets_list)
    results = metric.compute()

    return results


def draw_bboxes_pil(pil_img, bboxes, color=(0, 255, 0), thickness=2, add_scores=False):
    """Draw bounding boxes on a PIL image."""
    try:
        if pil_img.mode!= 'RGB':
            pil_img = pil_img.convert('RGB')
        
        img_array = np.array(pil_img)
        if img_array.size == 0:
            print("Error: Empty image array")
            return pil_img
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    except Exception as e:
        print(f"Error converting image: {e}")
        return pil_img
    
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if add_scores and len(box) > 4:
            score = box[4]
            text = f"{score:.2f}"
            # Place text inside the box with padding 
            text_x, text_y = x1 + 5, y1 + 15  
            cv2.putText(img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA) 
            
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def visualize_bbox_grid_tensors(images_view1, images_view2,
                                 preds_view1, wbf_view1,
                                 preds_view2, wbf_view2,
                                 save_path="./image_result/bbox_grid.png", max_cols=10):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert tensors to PIL
    images_v1 = []
    images_v2 = []

    for img in images_view1:
        img = img.cpu()
        images_v1.append(transforms.ToPILImage()(img))

    for img in images_view2:
        img = img.cpu()
        images_v2.append(transforms.ToPILImage()(img))

    rows = 4
    total_images = len(images_v1)
    for chunk_idx in range(0, total_images, max_cols):
        start_col = chunk_idx
        end_col = min(chunk_idx + max_cols, total_images)
        cols = end_col - start_col

        rows = 4
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        row_titles = ["View1 Base", "View1 WBF", "View2 Base", "View2 WBF"]

        for col in range(cols):
            actual_col = start_col + col
            for row in range(rows):
                ax = axes[row, col]
                if row in (0, 1):  # view1
                    img = images_v1[actual_col]
                    bboxes = preds_view1[actual_col] if row == 0 else wbf_view1[actual_col]
                    nms_bboxes = apply_classwise_nms(pred_boxes=bboxes, iou_thresh=0.5)
                    max_bbox= max(nms_bboxes, key=lambda x: x[4])
                else:  # view2
                    img = images_v2[actual_col]
                    bboxes = preds_view2[actual_col] if row == 2 else wbf_view2[actual_col]
                    nms_bboxes = apply_classwise_nms(pred_boxes=bboxes, iou_thresh=0.5)
                    max_bbox= max(nms_bboxes, key=lambda x: x[4])

                img_with_boxes = draw_bboxes_pil(img, [max_bbox], add_scores=True)
                ax.imshow(img_with_boxes)
                ax.axis('off')
                if row == 0:
                    ax.set_title(f"Object {actual_col+1}", fontsize=10)
        
        for row in range(rows):
            axes[row, 0].text(-0.1, 0.5, row_titles[row], transform=axes[row,0].transAxes,
            fontsize=12, rotation=90, verticalalignment='center', horizontalalignment='right')
        
        plt.tight_layout()
        base_path = save_path.replace('.png', '')

        if chunk_idx == 0:
            chunk_save_path = save_path
        else:
            chunk_num = (chunk_idx // max_cols) + 1
            chunk_save_path = f"{base_path}_part{chunk_num}.png"

        plt.savefig(chunk_save_path, dpi=300)
        plt.close()
        print(f"Saved visualization grid to {chunk_save_path}")


def visualize_bbox_grid_tensors_single_view(
        images_view1, preds_view1, save_path="./image_result/bbox_grid_single.png", max_cols=10):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert tensors to PIL
    images_v1 = []

    for img in images_view1:
        img = img.cpu()
        images_v1.append(transforms.ToPILImage()(img))

    total_images = len(images_v1)
    for chunk_idx in range(0, total_images, max_cols):
        start_col = chunk_idx
        end_col = min(chunk_idx + max_cols, total_images)
        cols = end_col - start_col

        rows = 1
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

        if cols == 1:
            axes = [axes] # Make it a list for consistent indexing

        for col in range(cols):
            actual_col = start_col + col
            
            ax = axes[col]
            img = images_v1[actual_col]
            bboxes = preds_view1[actual_col] 

            if bboxes:
                nms_bboxes = apply_classwise_nms(pred_boxes=bboxes, iou_thresh=0.5)
                if nms_bboxes:
                    max_bbox= max(nms_bboxes, key=lambda x: x[4])
                    img_with_boxes = draw_bboxes_pil(img, [max_bbox], add_scores=True)
                else:
                    img_with_boxes = img
            else:
                img_with_boxes = img

            ax.imshow(img_with_boxes)
            ax.axis('off')
            ax.set_title(f"Image {actual_col+1}", fontsize=12)
        
        fig.suptitle("Single View Detection Resutls", fontsize=16, y=0.95)
       
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        base_path = save_path.replace('.png', '')
        if chunk_idx == 0:
            chunk_save_path = save_path
        else:
            chunk_num = (chunk_idx // max_cols) + 1
            chunk_save_path = f"{base_path}_part{chunk_num}.png"

        plt.savefig(chunk_save_path, dpi=300)
        plt.close()
        print(f"Saved single view visualization to {chunk_save_path}")


def decode_pred(pred, anchors, num_classes, image_size, conf_thresh = 0.0):
    """
    Decode YOLO predictions into bounding boxes.
    Args:
        pred: Tensor of shape [B, A*(5+C), S, S] 
        anchors: tensor of shape [A, 2] in grid units 
        num_classes: number of classes
        image_size: pixel width/height of input image
        conf_thresh: objectness threshold

    Returns:
        List[box]: a list of [xmin, ymin, xmax, ymax, confidence]
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

