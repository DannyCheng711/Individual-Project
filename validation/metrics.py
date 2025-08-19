# Standard library
import os
# Third-party
import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics import auc
import numpy as np
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import batched_nms
# Local application
from config import VOC_CLASSES, VOC_ANCHORS
from .bboxprep import decode_pred, target_tensor_to_gt, xyxy_to_cxcywh_norm, calculate_iou_cxcy, calculate_iou_xyxy_tensor


load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")


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
            ious.append(calculate_iou_cxcy(gt_box, pred_box))
        
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

def compute_map(pred_entries, all_gt_boxes, num_classes, iou_threshold = 0.5, epoch = None, save_dir=None, last_epoch=None):
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
        if epoch is not None and last_epoch is not None and epoch == last_epoch:
            class_name = VOC_CLASSES[class_id]
            title = f"Precision–Recall Curve [{class_name}] (IoU ≥ 0.5)"
            plot_pr_curve(recall_vals, precision_vals, ap=ap, title=title,
                save_dir=os.path.join(save_dir, f"pr_curve_class_{class_id}.png"))
        
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

def eval_metrics(val_loader, model, image_size, epoch = None, ttl_epoch = None, iou_threshold = 0.5, save_dir=None, class_id=None):

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
            for b, (imgs, targets) in enumerate(tqdm(val_loader, desc="Validating", ncols=80)):

                imgs = imgs.to(DEVICE)
                preds = model(imgs) # [B, A*(5+C), S, S]
                
                decoded_preds = decode_pred(
                    preds, anchors=torch.tensor(VOC_ANCHORS, dtype=torch.float32), num_classes=len(VOC_CLASSES), image_size=image_size, conf_thresh=0.01)

                for img_idx in range(len(imgs)):
                    img_id = f"{b}_{img_idx}"

                    # Extract ground truth boxes from targets [S, S, A, 5+C]
                    target_tensor = targets[img_idx] # [S, S, A, 5+C]

                    # GT (use helper)
                    gt_xyxy, gt_labels = target_tensor_to_gt(target_tensor, image_size)
                    if gt_xyxy.numel() == 0:
                        all_gt_boxes_by_img[img_id] = []
                    else:
                        gt_cxcywh = xyxy_to_cxcywh_norm(gt_xyxy, image_size)  # (N,4) normalized
                        # cx, cy, w, h, classid
                        gt_list = [(gt_cxcywh[k].tolist(), int(gt_labels[k].item())) for k in range(len(gt_labels))]
                        all_gt_boxes_by_img[img_id] = gt_list

                    # Predictions (keep your NMS, then convert to normalized cxcywh)
                    pred_boxes = decoded_preds[img_idx]  # [[x1,y1,x2,y2,conf,cls], ...] in pixels
                    if pred_boxes.shape[0] > 0:
                        boxes = pred_boxes[:, :4]
                        scores = pred_boxes[:, 4]
                        labels = pred_boxes[:, 5].long()

                        keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
                        boxes = boxes[keep]
                        scores = scores[keep]
                        labels = labels[keep]
                        pred_cxcywh = xyxy_to_cxcywh_norm(boxes, image_size)  # normalized (N,4)

                        for idx in range(boxes.shape[0]):
                            all_pred_entries.append(
                                (img_id, float(scores[idx].item()), pred_cxcywh[idx].tolist(), int(labels[idx].item()))
                        )
                    
        # Compute all classes mAP
        if class_id is not None:
            ap, recall_vals, precision_vals = compute_ap_per_class(
                all_pred_entries, all_gt_boxes_by_img, class_id, iou_threshold
            )
            results = {VOC_CLASSES[class_id]: ap}
        
            print(f"AP@{iou_threshold} for {VOC_CLASSES[class_id]}: {ap:.4f}")

            return {"full": results, "ap": float(ap)}

        else:
            map_score, class_aps =compute_map(
                all_pred_entries, all_gt_boxes_by_img, len(VOC_CLASSES), iou_threshold, epoch = epoch, save_dir=save_dir, last_epoch=ttl_epoch-1)
            
            results = {}
            results['map50'] = map_score

            print(f"[Validation Result]")
            print(f"mAP@{iou_threshold}: {map_score:.4f}")

            for class_id, ap in class_aps.items():
                class_name = VOC_CLASSES[class_id]
                results[class_name] = ap

            return {"full": results, "map50": float(results["map50"].item())}


def eval_metrics_pkg(val_loader, model, image_size):

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for b, (imgs, targets) in enumerate(tqdm(val_loader, desc="Validating", ncols=80)):
            imgs = imgs.to(DEVICE)
            preds = model(imgs) # [B, A*(5+C), S, S] 
            decoded_preds = decode_pred(
                preds, anchors=torch.tensor(torch.tensor(VOC_ANCHORS, dtype=torch.float32), dtype=torch.float32), num_classes=len(VOC_CLASSES), image_size=image_size, conf_thresh=0.01)

            for img_idx in range(len(imgs)):
                # Predictions: decoded (xyxy pixels) -> classwise NMS -> pack dict
                pred_boxes = decoded_preds[img_idx] # [[x1,y1,x2,y2,conf,cls], ...]
                if pred_boxes.shape[0] > 0:
                    boxes = pred_boxes[:, :4]
                    scores = pred_boxes[:, 4]
                    labels = pred_boxes[:, 5].long()
                    # Batched NMS
                    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    preds_list.append({
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels
                    })

                else:
                    preds_list.append({
                        "boxes": torch.empty((0, 4), device=DEVICE, dtype=torch.float32),
                        "scores": torch.empty((0, ), device=DEVICE, dtype=torch.float32),
                        "labels": torch.empty((0, ), device=DEVICE, dtype=torch.long)
                    })

                target_tensor = targets[img_idx] # [S, S, A, 5+C]
                # Ground truth: use shared helper -> pack dict
                gt_xyxy, gt_labels = target_tensor_to_gt(target_tensor, image_size)

                targets_list.append({
                    "boxes": gt_xyxy.clone().to(DEVICE),
                    "labels": gt_labels.clone().to(DEVICE)
                })

    # Compute mAP
    metric.update(preds_list, targets_list)
    results = metric.compute()

    return {"full": results, "map50": float(results["map_50"].item())}



def eval_metric_col(predictions, ground_truths, device=None):
    metric = MeanAveragePrecision(box_format = "xyxy", iou_type="bbox", class_metrics=True)

    preds_list = []
    targets_list = []
    
    for pred_boxes, gt_boxes in zip(predictions, ground_truths):

        if isinstance(gt_boxes, list):
            gt_boxes = torch.tensor(gt_boxes, device=device) if len(gt_boxes) > 0 else torch.empty((0, 5), device=device)

        if len(pred_boxes) > 0:
            boxes = pred_boxes[:, :4]
            scores = pred_boxes[:, 4]
            labels = pred_boxes[:, 5].long()
            keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
            nms_pred_boxes = pred_boxes[keep]
        else:
            nms_pred_boxes = torch.empty((0, 6), device=device)

        # Format predictions
        if nms_pred_boxes.shape[0] > 0: 
            preds_list.append({
                "boxes": nms_pred_boxes[:, :4],
                "scores": nms_pred_boxes[:, 4],
                "labels": nms_pred_boxes[:, 5].long()
            })
        else:
            preds_list.append({
                "boxes": torch.empty((0, 4), device=device, dtype=torch.float32),
                "scores": torch.empty((0,), device=device, dtype=torch.float32),
                "labels": torch.empty((0,), device=device, dtype=torch.long) 
            })

        # Format ground truth 
        if gt_boxes.shape[0] > 0:
            targets_list.append({
                "boxes": gt_boxes[:, :4],
                "labels": gt_boxes[:, 4].long()
            })
        else:
            targets_list.append({
                "boxes": torch.empty((0, 4), device=device, dtype=torch.float32),
                "labels": torch.empty((0,), device=device, dtype=torch.long)
            })
    
    metric.update(preds_list, targets_list)
    results = metric.compute()

    return results



def eval_metric_col_handcraft(all_preds, all_gt_boxes, iou_threshold=0.5, device=None):
    """
    Compute Average Precision for a specific class
    """
    pred_entries = []
    for img_id, pred_boxes in enumerate(all_preds):
        if pred_boxes.shape[0] == 0:
            continue

        boxes = pred_boxes[:, :4]
        scores = pred_boxes[:, 4]
        labels = pred_boxes[:, 5].long()
        keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
        nms_pred_boxes = pred_boxes[keep]

        for pred_box in nms_pred_boxes:
            conf = pred_box[4] # scalar of tensor
            pred_entries.append((img_id, conf, pred_box[:4])) 

    # Sort predictions by confidence
    pred_entries = sorted(pred_entries, key=lambda x:x[1], reverse = True)

    gt_boxes_entries = {}
    for img_id, gt_boxes in enumerate(all_gt_boxes):
        if isinstance(gt_boxes, list):
            gt_boxes = torch.tensor(gt_boxes, device=DEVICE) if len(gt_boxes) > 0 else torch.empty((0, 5), device=device)
        converted_gt = gt_boxes[:, :4] if gt_boxes.shape[0] > 0 else torch.empty((0, 4), device=device)
        
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
        ious = calculate_iou_xyxy_tensor(
            gt_list, pred_box.unsqueeze(0)).squeeze(1) if gt_list.shape[0] > 0 else torch.tensor([])

        if len(ious) == 0:
            fp_list.append(1)
            tp_list.append(0)
            continue
            
        best_iou, best_idx = torch.max(ious, dim=0)

        if best_iou >= iou_threshold and used_gt[img_id][best_idx] == 0:
            tp_list.append(1)
            fp_list.append(0)
            used_gt[img_id][best_idx] = 1
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = torch.cumsum(torch.tensor(tp_list), dim=0)
    fp_cum = torch.cumsum(torch.tensor(fp_list), dim=0)

    recalls = tp_cum / (total_gt + 1e-6)  # recall = TP / (TP + FN)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6) # precision = TP / (TP + FP)

    recall_vals = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precision_vals = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    # Monotonic non-increasing smoothing
    for i in range(len(precision_vals) - 2, -1, -1):
        precision_vals[i] = max(precision_vals[i], precision_vals[i + 1])

    ap = torch.trapz(precision_vals, recall_vals).item()

    return ap, recall_vals.numpy(), precision_vals.numpy()
