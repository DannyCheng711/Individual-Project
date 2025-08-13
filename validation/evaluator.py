# Standard library
import os
# Third-party
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import batched_nms

from tqdm import tqdm
# Local application
from config import VOC_CLASSES
from .bbox_utils import decode_pred, target_tensor_to_gt, xyxy_to_cxcywh_norm, apply_classwise_nms
from .metrics import compute_map
from .visualization import visualize_predictions
from dataset.vocdatset import YoloVocDataset

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")

class Evaluator:
    def __init__(self, val_voc_raw, anchors, num_classes, image_size, grid_num, batch_size, epoch_num, save_dir, pkg):
        self.val_voc_raw = val_voc_raw
        self.anchors = anchors
        self.num_classes = num_classes
        self.image_size = image_size
        self.grid_num = grid_num
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.save_dir = save_dir
        self.pkg = pkg

    def yolo_collate_fn(self, batch):
        images, targets = [], []
        for img, target in batch:
            images.append(img)
            targets.append(target)  # keep list of [num_objects_i, 5] tensors
        return torch.stack(images, dim=0).to(DEVICE), torch.stack(targets, dim=0).to(DEVICE) # [B, 3, H, W], [B, S, S, A, 5 + C]

    def get_val_loader(self):
        anchors_list = self.anchors.tolist()
        val_dataset = YoloVocDataset(
            voc_dataset=self.val_voc_raw, 
            image_size=self.image_size,
            S=self.grid_num,
            anchors=anchors_list,
            num_classes=self.num_classes
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.yolo_collate_fn)
        return val_loader
    
    def evaluate(self, model, epoch=None, iou_thresh=0.5, visual_conf_thresh=0.5):
        val_loader = self.get_val_loader()
        if epoch == self.epoch_num - 1:
            visualize_predictions(val_loader, model, self.anchors, image_size=self.image_size, num_classes=self.num_classes, conf_thresh=visual_conf_thresh, save_dir=self.save_dir)
        if self.pkg:
            map_result = self.eval_metrics_pkg(val_loader, model, num_classes=self.num_classes)
        else:
            map_result = self.eval_metrics(val_loader, model, epoch=epoch, iou_threshold=iou_thresh, save_dir=self.save_dir)

        return map_result
    
    def eval_metrics(self, val_loader, model, epoch = None, iou_threshold = 0.5, save_dir=None):

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
                    preds, anchors=self.anchors, num_classes=self.num_classes, image_size=self.image_size, conf_thresh=0.01)

                for img_idx in range(len(imgs)):
                    img_id = f"{b}_{img_idx}"

                    # Extract ground truth boxes from targets [S, S, A, 5+C]
                    target_tensor = targets[img_idx] # [S, S, A, 5+C]

                    # GT (use helper)
                    gt_xyxy, gt_labels = target_tensor_to_gt(target_tensor, self.image_size)
                    if gt_xyxy.numel() == 0:
                        all_gt_boxes_by_img[img_id] = []
                    else:
                        gt_cxcywh = xyxy_to_cxcywh_norm(gt_xyxy, self.image_size)  # (N,4) normalized
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
                        pred_cxcywh = xyxy_to_cxcywh_norm(boxes, self.image_size)  # normalized (N,4)

                        for idx in range(boxes.shape[0]):
                            all_pred_entries.append(
                                (img_id, float(scores[idx].item()), pred_cxcywh[idx].tolist(), int(labels[idx].item()))
                        )
                    
        # Compute all classes mAP
        map_score, class_aps =compute_map(
            all_pred_entries, all_gt_boxes_by_img, self.num_classes, iou_threshold, epoch = epoch, save_dir=save_dir, last_epoch=self.epoch_num-1)
        
        results = {}
        results['map50'] = map_score

        print(f"[Validation Result]")
        print(f"mAP@{iou_threshold}: {map_score:.4f}")

        for class_id, ap in class_aps.items():
            class_name = VOC_CLASSES[class_id]
            results[class_name] = ap

        return {"full": results, "map50": float(results["map50"].item())}


    def eval_metrics_pkg(self, val_loader, model, num_classes):

        model.eval()
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
        
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for b, (imgs, targets) in enumerate(tqdm(val_loader, desc="Validating", ncols=80)):
                imgs = imgs.to(DEVICE)
                preds = model(imgs) # [B, A*(5+C), S, S] 
                decoded_preds = decode_pred(
                    preds, anchors=self.anchors, num_classes=num_classes, image_size=self.image_size, conf_thresh=0.01)

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
                    gt_xyxy, gt_labels = target_tensor_to_gt(target_tensor, self.image_size)

                    targets_list.append({
                        "boxes": gt_xyxy.clone().to(DEVICE),
                        "labels": gt_labels.clone().to(DEVICE)
                    })

        # Compute mAP
        metric.update(preds_list, targets_list)
        results = metric.compute()

        return {"full": results, "map50": float(results["map_50"].item())}