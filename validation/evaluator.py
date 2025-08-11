# Standard library
import os
# Third-party
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from torchmetrics.detection import MeanAveragePrecision
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
    def __init__(self, val_voc_raw, anchors, num_classes, image_size, grid_num, batch_size):
        self.val_voc_raw = val_voc_raw
        self.anchors = anchors
        self.num_classes = num_classes
        self.image_size = image_size
        self.grid_num = grid_num
        self.batch_size = batch_size

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
    
    def evaluate(self, model, epoch=None, pkg=False, iou_thresh=0.5, visual_conf_thresh=0.5):
        val_loader = self.get_val_loader()
        visualize_predictions(val_loader, model, self.anchors, image_size=self.image_size, num_classes=self.num_classes, conf_thresh=visual_conf_thresh, save_dir="./vocmain/images_voc/")
        if pkg:
            map_result = self.eval_metrics_pkg(val_loader, model)
            print(map_result)
        else:
            self.eval_metrics(val_loader, model, epoch=epoch, iou_threshold=iou_thresh, save_dir="./vocmain/images_voc/")

    
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
                    preds, anchors=self.anchors, num_classes=self.num_classes, image_size=self.image_size, conf_thresh=0.0)

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
                    nms_pred_boxes = apply_classwise_nms(pred_boxes=pred_boxes, iou_thresh=0.5) # apply nms on pred_boxes

                    if len(nms_pred_boxes):
                        # extract xyxy pixels from NMS output
                        pred_xyxy = torch.tensor([p[:4] for p in nms_pred_boxes], dtype=torch.float32)
                        pred_cxcywh = xyxy_to_cxcywh_norm(pred_xyxy, self.image_size)  # normalized (N,4)

                        for idx, (conf, cls_id) in enumerate((p[4], p[5]) for p in nms_pred_boxes):
                            all_pred_entries.append(
                                (img_id, float(conf), pred_cxcywh[idx].tolist(), int(cls_id)))
                            
        # Compute all classes mAP
        map_score, class_aps =compute_map(
            all_pred_entries, all_gt_boxes_by_img, self.num_classes, iou_threshold, epoch = epoch, save_dir=save_dir)
        
        results = {}
        results['mAP'] = map_score

        print(f"[Validation Result]")
        print(f"mAP@{iou_threshold}: {map_score:.4f}")

        for class_id, ap in class_aps.items():
            class_name = VOC_CLASSES[class_id]
            results[class_name] = ap

        return results


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
                    preds, anchors=self.anchors, num_classes=num_classes, image_size=self.image_size, conf_thresh=0.0)

                for img_idx in range(len(imgs)):
                    # Predictions: decoded (xyxy pixels) -> classwise NMS -> pack dict
                    pred_boxes = decoded_preds[img_idx] # [[x1,y1,x2,y2,conf,cls], ...]
                    nms_pred_boxes = apply_classwise_nms(pred_boxes=pred_boxes, iou_thresh=0.5) # Do nms on pred_boxes

                    if len(nms_pred_boxes) > 0:
                        preds_list.append({
                            "boxes": torch.tensor([[p[0], p[1], p[2], p[3]] for p in nms_pred_boxes], device=DEVICE),
                            "scores": torch.tensor([p[4] for p in nms_pred_boxes], device=DEVICE),
                            "labels": torch.tensor([p[5] for p in nms_pred_boxes], device=DEVICE)
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
                        "boxes": torch.tensor(gt_xyxy, device=DEVICE),
                        "labels": torch.tensor(gt_labels, device=DEVICE)
                    })

        # Compute mAP
        metric.update(preds_list, targets_list)
        results = metric.compute()

        return results