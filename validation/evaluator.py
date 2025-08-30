# Standard library
import os
# Third-party
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv
# Local application
from config import VOC_CLASSES
from .metrics import eval_metrics, eval_metrics_handcraft, eval_metrics_multiview, eval_metrics_tiny, eval_metrics_ram_tiny
from .visualization import visualize_predictions
from dataset.vocdatset import YoloVocDataset, MultiViewYoloVocDataset

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

    def yolo_collate_fn_multiview(self, batch):
        # batch: list of ((img_a, img_b), (gt_a, gt_b))
        img_a = [item[0][0] for item in batch]
        img_b = [item[0][1] for item in batch]
        gt_a = [item[1][0] for item in batch]
        gt_b = [item[1][1] for item in batch]
        # Stack each view
        imgs = (torch.stack(img_a, dim=0).to(DEVICE), torch.stack(img_b, dim=0).to(DEVICE))
        targets = (torch.stack(gt_a, dim=0).to(DEVICE), torch.stack(gt_b, dim=0).to(DEVICE))
        return imgs, targets


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
    
    def get_multiview_val_loader(self):
        anchors_list = self.anchors.tolist()
        val_dataset = MultiViewYoloVocDataset(
            voc_dataset=self.val_voc_raw, 
            image_size=self.image_size,
            S=self.grid_num,
            anchors=anchors_list,
            num_classes=self.num_classes
        )
        train_loader = DataLoader(val_dataset , batch_size=self.batch_size, shuffle=True, collate_fn=self.yolo_collate_fn_multiview)
        return train_loader
    
    def evaluate(self, model, epoch=None, iou_thresh=0.5, visual_conf_thresh=0.5):
        val_loader = self.get_val_loader()
        if epoch == self.epoch_num - 1:
            visualize_predictions(val_loader, model, self.anchors, image_size=self.image_size, num_classes=self.num_classes, conf_thresh=visual_conf_thresh, save_dir=self.save_dir)
        if self.pkg:
            map_result = eval_metrics(val_loader, model, image_size=self.image_size)
        else:
            map_result = eval_metrics_handcraft(val_loader, model, image_size=self.image_size, epoch=epoch, ttl_epoch=self.epoch_num, iou_threshold=iou_thresh, save_dir=self.save_dir)

        return map_result
    
    def evaluate_multiview(self, model):
        val_loader = self.get_multiview_val_loader()
        # if epoch == self.epoch_num - 1:
        #    visualize_predictions(val_loader, model, self.anchors, image_size=self.image_size, num_classes=self.num_classes, conf_thresh=visual_conf_thresh, save_dir=self.save_dir)
        
        map_result = eval_metrics_multiview(val_loader, model, image_size=self.image_size)

        return map_result
    
    def evaluate_tiny(self, tflite_path):
        val_loader = self.get_val_loader()
        map_result = eval_metrics_tiny(val_loader, tflite_path, image_size=self.image_size)
        ram_footprint, inference_time = eval_metrics_ram_tiny(tflite_path)
        return map_result, ram_footprint, inference_time 
    

        