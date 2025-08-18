import os
import random 
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dotenv import load_dotenv
from config import VOC_ANCHORS, VOC_CLASSES
from dataset.vocdatset import YoloVocDataset
from models.mcunet.mcunet.model_zoo import net_id_list, build_model
from models.dethead.yolodet import McuYolo, Yolov2Loss, MobilenetV2Taps, McunetTaps
from .logging import RunManager 

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")

class Trainer:
    def __init__(self, train_voc_raw, anchors, num_classes, image_size, grid_num, epoch_num, batch_size, aug):
        # anchor is grid unit 
        self.anchors = anchors
        self.num_classes = num_classes # VOC has 20 classes
        self.image_size = image_size # mcunet
        self.epoch_num = epoch_num  
        self.grid_num = grid_num
        self.batch_size = batch_size
        self.train_voc_raw = train_voc_raw
        self.aug = aug

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # store for logs
        self.base_lr = 1e-3
        self.weight_decay = 1e-4
        self.warmup_epochs = 5
        self.eta_min = self.base_lr * 0.05

    def yolo_collate_fn(self, batch):
        images, targets = [], []
        for img, target in batch:
            images.append(img)
            targets.append(target)  # keep list of [num_objects_i, 5] tensors
        return torch.stack(images, dim=0).to(DEVICE), torch.stack(targets, dim=0).to(DEVICE) # [B, 3, H, W], [B, S, S, A, 5 + C]

    def param_groups_no_decay(self, model, wd):
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias"):  # BN/affine/bias -> no decay
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    
    def model_construct(self, net_id):
        print(net_id_list)
        self.anchors = self.anchors.to(DEVICE)
        backbone, _, _ = build_model(net_id=net_id, pretrained=True)
        # Load model and loss 
        # self, taps, num_classes=20, num_anchors=5, final_ch=320, passthrough_ch=96, mid_ch=512, s2d_r=2
        if net_id == "mbv2-w0.35":
            taps = MobilenetV2Taps(backbone, passthrough_idx=12, final_idx=16)
            self.model = McuYolo(taps=taps, num_classes=self.num_classes, num_anchors=len(self.anchors), final_ch=112, passthrough_ch=32, mid_ch=512).to(DEVICE)
        if net_id == "mcunet-in4":
            taps = McunetTaps(backbone, passthrough_idx=12, final_idx=16)
            self.model = McuYolo(taps=taps, num_classes=self.num_classes, num_anchors=len(self.anchors), final_ch=320, passthrough_ch=96, mid_ch=512).to(DEVICE)

        self.criterion = Yolov2Loss(num_classes=self.num_classes, anchors=self.anchors).to(DEVICE)
        
        # AdamW + no-decay on BN/bia
        pg = self.param_groups_no_decay(self.model, self.weight_decay)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer = optim.AdamW(pg, lr=self.base_lr, betas=(0.9, 0.999))
        # warmup + cosine LR - Cosine will run for E-W epochs after warmup
        E, W = self.epoch_num, self.warmup_epochs
        # eta_min: lowerbound of the learning rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=E - W, eta_min=self.eta_min
        )
 
    def get_train_loader(self):
        anchors_list = self.anchors.tolist()
        train_dataset = YoloVocDataset(
            voc_dataset=self.train_voc_raw, 
            image_size=self.image_size,
            S=self.grid_num,
            anchors=anchors_list,
            num_classes=self.num_classes,
            aug=self.aug
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.yolo_collate_fn)
        return train_loader

    def model_train(self, evaluator=None, model_path = None):
        """
        Train the model, save checkpoints, and use evaluator if provided.
        """
        print(f"[INFO] Logging to {model_path}")
        run_manager = RunManager(run_dir=model_path)
        loss_log = {
            'total': [], 'coord': [], 'obj': [], 'class': []} 
        train_loader = self.get_train_loader()
        best_map = -1.0
        best_epoch = -1

        E, W = self.epoch_num, self.warmup_epochs

        for epoch in range(self.epoch_num):
            self.model.train()
            
            cur_lr = self.optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch}] LR: {cur_lr:.6g}")

            sum_total = 0
            sum_coord = 0
            sum_obj = 0
            sum_cls = 0
            n_batches = 0

            # imgs: a tensor of batch_size images, shape [batch_size, C, H, W]
            # i: counts which batch you’re on (starting from 0).
            # step: (data number / batchsize)
            for i, (imgs, targets) in enumerate(train_loader):
                imgs = imgs.to(DEVICE)
                preds = self.model(imgs)
                loss_dict = self.criterion(preds, targets, imgs = imgs)
                loss = loss_dict['total']    
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step() 

                 # Logging
                n_batches += 1
                sum_total += float(loss)
                sum_coord += float(loss_dict['coord'])
                sum_obj   += float(loss_dict['obj'])
                sum_cls   += float(loss_dict['class'])

                if i % 10 == 0:
                    print(
                        f"[Epoch {epoch}] Batch {i} "
                        f"Loss: {loss.item():.4f}, "
                        f"Coord: {loss_dict['coord'].item():.4f}, "
                        f"Obj: {loss_dict['obj'].item():.4f}, "
                        f"Class: {loss_dict['class'].item():.4f}"
                    )

            if n_batches == 0:
                avg_total = 0
                avg_coord = 0
                avg_obj = 0
                avg_cls = 0
            else:
                avg_total = sum_total / n_batches
                avg_coord = sum_coord / n_batches
                avg_obj   = sum_obj   / n_batches
                avg_cls   = sum_cls   / n_batches
            
            loss_log['total'].append(avg_total)
            loss_log['coord'].append(avg_coord)
            loss_log['obj'].append(avg_obj)
            loss_log['class'].append(avg_cls)
            
            # warmup → cosine step
            if epoch < W:
                warm_frac = (epoch + 1) / W
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.base_lr * warm_frac
            else:
                self.scheduler.step()
                
            # evalutate on validation set
            # validation
            val_map50 = None
            if evaluator is not None:
                val_result = evaluator.evaluate(self.model, epoch=epoch)  # make it return dict
                # expect: {"mAP": float, "Recall": float, ...}
                val_map50 = float(val_result.get("map50", 0.0))

            # per-epoch CSV log 
            run_manager.log_epoch(
                epoch=epoch,
                lr=self.optimizer.param_groups[0]['lr'],
                train_loss= avg_total if avg_total is not None else 0.0,
                val_map50=val_map50
            )

            # config snapshot once (keeps ckpt self-describing)
            """
            cfg = {
                "S": getattr(self, "grid_num", None),
                "resolution": getattr(self, "image_size", None),
                "num_classes": self.num_classes,
                "anchors": [a.tolist() if hasattr(a, "tolist") else a for a in self.anchors],
                "backbone": "mcunet-in4",
                "optimizer": "Adam",
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            """
            cfg = {
                "S": self.grid_num,
                "resolution": self.image_size,
                "num_classes": self.num_classes,
                "anchors": [a.tolist() if hasattr(a, "tolist") else a for a in self.anchors],
                "backbone": "mcunet-in4",
                "optimizer": "AdamW",
                "lr_base": self.base_lr,
                "weight_decay": self.weight_decay,
                "warmup_epochs": W,
                "eta_min": self.eta_min,
                "seed": 42,
                "grad_clip_norm": 10.0,
            }

            # save best (overwrite)
            if val_map50 is not None and val_map50 > best_map:
                best_map = val_map50
                best_epoch = epoch
                run_manager.save_ckpt("best.pth", self.model, self.optimizer, epoch, best_map, cfg)
                print(f"[TRAIN] Current Best mAP@0.5={best_map:.4f} @ epoch {best_epoch}")
        # always save last at the end
        run_manager.save_ckpt("last.pth", self.model, self.optimizer, self.epoch_num-1, best_map, cfg)
        print(f"[TRAIN] Best mAP@0.5={best_map:.4f} @ epoch {best_epoch}")
        
        return self.model, loss_log
