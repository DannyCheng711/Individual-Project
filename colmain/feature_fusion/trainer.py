import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection

from dataset.voc.vocdataset import MultiViewYoloVocDataset
from models.dethead.yolodet import Yolov2Loss 
from models.dethead.mvyolodet import MultiViewMcuYolo
from models.dethead.mvdet_test import MultiViewYolo
from utils.logging import RunManager 
from dotenv import load_dotenv
from config import VOC_CLASSES, VOC_ANCHORS, VOC_CLASS_TO_IDX
from validation.visualization import plot_loss
from validation.evaluator import Evaluator


load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")


class MultiViewTrainer:
    def __init__(self, train_voc_raw, anchors, num_classes, image_size, grid_num, epoch_num, batch_size, detach_peer):
        # anchor is grid unit 
        self.anchors = anchors
        self.num_classes = num_classes # VOC has 20 classes
        self.image_size = image_size # mcunet
        self.epoch_num = epoch_num  
        self.grid_num = grid_num
        self.batch_size = batch_size
        self.train_voc_raw = train_voc_raw
        self.detach_peer = detach_peer

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # store for logs
        self.base_lr = 1e-3
        self.weight_decay = 1e-4
        self.warmup_epochs = 5
        self.eta_min = self.base_lr * 0.05

    def yolo_collate_fn(self, batch):
    # batch: list of ((img_a, img_b), (gt_a, gt_b))
        img_a = [item[0][0] for item in batch]
        img_b = [item[0][1] for item in batch]
        gt_a = [item[1][0] for item in batch]
        gt_b = [item[1][1] for item in batch]
        # Stack each view
        imgs = (torch.stack(img_a, dim=0).to(DEVICE), torch.stack(img_b, dim=0).to(DEVICE))
        targets = (torch.stack(gt_a, dim=0).to(DEVICE), torch.stack(gt_b, dim=0).to(DEVICE))
        return imgs, targets

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
    
    
    def model_construct(self):
        self.anchors = self.anchors.to(DEVICE)
        # Load model and loss 
        # self, taps, num_classes=20, num_anchors=5, final_ch=320, passthrough_ch=96, mid_ch=512, s2d_r=2
        # Two devices (or two views)
        self.system = MultiViewMcuYolo(num_classes=len(VOC_CLASSES), num_anchors=len(VOC_ANCHORS)).to(DEVICE)
        self.criterion = Yolov2Loss(num_classes=self.num_classes, anchors=self.anchors).to(DEVICE)
        
        # AdamW + no-decay on BN/bia
        # pg = self.param_groups_no_decay(self.model, self.weight_decay)
        pg = self.param_groups_no_decay(self.system, self.weight_decay)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer = optim.AdamW(pg, lr=self.base_lr, betas=(0.9, 0.999))
        # warmup + cosine LR - Cosine will run for E-W epochs after warmup
        E, W = self.epoch_num, self.warmup_epochs
        # eta_min: lowerbound of the learning rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15 - W, eta_min=self.eta_min # 15 is more robust in occlusion scenario
        )
 
    def get_train_loader(self):
        anchors_list = self.anchors.tolist()
        train_dataset = MultiViewYoloVocDataset(
            voc_dataset=self.train_voc_raw, 
            image_size=self.image_size,
            S=self.grid_num,
            anchors=anchors_list,
            num_classes=self.num_classes
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
            self.system.train()
            
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
                imgs_a, imgs_b = imgs
                targets_a, targets_b = targets
                imgs_a = imgs_a.to(DEVICE)
                imgs_b = imgs_b.to(DEVICE)
                targets_a = targets_a.to(DEVICE)
                targets_b = targets_b.to(DEVICE)

                preds, _ = self.system(imgs_a, imgs_b, feature_pass_mode=False)
                # preds = self.system(imgs_a, imgs_b)

                loss_a = self.criterion(preds, targets_a, imgs=imgs_a)
                loss_b = self.criterion(preds, targets_b, imgs=imgs_b)

                loss = 0.5 * (loss_a['total'] + loss_b['total'])

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.system.parameters(), 10.0)
                self.optimizer.step() 

                 # Logging
                n_batches += 1
                sum_total += float(loss)
                sum_coord += float((loss_a['coord'] + loss_b['coord']) / 2)
                sum_obj   += float((loss_a['obj'] + loss_b['obj']) / 2)
                sum_cls   += float((loss_a['class'] + loss_b['class']) / 2)

                if i % 10 == 0:
                    print(
                        f"[Epoch {epoch}] Batch {i} "
                        f"Loss: {loss.item():.4f}, "
                        f"Coord: {float((loss_a['coord'] + loss_b['coord']) / 2):.4f}, "
                        f"Obj: {float((loss_a['obj'] + loss_b['obj']) / 2):.4f}, "
                        f"Class: {float((loss_a['class'] + loss_b['class']) / 2):.4f}"
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
                val_result = evaluator.evaluate_multiview(self.system) 
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
            }

            # save “best”
            if val_map50 is not None and val_map50 > best_map:
                best_map = val_map50
                best_epoch = epoch
                # example: pack both state_dicts
                run_manager.save_ckpt(
                    "best.pth",
                    self.system,
                    self.optimizer, epoch, best_map, cfg
                )
                print(f"[TRAIN] Current Best mAP@0.5={best_map:.4f} @ epoch {best_epoch}")

        # always save last at the end
        run_manager.save_ckpt("last.pth", self.system, self.optimizer, self.epoch_num-1, best_map, cfg)
        
        print(f"[TRAIN] Best mAP@0.5={best_map:.4f} @ epoch {best_epoch}")
        
        return self.system, loss_log


if __name__ == "__main__":

    grid_num = 5
    image_size = 160
    epoch_num = 1

    run_name = f"multiview_S{grid_num}_res{image_size}"
    base_run_dir = os.path.join("./colmain/feature_fusion/runs", run_name)
    os.makedirs(base_run_dir, exist_ok=True)
    eval_dir = os.path.join(base_run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    logs_dir = os.path.join(base_run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    loss_plot_path = os.path.join(base_run_dir, f"loss_plot_{run_name}.png")
    final_model_path = os.path.join(base_run_dir, "multiview_yolo_final.pth")

    train_voc_raw = VOCDetection(
        root=VOC_ROOT, year="2012", image_set="train", download=False)
    
    # 2. Set anchors, classes, and other parameters
    anchors = torch.tensor(VOC_ANCHORS, dtype=torch.float32)  # shape [A, 2]
    num_classes = len(VOC_CLASS_TO_IDX)

    # 3. Initialize the trainer
    trainer = MultiViewTrainer(
        train_voc_raw=train_voc_raw,
        anchors=anchors,
        num_classes=num_classes,
        image_size=image_size,
        grid_num=grid_num,
        epoch_num=epoch_num,
        batch_size=32,
        detach_peer=False
    )

    # 4. Construct the model
    trainer.model_construct()

    # 5. Train the model (optionally pass an evaluator and model_path for logging)

    val_voc_raw = VOCDetection(root=VOC_ROOT, year="2012", image_set="val", download=False)
    
    evaluator = Evaluator(
        val_voc_raw,
        trainer.anchors,
        trainer.num_classes,
        trainer.image_size,
        grid_num=trainer.grid_num,
        batch_size=16, 
        epoch_num=trainer.epoch_num,
        save_dir=eval_dir,
        pkg=True
    )

    trained_model, loss_log = trainer.model_train(
        evaluator=evaluator, 
        model_path=base_run_dir,
    )

    # 6. Save or use the trained_model as needed
    torch.save(trained_model.state_dict(), final_model_path)

    plot_loss(loss_log, loss_plot_path)
    print("Training complete. Loss plot saved.")