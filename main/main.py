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
from validation import evaluator
import preprocess
import model
import ast

# Setting

"""
# COCO
anchors = [[11.579129865030023 / 32 , 23.897646887526435 / 32], 
    [22.5801077336668254 / 32, 52.59153602786403 / 32], 
    [36.01190595457166 / 32, 90.59964938081846 / 32], 
    [56.857930905762814 / 32, 133.10148018691575 / 32], 
    [125.51095889848243/ 32, 134.68714790018956 / 32]]
"""

class PreTrainStage():
    def __init__(self, train_voc_raw, val_voc_raw,  num_classes = 1, image_size = 160, epoch_num = 150):

        self.anchors = [[8.851960381366869 / 32 , 20.71491425034285 / 32], 
            [27.376232242192298 / 32, 56.73302518805578 / 32], 
            [42.88177452824786 / 32, 98.24243329589638 / 32], 
            [67.68032082718717 / 32, 132.7704493338952 / 32], 
            [131.16250016574756/ 32, 137.1476847579408/ 32]]

        self.num_classes = num_classes # COCO 80, but we only detect human 
        self.image_size = image_size # mcunet
        self.epoch_num = epoch_num  #80
        self.train_voc_raw = train_voc_raw
        self.val_voc_raw = val_voc_raw
        self.net = None
        self.loss_fn = None
        self.optimiser = None

    def yolo_collate_fn(self, batch):
        images = []
        targets = []

        for img, target in batch:
            images.append(img)
            targets.append(target)  # keep list of [num_objects_i, 5] tensors

        return torch.stack(images, dim=0), targets

    def save_preds_image(self, img_tensor, pred_tensor, step, save_dir="./images_voc", threshold = 0.5):
        """
        pred_tensor: Tensor of shape [B, S, S, A*(5+C)] 
        """

        os.makedirs(save_dir, exist_ok=True)
        img = transforms.functional.to_pil_image(img_tensor.cpu())
        draw = ImageDraw.Draw(img)

        B, S1, S2, pred_dim = pred_tensor.shape
        assert S1 == S2, "Grid should be square"
        A = 5 
        C = pred_dim // A - 5

        # reshape to [B, A, S, S, 5+C]
        pred_tensor = pred_tensor.reshape(B, S1, S2, A, 5 + C)
        pred_tensor = pred_tensor.permute(0, 3, 1, 2, 4).contiguous()

        for a in range(A):
            for i in range(S1):
                for j in range(S2):
                    # index 0 for debugging 
                    tx, ty, tw, th, obj_score = pred_tensor[0, a, i, j, :5]
                    obj = torch.sigmoid(obj_score)
                    
                    if obj < threshold: continue

                    class_logits = pred_tensor[0, a, i, j, 5:]
                    class_probs = torch.softmax(class_logits, dim=-1)                
                    top_class = torch.argmax(class_probs).item()
                    # top_class_name = [k for k, v in coco_label_map.items() if v == top_class][0]

                    # decode box
                    # tx, ty: Model-predicted offsets relative to grid cell top-left
                    # cx, cy: Final box center coordinates, normalized to [0, 1] over the entire image
                    cx = (j + torch.sigmoid(tx)) / S2
                    cy = (i + torch.sigmoid(ty)) / S1
                    # normalised to image width
                    bw = self.anchors[a][0] * torch.exp(tw) / S2
                    bh = self.anchors[a][1] * torch.exp(th) / S1

                    # Convert to pixel coordinations
                    w, h = img.size
                    xmin = (cx - bw / 2) * w
                    ymin = (cy - bh / 2) * h
                    xmax = (cx + bw / 2) * w
                    ymax = (cy + bh / 2) * h 
                    # Draw box
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
                    draw.text((xmin, ymin - 10), f"person {obj:.2f}", fill="white")

        img.save(os.path.join(save_dir, f"step{step}.jpg"))


    def plot_loss(self, loss_log, save_path):
        for k in loss_log:
            plt.plot(loss_log[k], label = k)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss over time")
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def model_construct(self):

        # Load model and loss 
        self.net = model.McuYolo(num_classes = self.num_classes, num_anchors = len(self.anchors)).to(DEVICE)
        self.loss_fn = model.Yolov2Loss(num_classes = self.num_classes, anchors = self.anchors).to(DEVICE)
        self.optimiser = optim.Adam(self.net.parameters(), lr=1e-4)  # already reasonable

        # optimiser = optim.SGD(
        #     net.parameters(),
        #     lr=1e-4, # start with 1e-3
        #     momentum=0.9,
        #     weight_decay=5e-4
        # )

        # Reduce LR by 10× at epoch 60 and 90
        # scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=[60, 90], gamma=0.1)


        # Load dataset
        # image_dir, ann_file, coco_label_map, image_size=160, max_samples=None)
        """
        train_dataset = preprocess.YoloDataset(
            image_dir = DATASET_ROOT + "train/data", 
            # ann_file =  DATASET_ROOT + "raw/instances_train2017.json", 
            ann_file =  DATASET_ROOT + "raw/filtered_instances_train2017.json", 
            max_samples = None)
        val_dataset = preprocess.YoloDataset(
            image_dir = DATASET_ROOT + "validation/data", 
            # ann_file =  DATASET_ROOT + "raw/instances_val2017.json", 
            ann_file =  DATASET_ROOT + "raw/filtered_instances_val2017.json", 
            max_samples = None)
        """

        
    def model_train(self, image_size = 160):
        

        # batch_size: number of images (samples) processed together in one forward pass
        # images: a tensor of batch_size images, shape [2, C, H, W]
        # targets: a list of batch_size elements 
        train_dataset = preprocess.YoloVocDataset(self.train_voc_raw, "train", image_size=image_size)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn = self.yolo_collate_fn)

        # Freeze the backbone parameters 
        ## self.net.freeze_backbone() 
        # Training Loop
        # loss_log = {'total': [], 'coord': [], 'obj': [], 'cls': []}
        loss_log = {'total': [], 'coord': [], 'obj': []}
        for epoch in range(self.epoch_num):
            self.net.train()
            # print(f"Model is on device: {next(net.parameters()).device}")
            # imgs: a tensor of batch_size images, shape [batch_size, C, H, W]
            # i: counts which batch you’re on (starting from 0).
            # targets: a list of batch_size elements
            # step: (data number / batchsize)
            for i, (imgs, targets) in enumerate(train_loader):
                imgs = imgs.to(DEVICE)

                # print(f"Image batch is on device: {imgs.device}")
                
                preds = self.net(imgs)
                loss_dict = self.loss_fn(preds, targets, imgs = imgs)
                loss = loss_dict['total']    
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step() 

                # Logging
                loss_log['total'].append(loss.item())
                loss_log['coord'].append(loss_dict['coord'].item())
                loss_log['obj'].append(loss_dict['obj'].item())
                # loss_log['cls'].append(loss_dict['cls'].item())
        
                if i % 10 == 0:
                    print(
                        f"[Epoch {epoch}] Batch {i} "
                        f"Loss: {loss.item():.4f}, "
                        f"Loss_coord: {loss_dict['coord'].item():.4f}, "
                        f"Loss_obj: {loss_dict['obj'].item():.4f}, "
                        # f"Loss_cls: {loss_dict['cls'].item():.4f}"
                    )

                if epoch == self.epoch_num - 1:
                    # only save first image for debugging
                    for img_idx in range(min(2, imgs.size(0))):
                        self.save_preds_image(imgs[img_idx], preds, img_idx)


            if epoch % 10 == 0:
                self.model_val(epoch)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict()
                }
                torch.save(checkpoint, f"./saved_models/yolovoc_150_aug_epoch_{epoch}.pth")


        return self.net, loss_log

    def train_result_save(self, net, loss_log, save_path):
        self.plot_loss(loss_log, save_path = save_path)
        # Save model weights after training
        model_save_path = "./saved_models/yolovoc_final_150_aug.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(net.state_dict(), model_save_path)
        print(f" Model saved to {model_save_path}")
    
    def load_model(self, model_path):
        self.model_construct()
        self.net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.net.eval # ensures correct test-time behavior!
        print(f"Model loaded from {model_path}")

    def model_val(self, epoch = None):
        val_dataset = preprocess.YoloVocDataset(self.val_voc_raw, "val", self.image_size)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn = self.yolo_collate_fn)
    
        # Validation
        self.net.eval()
        evaluator(val_loader, self.net, self.anchors, epoch, iou_threshold = 0.5, image_size = 160)
        

if __name__ == "__main__":

    train_voc_raw = VOCDetection(
            root = VOC_ROOT, year = "2012", image_set = "train", download = False)
    val_voc_raw = VOCDetection(
        root = VOC_ROOT, year = "2012", image_set = "val", download = False)
        
    trainer = PreTrainStage(train_voc_raw, val_voc_raw, num_classes = 1, image_size = 160, epoch_num = 150)
    trainer.model_construct()
    net, log = trainer.model_train()
    trainer.train_result_save(net, log, "./images_voc/loss_plot_150_aug.png")
    ## trainer.load_model("./saved_models/yolovoc_final.pth")
    # trainer.model_val()