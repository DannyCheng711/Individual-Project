import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import DEVICE, DATASET_ROOT

import preprocess
import model
import ast

# Setting
# [[3.787375055964466, 8.848904227653101], 
# [9.548669994570474, 23.4700443938721], 
# [18.833913955423565, 48.82082770930397], 
# [31.108550296889405, 84.25270801120334], 
# [74.38241622664715, 130.3201909498735]]
"""
anchors = [
    [1.0, 1.0],   # 32×32 — medium square
    [1.5, 1.0],   # 48×32 — horizontal
    [1.0, 1.5],   # 32×48 — vertical
    [2.0, 2.0],   # 64×64 — large square
    [3.0, 2.0]    # 96×64 — wide object
]
"""
anchors = [[3.787375055964466 / 32 , 8.848904227653101 / 32], 
    [9.548669994570474 / 32, 23.4700443938721 / 32], 
    [18.833913955423565 / 32, 48.82082770930397 / 32], 
    [31.108550296889405 / 32, 84.25270801120334 / 32], 
    [74.38241622664715 / 32, 130.3201909498735 / 32]]

num_classes = 1 # COCO 80, but we only detect human 
image_size = 160 # mcunet
epoch_num = 160

def yolo_collate_fn(batch):
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)  # keep list of [num_objects_i, 5] tensors

    return torch.stack(images, dim=0), targets

def save_preds_image(img_tensor, pred_tensor, step, save_dir="./images", threshold = 0.5):
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
                top_class_name = [k for k, v in coco_label_map.items() if v == top_class][0]

                # decode box
                # tx, ty: Model-predicted offsets relative to grid cell top-left
                # cx, cy: Final box center coordinates, normalized to [0, 1] over the entire image
                cx = (j + torch.sigmoid(tx)) / S2
                cy = (i + torch.sigmoid(ty)) / S1
                # normalised to image width
                bw = anchors[a][0] * torch.exp(tw) / S2
                bh = anchors[a][1] * torch.exp(th) / S1

                # Convert to pixel coordinations
                w, h = img.size
                xmin = (cx - bw / 2) * w
                ymin = (cy - bh / 2) * h
                xmax = (cx + bw / 2) * w
                ymax = (cy + bh / 2) * h 
                # Draw box
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
                draw.text((xmin, ymin - 10), f"{top_class_name} {obj:.2f}", fill="white")

    img.save(os.path.join(save_dir, f"step{step}.jpg"))

def plot_loss(loss_log, save_path = "./images/loss_plot.png"):
    for k in loss_log:
        plt.plot(loss_log[k], label = k)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# Read coco_labels
with open("coco_labels.txt", "r") as f:
    coco_label_map = ast.literal_eval(f.read())

# Load model and loss 
net = model.McuYolo(num_classes = num_classes, num_anchors = len(anchors)).to(DEVICE)
loss_fn = model.Yolov2Loss(num_classes = num_classes, anchors = anchors).to(DEVICE)
optimiser = optim.Adam(net.parameters(), lr=1e-4)  # already reasonable

# optimiser = optim.SGD(
#     net.parameters(),
#     lr=1e-4, # start with 1e-3
#     momentum=0.9,
#     weight_decay=5e-4
# )

# Reduce LR by 10× at epoch 60 and 90
scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=[60, 90], gamma=0.1)

# Load dataset
# image_dir, ann_file, coco_label_map, image_size=160, max_samples=None)
train_dataset = preprocess.YoloDataset(
    image_dir = DATASET_ROOT + "train/data", 
    # ann_file =  DATASET_ROOT + "raw/instances_train2017.json", 
    ann_file =  DATASET_ROOT + "raw/filtered_instances_train2017.json", 
    coco_label_map = coco_label_map, max_samples = None)
val_dataset = preprocess.YoloDataset(
    image_dir = DATASET_ROOT + "validation/data", 
    # ann_file =  DATASET_ROOT + "raw/instances_val2017.json", 
    ann_file =  DATASET_ROOT + "raw/filtered_instances_val2017.json", 
    coco_label_map = coco_label_map, max_samples = None)

# batch_size: number of images (samples) processed together in one forward pass
# images: a tensor of batch_size images, shape [2, C, H, W]
# targets: a list of batch_size elements 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn = yolo_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn = yolo_collate_fn)


# Freeze the backbone parameters 
net.freeze_backbone() 
# Training Loop
# loss_log = {'total': [], 'coord': [], 'obj': [], 'cls': []}
loss_log = {'total': [], 'coord': [], 'obj': []}
for epoch in range(epoch_num):
    net.train()
    # print(f"Model is on device: {next(net.parameters()).device}")
    # imgs: a tensor of batch_size images, shape [batch_size, C, H, W]
    # i: counts which batch you’re on (starting from 0).
    # targets: a list of batch_size elements
    # step: (data number / batchsize)
    for i, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(DEVICE)

        # print(f"Image batch is on device: {imgs.device}")
        
        preds = net(imgs)
        loss_dict = loss_fn(preds, targets, imgs = imgs)
        loss = loss_dict['total']    
        optimiser.zero_grad()
        loss.backward()
        optimiser.step() 

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

        if epoch == epoch_num - 1:
            # only save first image for debugging
            for img_idx, img in enumerate(imgs):
                save_preds_image(imgs[img_idx], preds, img_idx)
    
    # scheduler.step()

plot_loss(loss_log)

# Validation
net.eval()
with torch.no_grad():
    for i, (imgs, targets) in enumerate(val_loader):
        
        imgs = imgs.to(DEVICE)
        preds = net(imgs)
        loss_dict = loss_fn(preds, targets, imgs = imgs)
        loss = loss_dict['total']    

        if i % 10 == 0:
            print(f"[Step {i}, Loss: {loss.item():.4f}")
            # only save first image for debugging
            save_preds_image(imgs[0], preds, i)