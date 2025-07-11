import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import DEVICE, DATASET_ROOT

import preprocess
import model
import ast

# Setting
anchors = [[1,2], [2,1], [1.5, 1.5], [2,2], [1,1]]
num_classes = 80
image_size = 160 # mcunet

def yolo_collate_fn(batch):
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)  # keep list of [num_objects_i, 5] tensors

    return torch.stack(images, dim=0), targets

# Read coco_labels
with open("coco_labels.txt", "r") as f:
    coco_label_map = ast.literal_eval(f.read())

# Load model and loss 
net = model.McuYolo(num_classes = num_classes, num_anchors = len(anchors)).to(DEVICE)
loss_fn = model.Yolov2Loss(num_classes = num_classes, anchors = anchors).to(DEVICE)
optimiser = optim.Adam(net.parameters(), lr=1e-4) #optimiser: adam, sgd, ada ... 

# Load dataset
# image_dir, ann_file, coco_label_map, image_size=160, max_samples=None)
train_dataset = preprocess.YoloDataset(
    image_dir = DATASET_ROOT + "train/data", 
    ann_file =  DATASET_ROOT + "raw/instances_train2017.json", 
    coco_label_map = coco_label_map, max_samples = 5)
val_dataset = preprocess.YoloDataset(
    image_dir = DATASET_ROOT + "validation/data", 
    ann_file =  DATASET_ROOT + "raw/instances_val2017.json", 
    coco_label_map = coco_label_map, max_samples = 5)

# batch_size: number of images (samples) processed together in one forward pass
# images: a tensor of batch_size images, shape [2, C, H, W]
# targets: a list of batch_size elements 
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn = yolo_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn = yolo_collate_fn)


# Training Loop
for epoch in range(1):
    net.train()
    # imgs: a tensor of batch_size images, shape [batch_size, C, H, W]
    # targets: a list of batch_size elements
    for i, (imgs, targets) in enumerate(train_loader):
        if i >=3 : break

        imgs = imgs.to(DEVICE)
        preds = net(imgs)
        loss = loss_fn(preds, targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step() # update params: SGD

        print(f"Train Step {i}, Loss: {loss.item():.4f}")

# Validation
net.eval()
with torch.no_grad():
    for i, (imgs, targets) in enumerate(val_loader):
        if i >= 1: break
        imgs = imgs.to(DEVICE)
        preds = net(imgs)
        loss = loss_fn(preds, targets)

        print(f"Val Step {i}, Loss: {loss.item():.4f}")