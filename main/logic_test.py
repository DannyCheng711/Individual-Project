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
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np 

import preprocess
import model
import ast

# Setting
num_classes = 1 # COCO 80, but we only detect human 
image_size = 160 # mcunet
S = 5 

def compute_iou(box1, box2):
    # box format: (xmin, ymin, xmax, ymax)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def visualise_anchors_vs_gt(image_pil, gt_boxes, anchors, save_path=None):
    draw = ImageDraw.Draw(image_pil)
    w, h = image_pil.size
    
    for box in gt_boxes:
        cx, cy, bw, bh = box[:4]
        xmin = (cx - bw/2) * w
        ymin = (cy - bh/2) * h
        xmax = (cx + bw/2) * w 
        ymax = (cy + bh/2) * h
        draw.rectangle([xmin, ymin, xmax, ymax], outline = "red", width = 2)

    for i in range(S):
        for j in range(S):
            center_x = (j + 0.5) / S
            center_y = (i + 0.5) / S
            for aw, ah in anchors:
                ax = center_x
                ay = center_y
                bw = aw / S
                bh = ah / S

                anchor_box = [
                    (ax - bw / 2) * w,
                    (ay - bh / 2) * h,
                    (ax + bw / 2) * w,
                    (ay + bh / 2) * h
                ]

                for gt in gt_boxes:
                    gt_x, gt_y, gt_w, gt_h = gt[:4]
                    gt_box = [ 
                        (gt_x - gt_w / 2) * w,
                        (gt_y - gt_h / 2) * h,
                        (gt_x + gt_w / 2) * w, 
                        (gt_y + gt_h / 2) * h
                    ]
                    iou = compute_iou(anchor_box, gt_box)
                    if iou >= 0.5:
                        draw.rectangle(anchor_box, outline="blue", width = 1)
    
    image_pil.save(save_path)

# Read coco_labels
with open("coco_labels.txt", "r") as f:
    coco_label_map = ast.literal_eval(f.read())

# Load dataset
# image_dir, ann_file, coco_label_map, image_size=160, max_samples=None)
"""
train_dataset = preprocess.YoloDataset(
    image_dir = DATASET_ROOT + "train/data", 
    ann_file =  DATASET_ROOT + "raw/filtered_instances_train2017.json", 
    max_samples= None)
"""

train_voc_raw = VOCDetection(
    root = VOC_ROOT, year = "2012", image_set = "train", download = False)

train_dataset = preprocess.YoloVocDataset(train_voc_raw, image_size=160)


train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True)

all_boxes = []

for imgs, targets in tqdm(train_loader):
    for t in targets: # list of [x_center, y_center, width, height, class_id]
        for box in t:
            w = box[2].item() * image_size  # un-normalize to pixel scale
            h = box[3].item() * image_size
            all_boxes.append([w, h])

all_boxes_np = np.array(all_boxes)
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(all_boxes_np)
# Get anchor shapes
anchors_pixel = kmeans.cluster_centers_.tolist()
anchors_pixel = sorted(anchors_pixel, key=lambda x: x[0] * x[1])  # sort by area 
print(anchors_pixel)
cell_size = 160 / 5  # = 32
anchors = [[w / cell_size, h / cell_size] for (w, h) in anchors_pixel]

"""
for idx in range(10):
    img_tensor, targets = train_dataset[idx] 

    img_pil = transforms.functional.to_pil_image(img_tensor.cpu())

    visualise_anchors_vs_gt(
        img_pil, targets, anchors, save_path= f"./test/anchor_debug_{idx}.jpg")
"""