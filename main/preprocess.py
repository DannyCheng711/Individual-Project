import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
import numpy as np
import fiftyone as fo
import torch
import torchvision.transforms as transforms
from random import shuffle
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from config import DEVICE, DATASET_ROOT


class YoloDataset(Dataset):
    def __init__(self, image_dir, ann_file, coco_label_map, image_size=160, max_samples=None):

        # fo.config.dataset_zoo_dir = "/vol/bitbucket/cc2224/cocodataset"
        # fo.config.show_progress_bars = True
        # fo.config.launch_app = False
        # fo.zoo.load_zoo_dataset("coco-2017", split="train", max_samples=None, overwrite=False)
        # fo.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=None, overwrite=False)
        
        self.image_dir = image_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())[:max_samples] if max_samples else list(self.coco.imgs.keys())
        self.coco_label_map = coco_label_map
        self.image_size = image_size

        # print image info
        print(f"Total images: {len(self.ids)}")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        img_id = self.ids[idx]
        # print(f"Sample image id: {img_id}")
        # show image metadata
        img_info = self.coco.loadImgs(img_id)[0]
        # print(f"Image info: {img_info}")


        img_path = os.path.join(self.image_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        width, height = img_info['width'], img_info['height']
        image = self.transform(image)

        # show associated annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # print(ann_ids)
        anns = self.coco.loadAnns(ann_ids)
        # print("Annotations:")
        # for ann in anns:
        #     print(ann)

        target = []
        for ann in anns:
            if ann["iscrowd"]:
                continue
            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height

            class_id = self.coco_label_map.get(self.coco.loadCats(ann["category_id"])[0]["name"], -1)
            if class_id == -1:
                continue

            target.append(torch.tensor([x_center, y_center, norm_w, norm_h, class_id], device = DEVICE))

        return image, target