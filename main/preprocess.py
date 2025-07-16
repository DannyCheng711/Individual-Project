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
from PIL import Image, ImageDraw, ImageFont


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
        resize_image = transforms.functional.to_pil_image(self.transform(image))
        draw = ImageDraw.Draw(resize_image)
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

            x, y, w, h = ann["bbox"] # x, y are top-left
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height

            class_name = self.coco.loadCats(ann["category_id"])[0]["name"]
            if class_name != "person": continue 
            class_id = 0
            # class_id = self.coco_label_map.get(class_name, -1)
            # if class_id == -1: continue          

            # assign the GT box to a grid cell 
            cx = x_center * self.image_size
            cy = y_center * self.image_size
            bw = norm_w * self.image_size
            bh = norm_h * self.image_size

            xmin = cx - bw / 2
            ymin = cy - bh / 2
            xmax = cx + bw / 2
            ymax = cy + bh / 2

            # Draw bbox 
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            draw.text((xmin,  ymin - 10), class_name, fill="white")

            target.append(torch.tensor([x_center, y_center, norm_w, norm_h, class_id], device = DEVICE))

        # Skip sample if no valid "person" bbox found
        if len(target) == 0:
            return self.__getitem__((idx + 1) % len(self))

        if idx < 10:
            out_path = os.path.join("./images/", f"original_{img_id}.jpg")
            resize_image.save(out_path)

        return image, target