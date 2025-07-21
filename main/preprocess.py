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

class YoloVocDataset(Dataset):
    def __init__(self, voc_dataset, dataset_name, image_size = 160):
        self.voc = voc_dataset
        self.image_size = image_size
        if dataset_name == "train":
            print("This dataset is after augmentation")
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # color, brightness
                transforms.RandomHorizontalFlip(p=0.5), # mirror
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)), # rotation, transition, scaling
                transforms.Resize((image_size, image_size)), 
                transforms.ToTensor()
            ])
        else:
            print("This dataset is NOT after augmentation")
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, i):
        img, target = self.voc[i]
        img_width = int(target['annotation']['size']['width'])
        img_height = int(target['annotation']['size']['height'])

        objs = target['annotation'].get('object', [])
        if not isinstance(objs, list):
            objs = [objs]


        target = []
        for obj in objs:
            if obj['name'] != "person":
                continue
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])

            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            norm_w = (xmax - xmin) / img_width
            norm_h = (ymax - ymin) / img_height
            target.append(torch.tensor([x_center, y_center, norm_w, norm_h, 0], dtype=torch.float32))  # class_id=0

        # temp
        if len(target) == 0:
            return self.__getitem__((i + 1) % len(self))
        
        # Convert image and draw (optional)
        image = self.transform(img)

        target_tensor = torch.stack(target)  # shape: [num_objects, 5] 
        
        return image, target_tensor # image: [3, H, W]; target_tensor: [N, 5]



class YoloDataset(Dataset):
    def __init__(self, image_dir, ann_file, image_size=160, max_samples=None):

        # fo.config.dataset_zoo_dir = "/vol/bitbucket/cc2224/cocodataset"
        # fo.config.show_progress_bars = True
        # fo.config.launch_app = False
        # fo.zoo.load_zoo_dataset("coco-2017", split="train", max_samples=None, overwrite=False)
        # fo.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=None, overwrite=False)
        
        self.image_dir = image_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())[:max_samples] if max_samples else list(self.coco.imgs.keys())
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

            target.append(torch.tensor([x_center, y_center, norm_w, norm_h, class_id], dtype=torch.float32))

        # Skip sample if no valid "person" bbox found
        if len(target) == 0:
            return self.__getitem__((idx + 1) % len(self))

        """
        if idx < 10:
            out_path = os.path.join("./images/", f"original_{img_id}.jpg")
            resize_image.save(out_path)
        """

        target_tensor = torch.stack(target)
        
        return image, target_tensor