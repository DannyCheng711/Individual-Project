import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from config import VOC_CLASS_TO_IDX
from PIL import Image

class YoloVocDataset(Dataset):
    def __init__(self, voc_dataset, image_size = 160, S = 5, anchors = None, num_classes = 20, aug = False):
        self.voc = voc_dataset
        self.image_size = image_size
        self.S = S 
        self.anchors = anchors 
        self.num_classes = num_classes 

        if aug:
            print(f"This training dataset is after augmentation")
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # color, brightness
                # transforms.RandomHorizontalFlip(p=0.5), # mirror
                # transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)), # rotation, transition, scaling
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # imagenet normalisation
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

        boxes = []
        for obj in objs:

            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            class_name = obj['name'].lower().strip()
            
            # skip unknown class name 
            if class_name not in VOC_CLASS_TO_IDX:
                continue

            class_idx = VOC_CLASS_TO_IDX[class_name]

            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            norm_w = (xmax - xmin) / img_width
            norm_h = (ymax - ymin) / img_height

            boxes.append(torch.tensor([x_center, y_center, norm_w, norm_h, class_idx]))
        
        if len(boxes) == 0:
            return self.__getitem__((i + 1) % len(self))

        image = self.transform(img)
        target_tensor = torch.stack(boxes)

        # Encode to [S, S, A, 5+C]
        y_true_tensor = self.encode_targets(target_tensor)

        return image, y_true_tensor #[3, H, W], [S, S, A, 5+C]
    
    def encode_targets(self, boxes):
        """
        boxes: torch.Tensor [N, 5] in [x, y, w, h, class_id] format (normalized to [0,1] image unit) 
        returns: torch.Tensor [S, S, A, 5+C], (tx, ty, tw, th, obj_conf, class_logits)
        """
        S = self.S 
        A = len(self.anchors)
        C = self.num_classes
        y_true = torch.zeros((S, S, A, 5 + C), dtype=torch.float32)

        for box in boxes:
            cx, cy, w, h, cls = box.tolist() # image size unit 
            gi = int(cx * S)
            gj = int(cy * S) # to define the location of the grid cell 
            w_grid = w * S
            h_grid = h * S 

            # Add bounds checking
            gi = min(gi, S - 1)
            gj = min(gj, S - 1)

            # Anchor matching (by IoU of w/h)
            best_iou = 0
            best_a = 0 
            
            for idx, (aw, ah) in enumerate(self.anchors):
                inter = min(w_grid, aw) * min(h_grid, ah) # assume locate in the same center
                union = w_grid * h_grid + aw * ah - inter
                iou = inter / (union + 1e-6)
                if iou > best_iou:
                    best_iou = iou 
                    best_a = idx
            
            # Calculate offsets within grid cell (grid unit)
            tx = cx * S - gi # offset in the grid
            ty = cy * S - gj 
            tw = w_grid
            th = h_grid
            
            # Create one-hot encoding for class
            onehot = torch.zeros(C)
            onehot[int(cls)] = 1.0

            target_values = torch.cat([
                torch.tensor([tx, ty, tw, th, 1.0]),  # [tx, ty, tw, th, objectness]
                onehot  # class probabilities
            ])

            y_true[gj, gi, best_a] = target_values
        
        # grid unit 
        return y_true