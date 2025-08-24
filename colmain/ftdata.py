import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from config import VOC_CLASS_TO_IDX
from PIL import Image
import numpy as np

class MultiViewYoloVocDataset(Dataset):
    def __init__(self, voc_dataset, image_size = 160, S = 5, anchors = None, num_classes = 20):
        self.voc = voc_dataset
        self.image_size = image_size
        self.S = S 
        self.anchors = anchors 
        self.num_classes = num_classes 

        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
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

        yolo_boxes = []
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

            yolo_boxes.append(torch.tensor([x_center, y_center, norm_w, norm_h, class_idx]))
        
        if len(yolo_boxes) == 0:
            return self.__getitem__((i + 1) % len(self))

        target_tensor = torch.stack(yolo_boxes)

        # Prepare View A (Original)
        img_view_a = self.base_transform(img)
        gt_view_a = self.encode_targets(target_tensor.clone())

        # Prepare View B (Affine-transformed)
        # --- Get the affine parameters ---
        affine_params = transforms.RandomAffine.get_params(
            degrees=(5, 5),
            translate=(0.1, 0.1),
            scale_ranges=(0.9, 1.1),
            shears=None,
            img_size=img.size
        )
        angle, translations, scale, shear = affine_params

        # --- Compose affine matrix (torchvision uses inverse for image, so we use the same) ---
        center = [img.width * 0.5, img.height * 0.5]
        affine_matrix = TF._get_inverse_affine_matrix(
            center=center, angle=angle, translate=translations, scale=scale, shear=shear
        )

        affine_matrix_np = np.array([
            [affine_matrix[0], affine_matrix[1], affine_matrix[2]],
            [affine_matrix[3], affine_matrix[4], affine_matrix[5]]
        ], dtype=np.float32)

        # --- Apply affine to image ---
        img_view_b_pil = TF.affine(img, angle=angle, translate=translations, scale=scale, shear=shear)
        img_view_b = TF.resize(img_view_b_pil, (self.image_size, self.image_size))
        img_view_b = TF.to_tensor(img_view_b)

        # --- Apply affine to boxes ---
        boxes_affine = self.apply_affine_to_boxes(target_tensor.clone(), affine_matrix_np, (img.height, img.width))
        gt_view_b = self.encode_targets(boxes_affine)


        return (img_view_a, img_view_b), (gt_view_a, gt_view_b) #[3, H, W], [S, S, A, 5+C]
    
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
    
    def apply_affine_to_boxes(self, boxes, affine_matrix, img_size):
        """
        boxes: [N, 5] in [x_center, y_center, w, h, class_id] normalized
        affine_matrix: 2x3 matrix from torchvision
        img_size: (H, W)
        Returns: transformed boxes in [x_center, y_center, w, h, class_id] normalized
        """
        H, W = img_size
        new_boxes = []
        for box in boxes:
            x_c, y_c, w, h, cls = box.tolist()
            # Convert to pixel coordinates
            x_c *= W
            y_c *= H
            w *= W
            h *= H
            # Get box corners
            x1 = x_c - w/2
            y1 = y_c - h/2
            x2 = x_c + w/2
            y2 = y_c + h/2
            # Four corners
            corners = torch.tensor([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=torch.float32)
            # Add ones for affine
            ones = torch.ones((4, 1))
            corners = torch.cat([corners, ones], dim=1)  # [4, 3]
            # Apply affine
            M = torch.tensor(affine_matrix, dtype=torch.float32)  # [2, 3]
            new_corners = (M @ corners.T).T  # [4, 2]
            # Get new box
            x_min, y_min = new_corners.min(dim=0).values
            x_max, y_max = new_corners.max(dim=0).values
            # Clamp to image bounds
            x_min = x_min.clamp(0, W)
            y_min = y_min.clamp(0, H)
            x_max = x_max.clamp(0, W)
            y_max = y_max.clamp(0, H)
            # Convert back to center/w/h normalized
            new_x_c = (x_min + x_max) / 2 / W
            new_y_c = (y_min + y_max) / 2 / H
            new_w = (x_max - x_min) / W
            new_h = (y_max - y_min) / H
            new_boxes.append(torch.tensor([new_x_c, new_y_c, new_w, new_h, cls]))
        return torch.stack(new_boxes)