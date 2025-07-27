
import torch 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CATEGORY = ["bicycle", "car", "motorbike"]
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
VOC_CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}
