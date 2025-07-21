import torch
import os

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset config
DATASET_ROOT = "/vol/bitbucket/cc2224/cocodataset/coco-2017/"
VOC_ROOT = "/vol/bitbucket/cc2224/voc/"