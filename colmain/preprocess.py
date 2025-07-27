import os 
import json
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

class CO3DOcclusionDataset(Dataset):
    def __init__(self, images_dir, transform=None, generate_occlusions=False):
        """
        Args:
            images_dir: Directory containing RGB images
            transform: torchvision transforms
            generate_occlusions: if true, apply occlusion augmentation 
        """

        self.images_dir = images_dir
        self.transform = transform
        self.generate_occlusions = generate_occlusions
        self.image_mask_pairs = self._get_image_mask_pairs()

    # def _get_image_mask_pairs(self):
