import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
import numpy as np
import fiftyone as fo
from random import shuffle
from PIL import Image
from pycocotools.coco import COCO

# dataset = foz.load_zoo_dataset(
#     "coco-2017", split="validation")

def get_dataset(str):

    fo.config.dataset_zoo_dir = "/vol/bitbucket/cc2224/cocodataset"

    dataset = fo.zoo.load_zoo_dataset("coco-2017", split=str, max_samples=5)

    return dataset
