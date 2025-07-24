import tensorflow as tf

# Device config
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
# Dataset config
VOC_ROOT = "/vol/bitbucket/cc2224/voc/"
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
VOC_CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}