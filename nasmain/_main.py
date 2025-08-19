import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # on cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import numpy as np 
import json, sys, random
from dotenv import load_dotenv
from torchvision.datasets import VOCDetection
import tensorflow as tf
import torch

# Add parent directory to Python path for imports (temporary using colmain.validation)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import model 
from config import VOC_ANCHORS, VOC_CLASSES, INFERENCE_CLASS, VOC_CLASS_TO_IDX
from colmain.validation import decode_pred, evaluate_pred_gt, visualize_bbox_grid_tensors_single_view, evaluate_pred_gt_handcraft, plot_pr_curves_comp
from dataset.vocdatset import YoloVocDataset
from .mcunetYolo.tinynas.nn.proxyless_net import ProxylessNASNets
from .mcunetYolo.tinynas.tf.tf2_yolodet import TFObjectDetector
from .preprocess.prepro_config import fix_state_dict_keys, convert_pytorch_to_tf_weights
from .preprocess.prepro_io import process_image, filter_class_only


load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")



"""===== Model Evaluation ====="""

checkpoint = torch.load("yolovoc_150_aug_epoch_80.pth", map_location='cpu')
print(" Converting PyTorch keys to PyTorch NAS keys...")
fixed_state_dict = fix_state_dict_keys(checkpoint['model_state_dict'])

print(" Converting PyTorch keys to Tensorflow NAS keys...")
tf_weights = convert_pytorch_to_tf_weights(fixed_state_dict)

with open("./mcunet_config.json", "r") as f:
    mcunet_config = json.load(f)


# detector = model.ObjectDetector("./yolovoc_150_aug_epoch_80.pkl", anchors, 0.5)
base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "colmain", "co3d", "car")

with open(os.path.join(base_path, "manifest_with_occ.json"), 'r') as f:
    manifest = json.load(f)

# Collect all image and ground truthes paths for both views
image_paths = []
all_gt = []

# Process pairs of images
for i in range(0, len(manifest) - 1): 
    item = manifest[i]
    filename = item['occ30']

    # Temporary fix 
    if filename.startswith('./'):
        # Remove './' and prepend base_path
        filename = os.path.join(base_path, filename[11:])

    # Check if files exist
    if os.path.exists(filename):
        image_paths.append(filename)
        gt = item['bbox'] + [VOC_CLASS_TO_IDX[INFERENCE_CLASS]]
        all_gt.append(gt)

print(f"Processing {len(image_paths)} image pairs")

# Create batches for bot views (5 images per batch)
all_images, resized_gt = process_image(image_paths, all_gt, image_size=160)


print("Getting Pytorch predictions for view 1 ...")
# mAP: 0.3517
# mAP@50: 0.7166
# hand-craft mAP@50: 0.7219012612809114
pytorch_model = ProxylessNASNets.build_from_config(mcunet_config)
pytorch_model.load_state_dict(fixed_state_dict)
pytorch_model.eval()
## with torch.no_grad():
    ## preds = pytorch_model(all_images) # [All images, A*(5+C), S, S]
    
# TensorFlow
# mAP: 0.3517
# mAP@50: 0.7166
# hand-craft mAP@50: 0.7219012612809114
# TFLite (2.5 MB)
# TFLite INT8 ()

print("Getting TensorFlow predictions for view 1 ...")
with TFObjectDetector(
    mcunet_config=mcunet_config,
    tf_weights=tf_weights,
    conf_threshold=0.01,
    image_size=160
) as tf_detector:
    # TensorFlow 
    ## preds = tf_detector.predict(all_images)
    # TFLite
    sample_input = all_images[:1]
    tflite_model = convert_tf_to_tflite_from_session(tf_detector, sample_input)


# print("Getting TLite (INT8) predictions for view 1 ...")
preds = run_tflite_inference("yolo_mcunet_model.tflite", all_images)

# Decode predictions for both views 
print("Decoding predictions...")
# [ 
#   [ # images
#       # detections [x1, x2, y1, y2, conf, classid]]] images, detections
decoded_preds = decode_pred(
    preds, anchors=VOC_ANCHORS, num_classes=len(VOC_CLASSES), 
    image_size=160, conf_thresh= 0.001
)

class_preds = filter_class_only(decoded_preds, VOC_CLASS_TO_IDX[INFERENCE_CLASS])
# Evaluate occ rate 1 
print("Evaluating ...")
results = evaluate_pred_gt(class_preds, resized_gt, device=None)
ap, recall, precision = evaluate_pred_gt_handcraft(class_preds, resized_gt, iou_threshold=0.5)


print(f"\n ======== Results ========")

print(f" mAP: {results['map']:.4f}")
print(f" mAP@50: {results['map_50']:.4f}")
print(f" hand-craft mAP@50: {ap}")

visualize_bbox_grid_tensors_single_view(
    all_images, class_preds, save_path=f"./nas_result/bbox_grid_single_view.png"
)
