import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # on cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json, sys
from dotenv import load_dotenv
import tensorflow as tf
import torch

# Add parent directory to Python path for imports (temporary using colmain.validation)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import model 
from config import VOC_ANCHORS, VOC_CLASSES, INFERENCE_CLASS, VOC_CLASS_TO_IDX
from colmain.validation import evaluate_pred_gt, visualize_bbox_grid_tensors_single_view, evaluate_pred_gt_handcraft, plot_pr_curves_comp
from validation.bboxpreprocess import decode_pred
from models.mcunetYolo.tinynas.tf.tf2_yolodet import TFObjectDetector
from utils.config_utils import fix_state_dict_keys, convert_pytorch_to_tf_weights
from utils.image_utils import process_image, filter_class_only
from utils.weight_conversion import convert_tf_to_tflite_from_session, run_tflite_inference


load_dotenv()  # Loads .env from current directory

DEVICE = torch.device("cuda")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")


"""===== Model Evaluation ====="""

with open("./nasmain/model_config/mcunetYolo_config.json", "r") as f:
    mcunetYolo_config = json.load(f)
checkpoint = torch.load("./runs/mcunet_S5_res160_pkg_s2d/best.pth", map_location='cpu')

print(" Converting PyTorch keys to Tensorflow NAS keys...")
fixed_state_dict = fix_state_dict_keys(checkpoint['model'])
tf_weights = convert_pytorch_to_tf_weights(fixed_state_dict)


"======= Prepare CO3D Data ======="

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


"======= Read Model Config ======="
with open("./nasmain/model_config/mcunetYolo_config.json", "r") as f:
    mcunetYolo_config = json.load(f)

checkpoint = torch.load("./runs/mcunet_S5_res160_pkg_lrdecay/best.pth", map_location='cpu')
print(" Converting PyTorch keys to PyTorch NAS keys...")
fixed_state_dict = fix_state_dict_keys(checkpoint['model'])
tf_weights = convert_pytorch_to_tf_weights(fixed_state_dict)

print("Converting TF model to TFLite...")
with TFObjectDetector(
    mcunet_config=mcunetYolo_config,
    tf_weights=tf_weights,
    conf_threshold=0.01,
    image_size=160 # remember to change
) as tf_detector:
    # TensorFlow 
    # TFLite
    tflite_model = convert_tf_to_tflite_from_session(
        tf_detector, image_size=160, grid_num=5)


# print("Getting TLite (INT8) predictions ...")
preds = run_tflite_inference("./yolo_mcunet_model.tflite", all_images)

# Decode predictions for both views 
print("Decoding predictions...")
# [ 
#   [ # images
#       # detections [x1, x2, y1, y2, conf, classid]]] images, detections
decoded_preds = decode_pred(
    preds, anchors=torch.tensor(VOC_ANCHORS, dtype=torch.float32), num_classes=len(VOC_CLASSES), 
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
    all_images, class_preds, save_path=f"./nasmain/nas_result/bbox_grid_single_view.png"
)
