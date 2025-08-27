import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # on cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import torch
from dotenv import load_dotenv
from config import VOC_ANCHORS, VOC_CLASSES, INFERENCE_CLASS, VOC_CLASS_TO_IDX
from models.mcunetYolo.tinynas.nn.proxyless_net import ProxylessNASNets
from models.mcunetYolo.tinynas.tf.tf2_yolodet import TFObjectDetector
from utils.config_utils import fix_state_dict_keys, convert_pytorch_to_tf_weights
from utils.weight_conversion import convert_tf_to_tflite_from_session
from validation.evaluator import Evaluator
from torchvision.datasets import VOCDetection

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device("cuda")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")

"======= Read Model Config ======="
with open("./nasmain/model_config/mcunetYolo_config.json", "r") as f:
    mcunetYolo_config = json.load(f)

checkpoint = torch.load("./runs/mcunet_S5_res160_pkg_lrdecay/best.pth", map_location='cpu')
fixed_state_dict = fix_state_dict_keys(checkpoint['model'])

print("Estimating Pytorch FP32 model size...")
pytorch_model = ProxylessNASNets.build_from_config(mcunetYolo_config)
pytorch_model.eval()
pytorch_model.load_state_dict(fixed_state_dict)
total_params = sum(p.numel() for p in pytorch_model.parameters())
fp32_model_size_bytes = total_params * 4  # 4 bytes per float32


tf_weights = convert_pytorch_to_tf_weights(fixed_state_dict)


print("Converting TF model to TFLite...")
with TFObjectDetector(
    mcunet_config=mcunetYolo_config,
    tf_weights=tf_weights,
    conf_threshold=0.01,
    image_size=160 # remember to change
) as tf_detector:
    # TensorFlow, TFLite
    tflite_model = convert_tf_to_tflite_from_session(
        tf_detector, tflite_path= "./runs/mcunet_S5_res160_pkg_lrdecay/tinyml" ,image_size=160, grid_num=5)

print("Getting TLite (INT8) predictions ...")

val_voc_raw = VOCDetection(root=VOC_ROOT, year="2012", image_set="val", download=False)
    
evaluator = Evaluator(
    val_voc_raw,
    torch.tensor(VOC_ANCHORS, dtype=torch.float32), 
    len(VOC_CLASS_TO_IDX), 
    image_size=160,
    grid_num=5,
    batch_size=16, 
    epoch_num=160,
    save_dir="./runs/mcunet_S5_res160_pkg_lrdecay/tinyml/",
    pkg=True
)

results = evaluator.evaluate_tiny("./yolo_mcunet_model.tflite")

print(f"\n ======== Results ========")
print(f" mAP: {results['full']['map']:.4f}")
print(f" mAP@50: {results['map50']:.4f}")
print(f" Estimated FP32 model size: {fp32_model_size_bytes / 1024:.2f} KB ({fp32_model_size_bytes / (1024*1024):.2f} MB)")
print(f" Estimated INT8 model size: {len(tflite_model) / 1024:.2f} KB ({len(tflite_model) / (1024*1024):.2f} MB)")

