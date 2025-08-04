import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # on cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import requests
import torch
import re
import numpy as np
import tensorflow as tf
from config import DEVICE
from mcunet.mcunet.model_zoo import build_model
from mcunetYolo.tinynas.nn.proxyless_net import ProxylessNASNets
from mcunetYolo.tinynas.tf.tf2_proxyless_net import ProxylessNASNetsTF



""" ===== BRIDGE FUNCTION ===== """
def fix_state_dict_keys(checkpoint_state_dict):
    """Bridge: Convert PyTorch training keys to NAS config keys"""
    fixed_state_dict = {}
    skipped_keys = [] # classifier in backbone
    
    for key, value in checkpoint_state_dict.items():
        new_key = key
        skip_key = False
        
        # Fix backbone keys: backbone.xxx -> xxx
        if key.startswith('backbone.'):
            # skip backbone classifier
            if 'classifier.linear' in key:
                skip_key = True
                skipped_keys.append(key)
            else:
                new_key = key.replace('backbone.', '')
        
        # Fix detection head keys (Yolohead)
        elif key.startswith('conv1.'):
            new_key = key.replace('conv1.', 'classifier.layer1.conv1.')
        elif key.startswith('conv2.'):
            new_key = key.replace('conv2.', 'classifier.layer1.conv2.')
        elif key.startswith('conv3.'):
            new_key = key.replace('conv3.', 'classifier.layer1.conv3.')
        elif key.startswith('det_head.'):
            new_key = key.replace('det_head.', 'classifier.layer1.det_head.')
        elif key.startswith('space_to_depth.'):
            new_key = key.replace('space_to_depth.', 'classifier.layer1.space_to_depth.')
        
        if not skip_key:
            fixed_state_dict[new_key] = value
    
    print(f"Converted {len(fixed_state_dict)} keys, skipped {len(skipped_keys)} ImageNet keys")
    if skipped_keys:
        print("Skipped keys:")
        for key in skipped_keys:
            print(f"  {key}")
    
    return fixed_state_dict

# pytorch -> tensorflow weight conversion
def convert_pytorch_to_tf_weights(pytorch_state_dict):
    tf_weights = {}

    for pytorch_name, weight in pytorch_state_dict.items():
        # convert Pytorch name to TF variable name 
        tf_name = pytorch_name.replace('.', '/')

        if 'conv' in tf_name and 'weight' in tf_name:
            # Pytorch conv weight: [out_channels, in_channels, h, w]
            # TensorFlow conv weight: [h, w, in_channels, out_channels]
            if len(weight.shape) == 4:
                weight = weight.permute(2, 3, 1, 0)
        
        # remove pytorch's 0 layer numbering 
        if '/0/' in tf_name and not tf_name.startswith('blocks/'):
            # Remove /0/ from non-block layers (like conv1.0.weight -> conv1/weight)
            tf_name = tf_name.replace('/0/', '/')
        
        tf_weights[tf_name] = weight.detach().cpu().numpy()

    return tf_weights


""" ===== Test the pipeline ===== """
"""
# Load your model config
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Build the complete model
model = ProxylessNASNets.build_from_config(config)
model.eval()  # Set to evaluation mode

# Test with dummy input
dummy_input = torch.randn(1, 3, 416, 416)  # YOLO typical input size
print(f"Input shape: {dummy_input.shape}")

with torch.no_grad():
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Model built successfully!")

# Optional: Print model structure
print("\nModel structure:")
print(model)
"""


""" ===== Get the config of the backbone ===== """
backbone_fn, _, _ = build_model(net_id="mcunet-in4", pretrained=False)

# Step 2: Get the config from the backbone
mcunet_config = backbone_fn.config

# Step 3: Add your YOLO head configuration (same as training)
mcunet_config.update({
    "name": "mcunet_in4_yolo_trained",
    "classifier": {
        "name": "YOLOClassifier",
        "layer1": {
            "name": "McuYoloDetectionHead",
            "num_classes": 20,
            "num_anchors": 5
        },
        "layer2": None,
        "isConv": True,
        "need_intermediate_features": True
    }
})


""" ===== Test my model with pretrained weights (Pytorch) ===== """

# Build model with the correct architecture
model = ProxylessNASNets.build_from_config(mcunet_config)
model.eval()

checkpoint = torch.load("yolovoc_150_aug_epoch_80.pth", map_location='cpu')

print(" Converting PyTorch keys to PyTorch NAS keys...")
fixed_state_dict = fix_state_dict_keys(checkpoint['model_state_dict'])

model.load_state_dict(fixed_state_dict)
print(f" Model loaded! Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")


# Test with dummy input
dummy_input = torch.randn(1, 3, 160, 160)  # YOLO typical input size
print(f"Input shape: {dummy_input.shape}")

with torch.no_grad():
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


""" ===== Test my model with pretrained weights (Tensorflow) ===== """
print(f" Testing TensorFLow Model ...")
print(" Converting PyTorch NAS keys to TensorFlow NAS keys ...")

tf_weights = convert_pytorch_to_tf_weights(fixed_state_dict)

# Build model with the correct architecture
with tf.Graph().as_default() as graph:
    with tf.compat.v1.Session() as sess:
        input_shape = [1, 160, 160, 3]
        tf_input_placeholder = tf.compat.v1.placeholder(
            name = 'input', 
            dtype = tf.float32,
            shape = input_shape,
        )

        tf_model = ProxylessNASNetsTF(
            net_config=mcunet_config, 
            net_weights=tf_weights, 
            graph= graph,
            sess=sess,
            is_training= False,
            images=tf_input_placeholder,
            img_size=160,
            n_classes=20,
            S=5,
            A=5)

        tf_input = np.transpose(dummy_input.numpy(), (0, 2, 3, 1)) # NCHW -> NHWC for Tensorflow
        raw_predictions = tf_model.sess.run(tf_model.logits, feed_dict={
            tf_model.images: tf_input
        })

print(f"Raw output shape: {raw_predictions.shape}")  # [1, 5, 5, 125]
print(" TensorFlow model working correctly!")

