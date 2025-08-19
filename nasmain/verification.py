import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # on cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np 
import json
import tensorflow as tf
import torch

# import model 
from config import VOC_ANCHORS, VOC_CLASSES, INFERENCE_CLASS, VOC_CLASS_TO_IDX
from models.mcunetYolo.tinynas.nn.proxyless_net import ProxylessNASNets
from models.mcunetYolo.tinynas.tf.tf2_yolodet import TFObjectDetector
from utils.config_utils import fix_state_dict_keys, convert_pytorch_to_tf_weights
from utils.feature_extraction import extract_yolo_head_features_tf, extract_yolo_head_features_pt, find_bn_output_tensor


""" ===== Setup Models ===== """

# backbone_fn, _, _ = build_model(net_id="mcunet-in4", pretrained=False)
# mcunet_config = backbone_fn.config
# mcunet_config.update({
#     "name": "mcunet_in4_yolo_trained",
#     "classifier": {
#         "name": "YOLOClassifier",
#         "layer1": {
#             "name": "McuYoloDetectionHead",
#             "num_classes": 20,
#             "num_anchors": 5
#         },
#         "layer2": None,
#         "isConv": True,
#         "need_intermediate_features": True
#     }
# })

with open("./nasmain/model_config/mcunetYolo_config.json", "r") as f:
    mcunetYolo_config = json.load(f)

checkpoint = torch.load("./runs/mcunet_S5_res160_pkg_s2d/best.pth", map_location='cpu')
print(" Converting PyTorch keys to PyTorch NAS keys...")

fixed_state_dict = fix_state_dict_keys(checkpoint['model'])

print(f"===== Testing Pytorch Model =======")
# PyTorch YOLO head
dummy_input = torch.randn(1, 3, 160, 160)
pytorch_model = ProxylessNASNets.build_from_config(mcunetYolo_config)
pytorch_model.eval()
pytorch_model.load_state_dict(fixed_state_dict)
with torch.no_grad():
    pytorch_output = pytorch_model(dummy_input)
    # Extract feature maps, weights and bias 
    pt_yolo_feats = extract_yolo_head_features_pt(pytorch_model, dummy_input)
    pt_weight = pytorch_model.classifier.layer1.conv1[0].weight.detach().cpu().numpy()
    pt_bias = pytorch_model.classifier.layer1.conv1[0].bias.detach().cpu().numpy()
    # Convert to TensorFlow format: [H, W, in_channels, out_channels]
    pytorch_output_tf_format = pytorch_output.permute(0, 2, 3, 1).detach().cpu().numpy()
    pt_weight_tf_format = np.transpose(pt_weight, (2, 3, 1, 0)) 

print(f"PyTorch output: {pytorch_output.shape}, range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")

tf_weights = convert_pytorch_to_tf_weights(fixed_state_dict)

# print(f"===== Compare Pytorch and TensorFLow Model (Backbone) =======")

# with TFObjectDetector(
#     mcunet_config=mcunet_config,
#     tf_weights=tf_weights_origin,
#     anchors = None,
#     conf_threshold=0.001,
#     image_size=160
# ) as tf_detector:
     
#     # PyTorch features (check features output for each layer)
#     pt_feats = extract_block_features_pt(pytorch_model, dummy_input)

#     # TensorFlow features (check features output for each layer)
#     tf_feats = extract_block_features_tf(tf_detector.sess, tf_detector.tf_model.images, dummy_input.permute(0,2,3,1).numpy(), num_blocks=17)


print(f"===== Compare Pytorch and TensorFLow Model (YoloHead) =======")

with TFObjectDetector(
    mcunet_config=mcunetYolo_config,
    tf_weights=tf_weights,
    conf_threshold=0.01,
    image_size=160
) as tf_detector:
    
    # get operator name
    # for op in tf_detector.graph.get_operations():
    #   if "classifier" in op.name:
    #       print(op.name)
    #   if "blocks/12/" in op.name:
    #       print(op.name)
    
    # TensorFlow YOLO head
    tf_output = tf_detector.predict(dummy_input)

    # Extract feature maps, weights and bias 
    tf_yolo_feats= extract_yolo_head_features_tf(
        tf_detector.sess, 
        tf_detector.tf_model.images, 
        dummy_input.permute(0,2,3,1).numpy()
    )
    graph = tf_detector.sess.graph
    tf_weight = tf_detector.sess.run(graph.get_tensor_by_name(
        'classifier/layer1/conv1/conv/weight/Read/ReadVariableOp:0'
    ))
    tf_bias = tf_detector.sess.run(graph.get_tensor_by_name(
        'classifier/layer1/conv1/conv/bias/Read/ReadVariableOp:0'
    ))

    
print(f"TensorFlow real output: {tf_output.shape}, range: [{tf_output.min():.6f}, {tf_output.max():.6f}]")

# Test weight alignment 
if pt_weight_tf_format.shape == tf_weight.shape:
    diff = np.abs(pt_weight_tf_format - tf_weight)
    print("Weight diff (conv1):", np.max(diff))
    print("Mean diff (conv1):", np.mean(diff))

# Test output alignment
print(pytorch_output_tf_format.shape)
print(tf_output.shape)
if pytorch_output_tf_format.shape == tf_output.shape:
    
    real_diff = np.abs(pytorch_output_tf_format - tf_output).max()

    # Compare specific YOLO head layers
    for layer_name in ['yolo_input', 'yolo_conv1', 'yolo_conv2', 'yolo_passthrough_block', 'yolo_space_to_depth', 'yolo_concat', 'yolo_conv3']:
        if layer_name in pt_yolo_feats and layer_name in tf_yolo_feats:
            pt_feat = pt_yolo_feats[layer_name]
            tf_feat = tf_yolo_feats[layer_name].transpose(0, 3, 1, 2)  # Convert to PyTorch format
            
            if pt_feat.shape == tf_feat.shape:

                layer_diff = np.abs(pt_feat - tf_feat)
                print(f"{layer_name} difference: {layer_diff.max():.6f}")
              
    print(f"Real data max difference: {real_diff:.6f}")