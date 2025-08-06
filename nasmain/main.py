import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # on cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import requests
import torch
import re
import numpy as np
import tensorflow as tf
import copy
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

        # Special handling for YOLO head
        # --- Handle YOLO detection head mapping ---
        if tf_name.startswith("classifier/layer1/det_head/detector"):
            tf_name = tf_name.replace("detector", "conv")
            print(tf_name)

        # --- Handle conv1, conv2, conv3 (Sequential -> conv scope) ---
        if any(tf_name.startswith(f"classifier/layer1/conv{i}") for i in [1, 2, 3]):
            tf_name = tf_name.replace("/0/", "/conv/")
            print(tf_name)

        if 'conv' in tf_name and 'weight' in tf_name:
            # Pytorch conv weight: [out_channels, in_channels, h, w]
            # TensorFlow conv weight: [h, w, in_channels, out_channels]
            if pytorch_name.endswith('depth_conv.conv.weight'):
                weight = weight.permute(2, 3, 0, 1)

            elif len(weight.shape) == 4:
                weight = weight.permute(2, 3, 1, 0)
        
        # remove pytorch's 0 layer numbering 
        if '/0/' in tf_name and not tf_name.startswith('blocks/'):
            # Remove /0/ from non-block layers (like conv1.0.weight -> conv1/weight)
            tf_name = tf_name.replace('/0/', '/')
        
        tf_weights[tf_name] = weight.detach().cpu().numpy()
    return tf_weights

"""===== Tensorflow Model Class ====="""
class TFObjectDetector:
    def __init__(self, mcunet_config, tf_weights, anchors, conf_threshold, image_size=160):
        self.anchors = anchors
        self.conf_threshold = conf_threshold
        self.image_size = image_size

        # Build TF model 
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session()

            # None: allows any batch size
            input_shape = [None, image_size, image_size, 3] # B, H, W, C
            self.tf_input_placeholder = tf.compat.v1.placeholder(
                name='input',
                dtype=tf.float32,
                shape=input_shape,
            )

            self.tf_model = ProxylessNASNetsTF(
                net_config=mcunet_config,
                net_weights=tf_weights,
                graph=self.graph,
                sess=self.sess,
                is_training= False,
                images=self.tf_input_placeholder,
                img_size=image_size,
                n_classes=20,
                S=5,
                A=5
            )

    def predict(self, images):
        """
        predict using TF model
        Args:
            images: torch.Tensor [B, 3, H, W]
        Returns:
            predictions: numpy array [B, 5, 5, 125]
        """
        # Convert pytorch tensor to numpy 
        if isinstance(images, torch.Tensor):
            tf_input = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
        else:
            tf_input = np.transpose(images, (0, 2, 3, 1))

        with self.graph.as_default():
            predictions = self.sess.run(self.tf_model.logits, feed_dict={
                self.tf_model.images: tf_input
            })
        
        return predictions
    
    def close(self):
        if hasattr(self, 'sess') and self.sess is not None:
            try:
                self.sess.close()
            except (AttributeError, RuntimeError, TypeError):
                pass
            finally:
                self.sess = None

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        # clean up TF session
        try:
            self.close()
        except:
            pass

""" ===== Setup Models ===== """

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

print(mcunet_config)

checkpoint = torch.load("yolovoc_150_aug_epoch_80.pth", map_location='cpu')
print(" Converting PyTorch keys to PyTorch NAS keys...")
fixed_state_dict = fix_state_dict_keys(checkpoint['model_state_dict'])

print(" Converting PyTorch keys to Tensorflow NAS keys...")
tf_weights = convert_pytorch_to_tf_weights(fixed_state_dict)

tf_weights_origin = copy.deepcopy(tf_weights)

print(f"===== Testing Pytorch Model =======")
pytorch_model = ProxylessNASNets.build_from_config(mcunet_config)
pytorch_model.eval()
pytorch_model.load_state_dict(fixed_state_dict)

# Test with dummy input
dummy_input = torch.randn(1, 3, 160, 160)

with torch.no_grad():
    pytorch_output = pytorch_model(dummy_input)
    print(f"Pytorch Output Shape: {pytorch_output.shape}")


print(f"===== Testing TensorFLow Model =======")
with TFObjectDetector(
    mcunet_config=mcunet_config,
    tf_weights=tf_weights,
    anchors = None,
    conf_threshold=0.001,
    image_size=160
) as tf_detector:
    

    tf_output = tf_detector.predict(dummy_input)
    print(f" TF Output Shape: {tf_output.shape}")

    pytorch_output = pytorch_output.permute(0, 2, 3, 1)
    pytorch_output_np = pytorch_output.detach().cpu().numpy()

    if pytorch_output_np.shape == tf_output.shape:
        print(f"\n🔍 Value Range Analysis:")
        print(f"PyTorch  range: [{pytorch_output_np.min():.3f}, {pytorch_output_np.max():.3f}]")
        print(f"TensorFlow range: [{tf_output.min():.3f}, {tf_output.max():.3f}]")
        
        diff = np.abs(pytorch_output_np - tf_output)
        print(f"\nMax difference: {np.max(diff):.6f}")


print(f"===== Compare Pytorch and TensorFLow Model (Backbone) =======")

def extract_block_features_pt(model, x):
    feats = {}
    with torch.no_grad():
        x = model.first_conv(x)
        for i, block in enumerate(model.blocks):
            block_feats = {}
            conv_input = x

            print(f"\n[PT Block {i}]")
            
            # ----- Inverted bottleneck -----
            if hasattr(block.mobile_inverted_conv, "inverted_bottleneck") and block.mobile_inverted_conv.inverted_bottleneck:
                ib = block.mobile_inverted_conv.inverted_bottleneck
                conv_out = ib[0](conv_input)
                block_feats['inverted_bottleneck_conv'] = conv_out.cpu().numpy()
                print(f"  inverted_bottleneck_conv: {conv_out.shape} range [{conv_out.min():.4f}, {conv_out.max():.4f}]")
                
                bn_out = ib[1](conv_out)
                block_feats['inverted_bottleneck_bn'] = bn_out.cpu().numpy()
                print(f"  inverted_bottleneck_bn: {bn_out.shape} range [{bn_out.min():.4f}, {bn_out.max():.4f}]")
                
                act_out = ib[2](bn_out)
                block_feats['inverted_bottleneck_relu6'] = act_out.cpu().numpy()
                print(f"  inverted_bottleneck_relu6: {act_out.shape} range [{act_out.min():.4f}, {act_out.max():.4f}]")
                
                conv_input = act_out  # feed to next layer

            # ----- Depthwise -----
            depthwise = block.mobile_inverted_conv.depth_conv
            conv_out = depthwise[0](conv_input)
            block_feats['depthwise_conv'] = conv_out.cpu().numpy()
            print(f"  depthwise_conv      : {conv_out.shape} range [{conv_out.min():.4f}, {conv_out.max():.4f}]")

            bn_out = depthwise[1](conv_out)
            block_feats['depthwise_bn'] = bn_out.cpu().numpy()
            print(f"  depthwise_bn        : {bn_out.shape} range [{bn_out.min():.4f}, {bn_out.max():.4f}]")

            act_out = depthwise[2](bn_out)
            block_feats['depthwise_relu6'] = act_out.cpu().numpy()
            print(f"  depthwise_relu6     : {act_out.shape} range [{act_out.min():.4f}, {act_out.max():.4f}]")

            # ----- Point Linear -----
            point_linear = block.mobile_inverted_conv.point_linear
            conv_out = point_linear[0](act_out)
            block_feats['point_linear_conv'] = conv_out.cpu().numpy()
            print(f"  point_linear_conv   : {conv_out.shape} range [{conv_out.min():.4f}, {conv_out.max():.4f}]")

            bn_out = point_linear[1](conv_out)
            block_feats['point_linear_bn'] = bn_out.cpu().numpy()
            print(f"  point_linear_bn     : {bn_out.shape} range [{bn_out.min():.4f}, {bn_out.max():.4f}]")

            feats[f'block_{i}'] = block_feats
            x = block(x)  # Run block fully for next iteration

    return feats

def find_bn_output_tensor(graph, bn_scope):
    # Find the final BN output tensor inside a batchnorm scope.
    candidates = []
    for op in graph.get_operations():
        if op.name.startswith(bn_scope) and 'batchnorm/add' in op.name:
            candidates.append(op.name)
    if not candidates:
        raise KeyError(f"No add op found in {bn_scope}")
    candidates.sort()  # ensures add, add_1, add_2 sorted (e.g. /bn/batch_normalization_52_1/batchnorm/add_1:0)
    return graph.get_tensor_by_name(candidates[-1] + ":0")  # pick the last one

def extract_block_features_tf(sess, input_placeholder, input_data, num_blocks=17):
    # Extract Conv, BN, ReLU outputs for each block in TF.
    graph = sess.graph
    tensors, names = [], []

    for block_idx in range(num_blocks):
        base = f"blocks/{block_idx}/mobile_inverted_conv"

        # Inverted bottleneck
        try:
            tensors.append(graph.get_tensor_by_name(f"{base}/inverted_bottleneck/conv/Conv2D:0"))
            names.append(f"block_{block_idx}/inverted_bottleneck/conv")
            tensors.append(find_bn_output_tensor(graph, f"{base}/inverted_bottleneck/bn"))
            names.append(f"block_{block_idx}/inverted_bottleneck/bn")
            tensors.append(graph.get_tensor_by_name(f"{base}/inverted_bottleneck/Relu6:0"))
            names.append(f"block_{block_idx}/inverted_bottleneck/relu6")
        except KeyError:
            pass

        # Depthwise
        tensors.append(graph.get_tensor_by_name(f"{base}/depth_conv/conv/depthwise:0"))
        names.append(f"block_{block_idx}/depthwise/conv")
        tensors.append(find_bn_output_tensor(graph, f"{base}/depth_conv/bn"))
        names.append(f"block_{block_idx}/depthwise/bn")
        tensors.append(graph.get_tensor_by_name(f"{base}/depth_conv/Relu6:0"))
        names.append(f"block_{block_idx}/depthwise/relu6")

        # Point linear
        tensors.append(graph.get_tensor_by_name(f"{base}/point_linear/conv/Conv2D:0"))
        names.append(f"block_{block_idx}/point_linear/conv")
        tensors.append(find_bn_output_tensor(graph, f"{base}/point_linear/bn"))
        names.append(f"block_{block_idx}/point_linear/bn")

        # Residual
        try:
            tensors.append(graph.get_tensor_by_name(f"blocks/{block_idx}/add:0"))
            names.append(f"block_{block_idx}/residual")
        except KeyError:
            pass

    outputs = sess.run(tensors, feed_dict={input_placeholder: input_data})

    # Organize & print
    feats = {}
    print("\n==== TensorFlow Block Features ====")
    for name, val in zip(names, outputs):
        parts = name.split('/')
        block = parts[0]
        layer_key = "_".join(parts[1:])  # flexible: handles 2+ parts
        if block not in feats:
            feats[block] = {}
            print(f"\n[{block}]")
        feats[block][layer_key] = val
        print(f"  {layer_key:<20}: {val.shape} range [{val.min():.4f}, {val.max():.4f}]")

    return feats

"""
with TFObjectDetector(
    mcunet_config=mcunet_config,
    tf_weights=tf_weights_origin,
    anchors = None,
    conf_threshold=0.001,
    image_size=160
) as tf_detector:
     
    # PyTorch features (check features output for each layer)
    pt_feats = extract_block_features_pt(pytorch_model, dummy_input)

    # TensorFlow features (check features output for each layer)
    tf_feats = extract_block_features_tf(tf_detector.sess, tf_detector.tf_model.images, dummy_input.permute(0,2,3,1).numpy(), num_blocks=17)
"""

print(f"===== Compare Pytorch and TensorFLow Model (YoloHead) =======")

def extract_yolo_head_features_pt(model, x):
    """Extract YOLO head input & raw conv outputs in PyTorch (for ProxylessNASNets)."""
    feats = {}
    with torch.no_grad():
        # Pass through stem & blocks
        x = model.first_conv(x)
        for block in model.blocks:
            x = block(x)
        backbone_out = x  # Final feature map before YOLO head
        feats['backbone_out'] = backbone_out.cpu().numpy()
        print(f"[PT YOLO] backbone_out: {backbone_out.shape} range [{backbone_out.min():.4f}, {backbone_out.max():.4f}]")

        # Feature mix (if exists)
        if model.feature_mix_layer:
            backbone_out = model.feature_mix_layer(backbone_out)

        # YOLO head (assumes conv1, conv2, det_head exist)
        yolo_head = model.classifier.layer1

        # Conv1
        conv1_out = yolo_head.conv1[0](backbone_out)  # first layer in Sequential
        feats["yolo_conv1"] = conv1_out.cpu().numpy()
        print(f"[PT YOLO] yolo_conv1: {conv1_out.shape} range [{conv1_out.min():.4f}, {conv1_out.max():.4f}]")
        conv1_relu = yolo_head.conv1[1](conv1_out)  # ReLU6
        feats["yolo_conv1_relu"] = conv1_relu.cpu().numpy()
        print(f"[PT YOLO] yolo_conv1_relu: {conv1_relu.shape} range [{conv1_relu.min():.4f}, {conv1_relu.max():.4f}]")

        # Conv2
        conv2_out = yolo_head.conv2[0](conv1_relu)
        feats["yolo_conv2"] = conv2_out.cpu().numpy()
        print(f"[PT YOLO] yolo_conv2: {conv2_out.shape} range [{conv2_out.min():.4f}, {conv2_out.max():.4f}]")
        conv2_relu = yolo_head.conv2[1](conv2_out)
        feats["yolo_conv2_relu"] = conv2_relu.cpu().numpy()
        print(f"[PT YOLO] yolo_conv2_relu: {conv2_relu.shape} range [{conv2_relu.min():.4f}, {conv2_relu.max():.4f}]")

        # Detection head
        det_head_out = yolo_head.det_head(conv2_relu)
        feats["yolo_det_head"] = det_head_out.cpu().numpy()
        print(f"[PT YOLO] yolo_det_head: {det_head_out.shape} range [{det_head_out.min():.4f}, {det_head_out.max():.4f}]")


    return feats

def extract_yolo_head_features_tf(sess, input_placeholder, input_data):
    """Extract YOLO head input & raw conv outputs in TensorFlow."""
    graph = sess.graph
    tensors, names = [], []

    # for op in tf_detector.graph.get_operations():
    #    if "classifier" in op.name:
    #        print(op.name)
    #    if "blocks/16/" in op.name:
    #        print(op.name)

    # Last backbone feature map (input to YOLO head)
    tensors.append(graph.get_tensor_by_name("blocks/16/mobile_inverted_conv/point_linear/bn/batch_normalization_101_1/batchnorm/add_1:0"))
    # tensors.append(graph.get_tensor_by_name("blocks/16/mobile_inverted_conv/point_linear/conv/Conv2D:0"))
    names.append("yolo_input")

    # YOLO head raw output
    tensors.append(graph.get_tensor_by_name("classifier/layer1/conv1/conv/Conv2D:0"))
    names.append("yolo_conv1")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/conv1/Relu6:0"))
    names.append("yolo_conv1_relu")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/conv2/conv/Conv2D:0"))
    names.append("yolo_conv2")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/conv2/Relu6:0"))
    names.append("yolo_conv2_relu")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/det_head/conv/Conv2D:0"))
    names.append("yolo_det_head")

    outputs = sess.run(tensors, feed_dict={input_placeholder: input_data})
    feats = {}

    print("\n==== TensorFlow YOLO Head Features ====")
    for name, val in zip(names, outputs):
        feats[name] = val
        print(f"{name}: {val.shape} range [{val.min():.4f}, {val.max():.4f}]")

    return feats


with TFObjectDetector(
    mcunet_config=mcunet_config,
    tf_weights=tf_weights_origin,
    anchors = None,
    conf_threshold=0.001,
    image_size=160
) as tf_detector:
    
    # PyTorch YOLO head
    dummy_input_1 = torch.randn(1, 3, 160, 160)
    pt_yolo_feats = extract_yolo_head_features_pt(pytorch_model, dummy_input_1)

    # TensorFlow YOLO head
    tf_yolo_feats = extract_yolo_head_features_tf(tf_detector.sess, tf_detector.tf_model.images, dummy_input_1.permute(0,2,3,1).numpy())
    
    
    yolo_head = pytorch_model.classifier.layer1

    for name, layer in [
        ("conv1", yolo_head.conv1[0]),
        ("conv2", yolo_head.conv2[0]),
        ("det_head", yolo_head.det_head.detector)
    ]:
        w = layer.weight.data.cpu().numpy()
        b = layer.bias.data.cpu().numpy()
        print(f"[PT {name}] weight mean={w.mean():.4f}, std={w.std():.4f}, bias mean={b.mean():.4f}")


    with tf_detector.sess.as_default():
        for name in ["conv1/conv/weight/Read/ReadVariableOp:0", 
                    "conv2/conv/weight/Read/ReadVariableOp:0", 
                    "det_head/conv/weight/Read/ReadVariableOp:0"]:
            var = tf_detector.sess.graph.get_tensor_by_name(f"classifier/layer1/{name}")
            val = var.eval()
            print(f"[TF {name}] weight mean={val.mean():.4f}, std={val.std():.4f}")
    

"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from config import DEVICE, VOC_CLASSES
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np 
import json 

import model 
from ..colmain.validation import decode_pred, evaluate_pred_gt, visualize_bbox_grid_tensors, evaluate_pred_gt_handcraft, plot_pr_curves_comp
from fusion import weighted_boxes_fusion

# 5 views construct a batch
# we have 20 objects, so we have 20 batches for each occlusion rate

curclass = "car"
curclassid = 6

def process_image(image_paths, ground_truths, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    processed_images = []
    resized_gts = []

    for img_path, gt in zip(image_paths, ground_truths):
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size

        image_tensor = transform(image)
        processed_images.append(image_tensor)

        x1, y1, x2, y2, class_id = gt
        scale_x = image_size / orig_width
        scale_y = image_size / orig_height

        resized_gt = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, class_id]
        resized_gts.append([resized_gt])

    all_images = torch.stack(processed_images).to(DEVICE)

    return all_images, resized_gts

# Filter predictions to only keep specific detections
def filter_class_only(decoded_preds, spec_class_id):
    class_only_preds = []

    for pred_list in decoded_preds:
        boxes = []
        for box in pred_list:
            xmin, ymin, xmax, ymax, conf, class_id = box
            if int(class_id) == spec_class_id:
                boxes.append(box)
        class_only_preds.append(boxes)
    
    return class_only_preds

anchors = torch.tensor([
        [8.851960381366869 / 32, 20.71491425034285 / 32], 
        [27.376232242192298 / 32, 56.73302518805578 / 32], 
        [42.88177452824786 / 32, 98.24243329589638 / 32], 
        [67.68032082718717 / 32, 132.7704493338952 / 32], 
        [131.16250016574756 / 32, 137.1476847579408 / 32]
    ], dtype=torch.float32)

detector = model.ObjectDetector("./yolovoc_150_aug_epoch_80.pkl", anchors, 0.5)
base_path = "..colmain/co3d/car"

with open(os.path.join(base_path, "manifest_with_occ.json"), 'r') as f:
    manifest = json.load(f)

# Collect all image and ground truthes paths for both views
image_paths = []
all_gt = []

# Process pairs of images
for i in range(0, len(manifest) - 1): 
    item = manifest[i]

    filename = item['image']
    
    # Check if files exist
    if os.path.exists(filename):
        image_paths.append(filename)
        gt = item['bbox'] + [curclassid]
        all_gt.append(gt)

    print(f"Processing {len(image_paths)} image pairs")
        
    # Create batches for bot views (5 images per batch)
    all_images, resized_gt = process_image(image_paths, all_gt, image_size=160)

    print("Getting predictions for view 1 ...")
    with torch.no_grad():
        preds = detector.model(all_images) # [All images, A*(5+C), S, S]

    # Decode predictions for both views 
    print("Decoding predictions...")
    # [ 
    #   [ # images
    #       # detections [x1, x2, y1, y2, conf, classid]]] images, detections
    decoded_preds = decode_pred(
        preds, anchors=anchors, num_classes=len(VOC_CLASSES), 
        image_size=160, conf_thresh= 0.001
    )

    class_preds = filter_class_only(decoded_preds, curclassid)

    # Evaluate occ rate 1 
    print("Evaluating Occlusion Rate 1 ...")
    results = evaluate_pred_gt(class_preds, resized_gt, device=None)
    ap, recall, precision = evaluate_pred_gt_handcraft(class_preds, resized_gt, iou_threshold=0.5)


    print(f"\n ======== Results ========")

    print(f" mAP: {results['map']:.4f}")
    print(f" mAP@50: {results['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ap}")

    visualize_bbox_grid_tensors(
        all_images, [],
        class_preds, [],
        [], [],
        save_path=f"./nas_result/bbox_grid.png"
    )

    
"""