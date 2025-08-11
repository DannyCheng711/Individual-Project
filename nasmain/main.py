from PIL import Image
import numpy as np 
import json
import sys
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # on cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
import tensorflow as tf
import torch
import torch.optim as optim
from config import DEVICE, VOC_CLASSES, VOC_ROOT
from mcunetYolo.tinynas.nn.proxyless_net import ProxylessNASNets
from mcunetYolo.tinynas.tf.tf2_proxyless_net import ProxylessNASNetsTF

# Add parent directory to Python path for imports (temporary)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import model 
from colmain.validation import decode_pred, evaluate_pred_gt, visualize_bbox_grid_tensors_single_view, evaluate_pred_gt_handcraft, plot_pr_curves_comp
from vocmain._archived.__preprocess import YoloVocDataset
# 5 views construct a batch
# we have 20 objects, so we have 20 batches for each occlusion rate

curclass = "car"
curclassid = 6

anchors = torch.tensor([
        [8.851960381366869 / 32, 20.71491425034285 / 32], 
        [27.376232242192298 / 32, 56.73302518805578 / 32], 
        [42.88177452824786 / 32, 98.24243329589638 / 32], 
        [67.68032082718717 / 32, 132.7704493338952 / 32], 
        [131.16250016574756 / 32, 137.1476847579408 / 32]
    ], dtype=torch.float32)

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
        
        with self.graph.as_default():
            predictions = self.sess.run(self.tf_model.logits, feed_dict={
                self.tf_model.images: tf_input
            })
        
        # Temporary
        predictions_tensor = torch.from_numpy(predictions).float()
        predictions_tensor = predictions_tensor.permute(0, 3, 1, 2)
        # return predictions
        return predictions_tensor
    
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

        # --- Handle conv1, conv2, conv3 (Sequential -> conv scope) ---
        if any(tf_name.startswith(f"classifier/layer1/conv{i}") for i in [1, 2, 3]):
            tf_name = tf_name.replace("/0/", "/conv/")

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

    # all_images = torch.stack(processed_images).to(DEVICE)
    all_images = torch.stack(processed_images).to('cpu')

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


"""===== TFLite Model Methods ====="""
def convert_tf_to_tflite_from_session(tf_detector, sample_input):
    # Get TF session and model components 
    sess = tf_detector.sess
    graph = tf_detector.graph
    input_placeholder = tf_detector.tf_input_placeholder
    output_tensor = tf_detector.tf_model.logits

    if isinstance(sample_input, torch.Tensor):
        tf_input = sample_input.detach().cpu().numpy().transpose(0, 2, 3, 1)
    else:
        tf_input = np.transpose(sample_input, (0, 2, 3, 1))

    with graph.as_default():
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(
            sess = sess, 
            input_tensors = [input_placeholder],
            output_tensors = [output_tensor],
        )
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT] 

        # Quantization for even smaller models 
        print("🔧 Using VOC dataloader for quantization calibration...")
        converter.representative_dataset = lambda: representative_dataset_generator_voc(num_samples=1000)
        # Quantization settings
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Convert the model 
        try:
            tflite_model = converter.convert()
            
            tflite_path = "yolo_mcunet_model.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            
            print(f" TFLite Model Saved To: {tflite_path}")
            print(f" Model Size: {len(tflite_model) / 1024:.2f} KB")

            # test_tflite_model(tflite_path, tf_input)

            return tflite_model
        
        except Exception as e:
            print(f" TFLite conversion failed: {e}")
            return None


def test_tflite_model(tflite_path, tf_input):
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Get input and output tensors 
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("TFLite Model Details: ")
        print(f" Input shape: {input_details[0]['shape']}")
        print(f" Input dtype: {input_details[0]['dtype']}")
        print(f" Output shape: {output_details[0]['shape']}")
        print(f" Output dtype: {output_details[0]['dtype']}")

        # Prepare input data
        input_data = tf_input.astype(input_details[0]['dtype'])

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get output
        tflite_output = interpreter.get_tensor(output_details[0]['index'])

        print(f" TFLite inference successful!")
        print(f" Output shape: {tflite_output.shape}")
        print(f" Output range: [{tflite_output.min():.4f}, {tflite_output.max():.4f}]")
        
        return tflite_output
        
    except Exception as e:
        print(f" TFLite testing failed: {e}")
        return None

    
def representative_dataset_generator_voc(num_samples):
    train_voc_raw = VOCDetection(
        root=VOC_ROOT, year="2012", image_set="train", download=False)

    voc_dataset = YoloVocDataset(
        voc_dataset = train_voc_raw,
        image_size = 160,
        S = 5, 
        anchors = anchors, 
        num_classes = 20,
        aug = False
    )

    print(f" VOC dataset loaded with {len(voc_dataset)} training images")

    sample_indices = random.sample(
        range(len(voc_dataset)), num_samples)
    
    representative_data = []

    for idx in sample_indices:
        image, _ = voc_dataset[idx] # image is [3, 160, 160], target is [S, S, A, 5+C]
        # Convert Pytorch tensor to TF format: [3, H, W] -> [1, H, W, 3]
        tf_img = image.unsqueeze(0).numpy().transpose(0, 2, 3, 1) 
        representative_data.append(tf_img.astype(np.float32))
    
    print(f" Successfully processed {len(representative_data)} VOC representative images")

    if len(representative_data) == 0:
        raise Exception("No VOC images could be processed!")

    # yield each image for calibration one by one 
    for img in representative_data:
        yield [img]

def run_tflite_inference(tflite_path, input_images):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Convert input to TF format: [B, 3, H, W] -> [B, H, W, 3]
    if isinstance(input_images, torch.Tensor):
        tf_input = input_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    else:
        tf_input = np.transpose(input_images, (0, 2, 3, 1))

    # Quantize input if needed
    if input_details[0]['dtype'] == np.int8:
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        # Quantize: float32 -> int8
        tf_input = np.round(tf_input / scale + zero_point).astype(np.int8)


    all_outputs = []
    # Read single image as input sequentially 
    for i in range(tf_input.shape[0]):
        single_input = tf_input[i:i+1].astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], single_input)
        interpreter.invoke() 
        output = interpreter.get_tensor(output_details[0]['index'])
        if output_details[0]['dtype'] == np.int8:
            scale = output_details[0]['quantization'][0]
            zero_point = output_details[0]['quantization'][1]
            output = scale * (output.astype(np.float32) - zero_point)
        all_outputs.append(output)

    
    
    combined_output = np.concatenate(all_outputs, axis=0)
    # Convert back to Pytorch format for compatibility
    torch_output = torch.from_numpy(combined_output).float()
    torch_output = torch_output.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
    return torch_output


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
        gt = item['bbox'] + [curclassid]
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
    anchors = None,
    conf_threshold=0.001,
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
    preds, anchors=anchors, num_classes=len(VOC_CLASSES), 
    image_size=160, conf_thresh= 0.001
)

class_preds = filter_class_only(decoded_preds, curclassid)
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
