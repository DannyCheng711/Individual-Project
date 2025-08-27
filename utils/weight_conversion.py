import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # on cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np 
import random
from dotenv import load_dotenv
from torchvision.datasets import VOCDetection
import tensorflow as tf
import torch
from config import VOC_ANCHORS
from dataset.vocdatset import YoloVocDataset

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")

def get_dir_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

def convert_tf_to_tflite_from_session(tf_detector, tflite_path, image_size = None, grid_num = None):
    # Get TF session and model components 
    sess = tf_detector.sess
    graph = tf_detector.graph
    input_placeholder = tf_detector.tf_input_placeholder
    output_tensor = tf_detector.tf_model.logits

    with graph.as_default():
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(
            sess = sess, 
            input_tensors = [input_placeholder],
            output_tensors = [output_tensor],
        )
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT] 

        # Quantization for even smaller models 
        print(" Using VOC dataloader for quantization calibration...")
        converter.representative_dataset = lambda: representative_dataset_generator_voc(
            num_samples=1000, image_size=image_size, grid_num=grid_num)
        # Quantization settings
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Convert the model 
        try:
            tflite_model = converter.convert()
            tflite_model_path = os.path.join(tflite_path, "yolo_mcunet_model.tflite")
            with open(tflite_model_path, "wb") as f:
                f.write(tflite_model)
            
            print(f" TFLite Model Saved To: {tflite_model_path}")
            
            return tflite_model
        
        except Exception as e:
            print(f" TFLite conversion failed: {e}")
            return None

def representative_dataset_generator_voc(num_samples, image_size, grid_num):
    train_voc_raw = VOCDetection(
        root=VOC_ROOT, year="2012", image_set="train", download=False)

    voc_dataset = YoloVocDataset(
        voc_dataset = train_voc_raw,
        image_size = image_size,
        S = grid_num, 
        anchors = VOC_ANCHORS, 
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
            # De-Qantize
            output = scale * (output.astype(np.float32) - zero_point)
        all_outputs.append(output)

    combined_output = np.concatenate(all_outputs, axis=0)
    # Convert back to Pytorch format for compatibility
    torch_output = torch.from_numpy(combined_output).float()
    torch_output = torch_output.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
    return torch_output