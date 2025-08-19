import torch

def fix_state_dict_keys(checkpoint_state_dict):
    """Bridge: Convert PyTorch training keys to NAS config keys"""
    fixed_state_dict = {}
    skipped_keys = [] # classifier in backbone
    
    for key, value in checkpoint_state_dict.items():
        new_key = key
        skip_key = False

        if key.startswith('taps.'):
            key = key.replace('taps.', '')
            
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
        if tf_name.startswith("classifier/layer1/det_head"):
            tf_name = tf_name.replace("det_head", "det_head/conv")

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

