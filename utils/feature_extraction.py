import torch 
import numpy as np

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

def extract_yolo_head_features_pt(model, x):
    """Extract YOLO head input & raw conv outputs in PyTorch (for ProxylessNASNets)."""
    feats = {}
    with torch.no_grad():
        # Pass through stem & blocks
        x = model.first_conv(x)
        for i, block in enumerate(model.blocks):
            x = block(x)

            if i == 12:
                raw_passthrough = x.clone()
                feats["yolo_passthrough_block"] = x.clone().cpu().numpy()
                print(f"[PT YOLO] passthrough_block: {x.shape} range [{x.min():.4f}, {x.max():.4f}]")

                
        backbone_out = x  # Final feature map before YOLO head
        feats['yolo_input'] = backbone_out.cpu().numpy()
        print(f"[PT YOLO] yolo_input: {backbone_out.shape} range [{backbone_out.min():.4f}, {backbone_out.max():.4f}]")
        
        # Feature mix (if exists)
        if model.feature_mix_layer:
            backbone_out = model.feature_mix_layer(backbone_out)

        # YOLO head (assumes conv1, conv2, dethead exist)
        yolo_head = model.classifier.layer1

        # Conv1
        conv1_out = yolo_head.conv1[0](backbone_out)  # first layer in Sequential
        feats["yolo_conv1"] = conv1_out.clone().cpu().numpy() 
        print(f"[PT YOLO] yolo_conv1: {conv1_out.shape} range [{conv1_out.min():.4f}, {conv1_out.max():.4f}]")
        conv1_relu = yolo_head.conv1[1](conv1_out)  # ReLU6
        feats["yolo_conv1_relu"] = conv1_relu.clone().cpu().numpy()
        print(f"[PT YOLO] yolo_conv1_relu: {conv1_relu.shape} range [{conv1_relu.min():.4f}, {conv1_relu.max():.4f}]")

        # Conv2
        conv2_out = yolo_head.conv2[0](conv1_relu)
        feats["yolo_conv2"] = conv2_out.clone().cpu().numpy()
        print(f"[PT YOLO] yolo_conv2: {conv2_out.shape} range [{conv2_out.min():.4f}, {conv2_out.max():.4f}]")
        conv2_relu = yolo_head.conv2[1](conv2_out)
        feats["yolo_conv2_relu"] = conv2_relu.clone().cpu().numpy()
        print(f"[PT YOLO] yolo_conv2_relu: {conv2_relu.shape} range [{conv2_relu.min():.4f}, {conv2_relu.max():.4f}]")

        
        # Apply space-to-depth transformation manually
        if hasattr(yolo_head, 'space_to_depth'):
            # Use the actual layer
            space_to_depth_result = yolo_head.space_to_depth(raw_passthrough)
            feats["yolo_space_to_depth"] = space_to_depth_result.clone().cpu().numpy()
            print(f"[PT YOLO] space_to_depth: {space_to_depth_result.shape} range [{space_to_depth_result.min():.4f}, {space_to_depth_result.max():.4f}]")
            
            # Use the layer result for concatenation
            pt_concat = torch.cat([conv2_relu, space_to_depth_result], dim=1)
            feats["yolo_concat"] = pt_concat.clone().cpu().numpy()
            print(f"[PT YOLO] pt_concat: {pt_concat.shape} range [{pt_concat.min():.4f}, {pt_concat.max():.4f}]")

        # Conv3
        conv3_out = yolo_head.conv3[0](pt_concat)
        feats["yolo_conv3"] = conv3_out.clone().cpu().numpy()
        print(f"[PT YOLO] yolo_conv3: {conv3_out.shape} range [{conv3_out.min():.4f}, {conv3_out.max():.4f}]")
        conv3_relu = yolo_head.conv3[1](conv3_out)
        feats["yolo_conv3_relu"] = conv3_relu.clone().cpu().numpy()
        print(f"[PT YOLO] yolo_conv3_relu: {conv3_relu.shape} range [{conv3_relu.min():.4f}, {conv3_relu.max():.4f}]")

        # Detection head
        dethead_out = yolo_head.det_head(conv3_relu)
        feats["yolo_dethead"] = dethead_out.cpu().numpy()
        print(f"[PT YOLO] yolo_dethead: {dethead_out.shape} range [{dethead_out.min():.4f}, {dethead_out.max():.4f}]")


    return feats

def extract_yolo_head_features_tf(sess, input_placeholder, input_data):
    """Extract YOLO head input & raw conv outputs in TensorFlow."""
    graph = sess.graph
    tensors, names = [], []

    # Last backbone feature map (input to YOLO head)
    tensors.append(graph.get_tensor_by_name("blocks/16/mobile_inverted_conv/point_linear/bn/batch_normalization_50_1/batchnorm/add_1:0"))
    # tensors.append(graph.get_tensor_by_name("blocks/16/mobile_inverted_conv/point_linear/conv/Conv2D:0"))
    names.append("yolo_input")

    tensors.append(graph.get_tensor_by_name("blocks/12/add:0"))
    names.append("yolo_passthrough_block")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/space_to_depth/SpaceToDepth:0"))
    names.append("yolo_space_to_depth")

    tensors.append(graph.get_tensor_by_name("classifier/layer1/concat:0"))
    names.append("yolo_concat")
    
    # YOLO head raw output
    tensors.append(graph.get_tensor_by_name("classifier/layer1/conv1/conv/add:0"))
    names.append("yolo_conv1")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/conv1/Relu6:0"))
    names.append("yolo_conv1_relu")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/conv2/conv/add:0"))
    names.append("yolo_conv2")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/conv2/Relu6:0"))
    names.append("yolo_conv2_relu")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/conv3/conv/add:0"))
    names.append("yolo_conv3")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/conv3/Relu6:0"))
    names.append("yolo_conv3_relu")
    tensors.append(graph.get_tensor_by_name("classifier/layer1/det_head/conv/add:0"))
    names.append("yolo_dethead")

    outputs = sess.run(tensors, feed_dict={input_placeholder: input_data})
    
    feats = {}

    print("\n==== TensorFlow YOLO Head Features ====")
    for name, val in zip(names, outputs):
        feats[name] = val
        print(f"{name}: {val.shape} range [{val.min():.4f}, {val.max():.4f}]")

    return feats
    
