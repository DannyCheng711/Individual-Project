import numpy as np
import tensorflow as tf
from .tf2_base_ops import *

# Create a Custom Detection Head Compatible with YoloClassifier
class McuYoloDetectionHeadTF:
    def __init__(self, _id, num_classes=20, num_anchors=5):
        self.id = _id
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_channels = num_anchors * (5 + num_classes) # 5 x 5 + 20 = 125

    # 	net = a context object (config) for the whole network 
    def build(self, _input, net, init=None):
        output = _input

        # control variable naming 
        with tf.compat.v1.variable_scope(self.id):
            # conv1: 320 -> 512 (creates variables: self.id/conv1/weight, self.id/conv1/bias)
            with tf.compat.v1.variable_scope('conv1'):
                output = conv2d(output, 512, 1, stride=1, param_initializer=init, use_bias=True)
                output = activation(output, 'relu6') # cap at 6

            # conv2: 512 -> 512 (creates variables: self.id/conv2/weight, self.id/conv2/bias)
            with tf.compat.v1.variable_scope('conv2'):
                output = conv2d(output, 512, 1, stride=1, param_initializer=init, use_bias=True)
                output = activation(output, 'relu6') # cap at 6

            yolo_head = YoloHeadTF('det_head', 512, self.num_classes, self.num_anchors)
            output = yolo_head.build(output, net, init)
            
        return output
    
    def build_with_intermediate(self, _input, net, init=None, intermediate_features=None):
        output = _input
        with tf.compat.v1.variable_scope(self.id):
            with tf.compat.v1.variable_scope('conv1'):
                output = conv2d(output, 512, 1, stride=1, param_initializer=init, use_bias=True)
                output = activation(output, 'relu6')
            with tf.compat.v1.variable_scope('conv2'):
                output = conv2d(output, 512, 1, stride=1, param_initializer=init, use_bias=True)
                output = activation(output, 'relu6')

            passthrough_feat = intermediate_features.get('passthrough')
            if passthrough_feat is not None:
                print("[TF] Conv3 is being used!")
                space_to_depth_layer = SpaceToDepthTF('space_to_depth', block_size=2)
                passthrough = space_to_depth_layer.build(passthrough_feat, net, init)
                output = tf.concat([output, passthrough], axis=3)
                with tf.compat.v1.variable_scope('conv3'):
                    output = conv2d(output, 512, 1, stride=1, param_initializer=init, use_bias=True)
                    output = activation(output, 'relu6')
            else:
                print("[TF] Conv3 is skipped!")
            yolo_head = YoloHeadTF('det_head', 512, self.num_classes, self.num_anchors)
            output = yolo_head.build(output, net, init)
        return output

    @property
    def config(self):
        return {
            'name': 'McuYoloDetectionHead',
            'num_classes': self.num_classes,
            'num_anchors': self.num_anchors,
        }
    
    @staticmethod
    def build_from_config(config, _id = 'layer1'):
        return McuYoloDetectionHeadTF(
            _id, 
            num_classes = config.get('num_classes', 20),
            num_anchors= config.get('num_anchors', 5)
        )

class SpaceToDepthTF:
    def __init__(self, _id, block_size=2):
        self.id = _id
        self.block_size = block_size
    
    def build(self, _input, net, init=None):
        # Input: [B, H, W, C] (TensorFlow format)
        # Output: [B, H//block_size, W//block_size, C*block_size*block_size]    
        with tf.compat.v1.variable_scope(self.id):
            # output = tf.nn.space_to_depth(_input, self.block_size)
            input_shape = tf.shape(_input)
            B = input_shape[0]
            H = input_shape[1] 
            W = input_shape[2]
            C = input_shape[3]
            
            block_size = self.block_size
            out_H = H // block_size
            out_W = W // block_size
            out_C = C * (block_size * block_size)
            
            # Convert TF format to PyTorch format: [B, H, W, C] -> [B, C, H, W]
            x_pt_format = tf.transpose(_input, [0, 3, 1, 2])  # [B, C, H, W]
            
            # Apply PyTorch space-to-depth logic:
            # Reshape: [B, C, H, W] -> [B, C, out_H, block_size, out_W, block_size]
            x_reshaped = tf.reshape(x_pt_format, [B, C, out_H, block_size, out_W, block_size])
            
            # Permute: [B, C, out_H, block_size, out_W, block_size] -> [B, C, block_size, block_size, out_H, out_W]
            x_permuted = tf.transpose(x_reshaped, [0, 1, 3, 5, 2, 4])  # Same as PyTorch permute(0, 1, 3, 5, 2, 4)
            
            # Final reshape: [B, C, block_size, block_size, out_H, out_W] -> [B, out_C, out_H, out_W]
            x_final_pt = tf.reshape(x_permuted, [B, out_C, out_H, out_W])
            
            # Convert back to TensorFlow format: [B, out_C, out_H, out_W] -> [B, out_H, out_W, out_C]
            output = tf.transpose(x_final_pt, [0, 2, 3, 1])
            
        return output


class YoloHeadTF:
    def __init__(self, _id, in_channels, num_classes, num_anchors):
        self.id = _id
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_channels = num_anchors * (5 + num_classes)
    
    def build(self, _input, net, init=None):
        output = _input
        with tf.compat.v1.variable_scope(self.id):
            output = conv2d(output, self.output_channels, 1, stride=1, param_initializer=init, use_bias=True)
        return output
        
