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
    def build(self, _input, net, init=None, intermediate_features = None):
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

            # handle intermediate features if available (space_to_depth + concatenation)
            if intermediate_features is not None:
                passthrough_feat = intermediate_features.get('passthrough')
                if passthrough_feat is not None:
                    # apply space_to_depth to passthrough features
                    space_to_depth_layer = SpaceToDepthTF('space_to_depth', block_size=2)
                    passthrough = space_to_depth_layer.build(passthrough_feat, net, init)

                    # concatenate along channel dimension (NHWC)
                    output = tf.concat([output, passthrough], axis = 3)

                    # conv3: 512+384 -> 512 (creates variables: self.id/conv3/weight, self.id/conv3/bias)
                    with tf.compat.v1.variable_scope('conv3'):
                        output = conv2d(output, 512, 1, stride=1, param_initializer=init, use_bias=True)
                        output = activation(output, 'relu6') # cap at 6

            # det_head: 512 -> 125 (creates variables: self.id/det_head/detector/weight, self.id/det_head/detector/bias)            
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
        with tf.compat.v1.variable_scope(self.id):
            output = tf.nn.space_to_depth(_input, self.block_size)
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
        
