import numpy as np
import tensorflow as tf
from .tf2_base_ops import *
from .tf2_customised_layers import McuYoloDetectionHeadTF

class MBInvertedConvLayer:
    def __init__(
            self,
            _id,
            filter_num,
            kernel_size=3,
            stride=1,
            expand_ratio=6):
        self.id = _id
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio

    def build(self, _input, net, init=None):
        output = _input
        in_features = int(_input.get_shape()[3])
        with tf.compat.v1.variable_scope(self.id):
            if self.expand_ratio > 1:
                feature_dim = round(in_features * self.expand_ratio)

                with tf.compat.v1.variable_scope('inverted_bottleneck'):
                    output = conv2d(
                        output, feature_dim, 1, 1, param_initializer=init)
                    output = batch_norm(
                        output,
                        net.is_training,
                        epsilon=net.bn_eps,
                        decay=net.bn_decay,
                        param_initializer=init)
                    output = activation(output, 'relu6')
            with tf.compat.v1.variable_scope('depth_conv'):
                output = depthwise_conv2d(
                    output,
                    self.kernel_size,
                    self.stride,
                    param_initializer=init)
                output = batch_norm(
                    output,
                    net.is_training,
                    epsilon=net.bn_eps,
                    decay=net.bn_decay,
                    param_initializer=init)
                output = activation(output, 'relu6')
            with tf.compat.v1.variable_scope('point_linear'):
                output = conv2d(output, self.filter_num, 1,
                                1, param_initializer=init)
                output = batch_norm(
                    output,
                    net.is_training,
                    epsilon=net.bn_eps,
                    decay=net.bn_decay,
                    param_initializer=init)
        return output


class ConvLayer:
    def __init__(self, _id, filter_num, kernel_size=3, stride=1):
        self.id = _id
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride

    def build(self, _input, net, init=None):
        output = _input
        with tf.compat.v1.variable_scope(self.id):
            output = conv2d(
                output,
                self.filter_num,
                self.kernel_size,
                self.stride,
                param_initializer=init)

            output = batch_norm(
                output,
                net.is_training,
                epsilon=net.bn_eps,
                decay=net.bn_decay,
                param_initializer=init)

            output = activation(output, 'relu6')
        return output


# Support customized padding in Conv layer
class ConvLayerPad:
    def __init__(self, _id, filter_num, kernel_size=3, padding=1, stride=1):
        self.id = _id
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def build(self, _input, net, init=None):
        output = _input
        with tf.compat.v1.variable_scope(self.id):
            output = conv2d_pad(
                output,
                self.filter_num,
                self.kernel_size,
                self.stride,
                self.padding,
                param_initializer=init)

            output = batch_norm(
                output,
                net.is_training,
                epsilon=net.bn_eps,
                decay=net.bn_decay,
                param_initializer=init)

            output = activation(output, 'sigmoid')
        return output



class ConvLayer_fc:
    def __init__(self, _id, filter_num, kernel_size=3, stride=1):
        self.id = _id
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride

    def build(self, _input, net, init=None):
        output = _input
        with tf.compat.v1.variable_scope(self.id):
            output = conv2d(
                output,
                self.filter_num,
                self.kernel_size,
                self.stride,
                padding='VALID',
                param_initializer=init, scope_name='linear', use_bias=True)
        return output


class LinearLayer:
    def __init__(self, _id, n_units, drop_rate=0):
        self.id = _id
        self.n_units = n_units
        self.drop_rate = drop_rate

    def build(self, _input, net, init=None):
        output = _input
        with tf.compat.v1.variable_scope(self.id):
            if self.drop_rate > 0:
                output = dropout(output, 1 - self.drop_rate, net.is_training)
            output = fc_layer(
                output,
                self.n_units,
                use_bias=True,
                param_initializer=init)
        return output


TF_LAYER_REGISTRY = {
        'MBInvertedConvLayer': MBInvertedConvLayer,
        'ConvLayer': ConvLayer,
        'ConvLayerPad': ConvLayerPad,
        'ConvLayer_fc': ConvLayer_fc,
        'LinearLayer': LinearLayer,
        'McuYoloDetectionHead': McuYoloDetectionHeadTF,
}

def build_layer_from_config(config, _id):

    layer_name = config['name']
    if layer_name in TF_LAYER_REGISTRY:
        layer_class = TF_LAYER_REGISTRY[layer_name]
        return layer_class.build_from_config(config, _id)
    else:
        raise ValueError(f"Unknown layer type: {layer_name}")