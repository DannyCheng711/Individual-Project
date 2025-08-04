import tensorflow as tf 
import numpy as np 

USE_TORCH_PADDING = True

def conv2d(
        _input,
        out_features,
        kernel_size,
        stride=1,
        padding='SAME',
        param_initializer=None, scope_name='conv', use_bias=False):
    in_features = int(_input.get_shape()[3])

    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.compat.v1.variable_scope(scope_name):
        init_key = '%s/weight' % tf.compat.v1.get_variable_scope().name

        initializer = param_initializer.get(
            init_key, tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0))
        weight = tf.compat.v1.get_variable(
            name='weight',
            shape=[
                kernel_size,
                kernel_size,
                in_features,
                out_features],
            initializer=initializer)

        assert padding == 'SAME' or use_bias

        if padding == 'SAME':
            if USE_TORCH_PADDING:
                pad = kernel_size // 2
                paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
                output = tf.pad(output, paddings, 'CONSTANT')

                output = tf.nn.conv2d(
                    output, filters=weight, strides=[
                        1, stride, stride, 1], padding='VALID', data_format='NHWC')
            else:
                output = tf.nn.conv2d(
                    output, filters=weight, strides=[
                        1, stride, stride, 1], padding='SAME', data_format='NHWC')
        else:
            output = tf.nn.conv2d(
                output, filters=weight, strides=[
                    1, stride, stride, 1], padding='VALID', data_format='NHWC')

        if use_bias:
            init_key = '%s/bias' % tf.compat.v1.get_variable_scope().name
            initializer = param_initializer.get(
                init_key, tf.compat.v1.constant_initializer(
                    [0.0] * out_features))
            bias = tf.compat.v1.get_variable(
                name='bias', shape=[out_features],
                initializer=initializer
            )
            output = output + bias
    return output


# Support customized padding setting
def conv2d_pad(
        _input,
        out_features,
        kernel_size,
        stride=1,
        pad=None,
        param_initializer=None, scope_name='conv', use_bias=False):
    in_features = int(_input.get_shape()[3])

    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.compat.v1.variable_scope(scope_name):
        init_key = '%s/weight' % tf.compat.v1.get_variable_scope().name

        initializer = param_initializer.get(
            init_key, tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0))
        weight = tf.compat.v1.get_variable(
            name='weight',
            shape=[
                kernel_size,
                kernel_size,
                in_features,
                out_features],
            initializer=initializer)
        
         # Check padding specification and apply accordingly
        if pad is not None:
            # User-specified padding, ignore the padding argument for convolution method
            paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
            output = tf.pad(output, paddings, 'CONSTANT')
            padding_used = 'VALID'  # Use 'VALID' to apply the user-specified padding only
        
        # Perform the convolution
        output = tf.nn.conv2d(
            output, filters=weight, strides=[1, stride, stride, 1], padding=padding_used, data_format='NHWC')

        if use_bias:
            init_key = '%s/bias' % tf.compat.v1.get_variable_scope().name
            initializer = param_initializer.get(
                init_key, tf.compat.v1.constant_initializer(
                    [0.0] * out_features))
            bias = tf.compat.v1.get_variable(
                name='bias', shape=[out_features],
                initializer=initializer
            )
            output = output + bias
    return output


def depthwise_conv2d(
        _input,
        kernel_size,
        stride=1,
        padding='SAME',
        param_initializer=None):
    in_features = int(_input.get_shape()[3])

    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.compat.v1.variable_scope('conv'):
        init_key = '%s/weight' % tf.compat.v1.get_variable_scope().name
        initializer = param_initializer.get(
            init_key, tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0))
        weight = tf.compat.v1.get_variable(
            name='weight', shape=[kernel_size, kernel_size, in_features, 1],
            initializer=initializer
        )
        assert padding == 'SAME'
        if USE_TORCH_PADDING:
            if padding == 'SAME':
                pad = kernel_size // 2
                paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
                output = tf.pad(output, paddings, 'CONSTANT')

            output = tf.nn.depthwise_conv2d(
                output, weight, [
                    1, stride, stride, 1], 'VALID', data_format='NHWC')
        else:
            output = tf.nn.depthwise_conv2d(
                output, weight, [
                    1, stride, stride, 1], 'SAME', data_format='NHWC')
    return output


def avg_pool(_input, k=2, s=2):
    padding = 'VALID'
    assert int(_input.get_shape()[1]) == k == s

    output = tf.nn.avg_pool2d(
        _input, ksize=[
            k, k], strides=[
            s, s], padding=padding, data_format='NHWC')
    return output


def fc_layer(_input, out_units, use_bias=False, param_initializer=None):
    features_total = int(_input.get_shape()[-1])
    if not param_initializer:
        param_initializer = {}
    with tf.compat.v1.variable_scope('linear'):
        init_key = '%s/weight' % tf.compat.v1.get_variable_scope().name
        initializer = param_initializer.get(
            init_key, tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        weight = tf.compat.v1.get_variable(
            name='weight', shape=[features_total, out_units],
            initializer=initializer
        )
        output = tf.matmul(_input, weight)
        if use_bias:
            init_key = '%s/bias' % tf.compat.v1.get_variable_scope().name
            initializer = param_initializer.get(
                init_key, tf.compat.v1.constant_initializer(
                    [0.0] * out_units))
            bias = tf.compat.v1.get_variable(
                name='bias', shape=[out_units],
                initializer=initializer
            )
            output = output + bias
    return output


def batch_norm(
        _input,
        is_training,
        epsilon=1e-3,
        decay=0.9,
        param_initializer=None):
    with tf.compat.v1.variable_scope('bn'):
        scope = tf.compat.v1.get_variable_scope().name
        if param_initializer is not None:
            bn_init = {
                'beta': param_initializer['%s/bias' % scope],
                'gamma': param_initializer['%s/weight' % scope],
                'moving_mean': param_initializer['%s/running_mean' % scope],
                'moving_variance': param_initializer['%s/running_var' % scope],
            }
        else:
            bn_init = None

        batch_norm_layer = tf.keras.layers.BatchNormalization(
            scale=True,
            beta_initializer=bn_init['beta'] if bn_init else 'zeros',
            gamma_initializer=bn_init['gamma'] if bn_init else 'ones',  # ← Fixed your typo
            moving_mean_initializer=bn_init['moving_mean'] if bn_init else 'zeros',
            moving_variance_initializer=bn_init['moving_variance'] if bn_init else 'ones',
            epsilon=epsilon,
            momentum=1-decay
        )
        output = batch_norm_layer(_input, training=is_training)
    return output


def activation(x, activation='relu6'):
    if activation == 'relu6':
       return tf.nn.relu6(x)
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(x)
    else:
        raise ValueError('Do not support %s' % activation)


def flatten(_input):
    input_shape = _input.shape.as_list()
    if len(input_shape) != 2:
        return tf.reshape(_input, [-1, np.prod(input_shape[1:])])
    else:
        return _input


def dropout(_input, keep_prob, is_training=False):
    if is_training:
        return tf.nn.dropout(_input, rate = 1.0 - keep_prob)
    else:
        return _input


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v