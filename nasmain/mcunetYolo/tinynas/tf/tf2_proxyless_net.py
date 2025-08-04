from .tf2_layers import *
from .tf2_layers import build_layer_from_config

class MobileInvertedResidualBlock:

    def __init__(self, _id, mobile_inverted_conv, has_residual):
        self.id = _id
        self.mobile_inverted_conv = mobile_inverted_conv
        self.has_residual = has_residual

    def build(self, _input, net, init=None):
        output = _input
        with tf.compat.v1.variable_scope(self.id):
            output = self.mobile_inverted_conv.build(output, net, init)
            if self.has_residual:
                output = output + _input
        return output


class YOLOClassifier:
    def __init__(self, _id, S, layer1, layer2, isConv, A):
        self.id = _id
        self.S = S
        self.A = A
        self.layer1 = layer1
        self.layer2 = layer2
        self.isConv = isConv

    def build(self, _input, net, init=None, is_training=False):
        output = _input
        with tf.compat.v1.variable_scope(self.id):
            output = self.layer1.build(output, net, init)

            if self.layer2 is not None:
                output = self.layer2.build(output, net, init)
            
            # if self.isConv:
                # output = output.permute(0, 2, 3, 1)
                # output = tf.transpose(output, perm=[0, 2, 3, 1])
            # if is_training:
            #   return output
            # else:
            #     detection_boxes, detection_classes, detection_scores, num_boxes = self._decoder_tf(output, grid_num=self.S)
            #     return detection_boxes, detection_classes, detection_scores, num_boxes
    
            return output # Shape:[B S S A * (5 + C)]
    """
    @tf.function
    def _decoder_tf(self, prediction, grid_num=7):
        cell_size = 1. / grid_num
        boxes = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        cls_indexes = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        confidences = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        prediction = tf.convert_to_tensor(prediction, dtype=tf.float32)
        prediction = tf.squeeze(prediction)  # Assume prediction is of shape [S, S, (5*B+num_classes)]
        contain1 = tf.expand_dims(prediction[:, :, 4], -1)
        contain2 = tf.expand_dims(prediction[:, :, 9], -1)
        contain = tf.concat([contain1, contain2], axis=-1)

        mask1 = contain > 0.1
        mask2 = tf.equal(contain, tf.reduce_max(contain))
        mask = tf.math.logical_or(mask1, mask2)

        count = 0
        for i in tf.range(grid_num):
            for j in tf.range(grid_num):
                for b in tf.range(2):
                    if mask[i, j, b]:
                        # Extract the bounding box and the confidence
                        start_idx = b * 5
                        box = prediction[i, j, start_idx:start_idx + 4]
                        contain_prob = prediction[i, j, start_idx + 4]

                        # Compute the top-left corner of the cell
                        xy = tf.stack([tf.cast(j, tf.float32), tf.cast(i, tf.float32)]) * cell_size

                        # Create a new tensor with the updated rows and the rest of the original 'box'
                        box = tf.concat([box[:2] * cell_size + xy, box[2:]], axis=0)
                        adjusted_box = tf.concat([box[:2] - 0.5 * box[2:], box[:2] + 0.5 * box[2:]], axis=0)

                        # Get class probabilities and the class index
                        class_probs = prediction[i, j, 10:]
                        max_prob = tf.reduce_max(class_probs)
                        cls_index = tf.argmax(class_probs, output_type=tf.dtypes.int32)

                        if contain_prob * max_prob > 0.1:
                            boxes = boxes.write(count, tf.reshape(adjusted_box, (1, 4)))
                            cls_indexes = cls_indexes.write(count, cls_index)
                            confidences = confidences.write(count, contain_prob * max_prob)
                            count += 1
                        else:
                            boxes = boxes.write(count, tf.zeros((1, 4)))
                            cls_indexes = cls_indexes.write(count, 0)
                            confidences = confidences.write(count, 0)
                            count += 1

        # if count == 0:
        #     # No detected object
        #     boxes = boxes.write(count, tf.zeros((1, 4)))
        #     cls_indexes = cls_indexes.write(count, 0)
        #     confidences = confidences.write(count, 0)

        boxes = boxes.stack()
        cls_indexes = cls_indexes.stack()
        confidences = confidences.stack()

        keep = self._nms_tf(boxes, confidences)

        num_boxes = len(keep) # size 1 containing the number of detected boxes
        detection_boxes = tf.expand_dims(tf.gather(boxes, keep), axis=0) # shape [1, num_boxes, 4] with box locations
        detection_boxes = tf.reshape(detection_boxes, [1, num_boxes, 4])

        detection_classes = tf.expand_dims(tf.gather(cls_indexes, keep), axis=0) # shape [1, num_boxes] with class indices
        detection_classes = tf.reshape(detection_classes, [1, num_boxes])
        
        detection_scores = tf.expand_dims(tf.gather(confidences, keep), axis=0) # shape [1, num_boxes] with class scores
        detection_scores = tf.reshape(detection_scores, [1, num_boxes])

        return detection_boxes, detection_classes, detection_scores, num_boxes
    

    @tf.function
    def _nms_tf(self, b_boxes, scores, threshold=0.5):
        x1 = b_boxes[:, 0, 0]
        y1 = b_boxes[:, 0, 1]
        x2 = b_boxes[:, 0, 2]
        y2 = b_boxes[:, 0, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = tf.argsort(scores, direction='DESCENDING')
        keep = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        count = 0

        while tf.size(order) > 0:
            i = order[0] if tf.size(order) == 1 else order[0]
            keep = keep.write(count, i)
            count += 1

            if tf.size(order) == 1:
                break

            xx1 = tf.maximum(x1[i], tf.gather(x1, order[1:]))
            yy1 = tf.maximum(y1[i], tf.gather(y1, order[1:]))
            xx2 = tf.minimum(x2[i], tf.gather(x2, order[1:]))
            yy2 = tf.minimum(y2[i], tf.gather(y2, order[1:]))

            w = tf.maximum(0.0, xx2 - xx1)
            h = tf.maximum(0.0, yy2 - yy1)
            intersection = w * h
            union = areas[i] + tf.gather(areas, order[1:]) - intersection

            iou = intersection / union
            ids = tf.where(iou <= threshold)[:, 0]

            if tf.size(ids) == 0:
                break

            order = tf.gather(order, ids + 1)
        return keep.stack()
    """

class ProxylessNASNetsTF:

    def __init__(self, net_config, net_weights=None, graph=None, sess=None, is_training=True, images=None,
                 img_size=None, n_classes=20, S=5, A=5):
        if graph is not None:
            self.graph = graph
            slim = True
        else:
            self.graph = tf.Graph()
            slim = False

        self.net_config = net_config
        self.n_classes = n_classes
        self.S = S
        self.A = A

        with self.graph.as_default():
            self._define_inputs(slim=slim, is_training=is_training, images=images, img_size=img_size)
            
            """
            if is_training:
                logits = self.build(init=net_weights, is_training=True) # output pred from the network
                self.logits = logits
            else:
                # Return detection_boxes, detection_classes, detection_scores, num_boxes
                logits = self.build(init=net_weights, is_training=False)
                self.logits = logits
            """
            logits = self.build(init=net_weights, is_training=is_training)
            self.logits = logits 

            self.global_variables_initializer = tf.compat.v1.global_variables_initializer()
        self._initialize_session(sess)

    @property
    def bn_eps(self):
        return self.net_config['bn']['eps']

    @property
    def bn_decay(self):
        return 1 - self.net_config['bn']['momentum']

    def _initialize_session(self, sess):
        """ Initialize session, variables """
        config = tf.compat.v1.ConfigProto() 
        if sess is None:
            self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
        else:
            self.sess = sess
        self.sess.run(self.global_variables_initializer)

    def _define_inputs(self, slim=False, is_training=True, images=None, img_size=None):
        if isinstance(img_size, list) or isinstance(img_size, tuple):
            assert len(img_size) == 2
            shape = [None, img_size[0], img_size[1], 3]
        else:
            shape = [None, img_size, img_size, 3]
        if images is not None:
            self.images = images
        else:
            self.images = tf.compat.v1.placeholder(
                tf.float32,
                shape=shape,
                name='input_images')
        self.labels = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, self.S, self.S, self.A*(5 + self.n_classes)],
            name='labels')

        if slim:
            self.is_training = is_training
        else:
            self.is_training = tf.compat.v1.placeholder(
                tf.bool, shape=[], name='is_training')

    # Model definition
    def build(self, init=None, is_training=False):
        output = self.images # Start with input images

        if init is not None:
            for key in init:
                init[key] = tf.compat.v1.constant_initializer(init[key])

        # Backbone structure
        # first conv
        first_conv = ConvLayer(
            'first_conv',
            self.net_config['first_conv']['out_channels'],
            3,
            2)
        output = first_conv.build(output, self, init)

        # mobile inverted residual blocks
        for i, block_config in enumerate(self.net_config['blocks']):
            # No ZeroLayer in my backbone
            # if block_config['mobile_inverted_conv']['name'] == 'ZeroLayer':
            #     continue
            
            # create mobile inverted convolution
            mobile_inverted_conv = MBInvertedConvLayer(
                'mobile_inverted_conv',
                block_config['mobile_inverted_conv']['out_channels'],
                block_config['mobile_inverted_conv']['kernel_size'],
                block_config['mobile_inverted_conv']['stride'],
                block_config['mobile_inverted_conv']['expand_ratio'],
            )
            # wrap in residual block
            if block_config['shortcut'] is None: # or block_config['shortcut']['name'] == 'ZeroLayer':
                has_residual = False
            else:
                has_residual = True
            block = MobileInvertedResidualBlock(
                'blocks/%d' %
                i, mobile_inverted_conv, has_residual)
            output = block.build(output, self, init)

        # feature mix layer
        if self.net_config['feature_mix_layer'] is not None:
            feature_mix_layer = ConvLayer(
                'feature_mix_layer',
                self.net_config['feature_mix_layer']['out_channels'],
                1,
                1)
            output = feature_mix_layer.build(output, self, init)

        # Flexible Classifier building 
        print("Building classifier from config ...")

        # Build layer1 based on config 
        print(self.net_config['classifier'])
        layer1_config = self.net_config['classifier']['layer1']

        print(layer1_config)

        layer1 = build_layer_from_config(layer1_config, _id = 'layer1')
        
        # Create classifier with dynamic layers
        classifier = YOLOClassifier(
            'classifier',
            self.S,
            layer1, # COuld be ConvLayerPad ... 
            None,
            self.net_config['classifier']['isConv'],
            self.A
        )

        output = classifier.build(output, self, init, is_training)
        
        # Always return raw predictions
        return output # Shape: [1, 5, 5, 125]