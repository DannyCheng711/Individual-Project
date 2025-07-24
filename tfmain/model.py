import tensorflow as tf 
from tensorflow.keras import layers, models 
from mcunet.mcunet.model_zoo import net_id_list, build_model, download_tflite
from config import DEVICE, VOC_ROOT

class YoloHead(tf.keras.layers.Layer):
    def __init__(self, num_classes, num_anchors):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_channels = num_anchors * (5 + num_classes)
        self.detector = tf.keras.layers.Conv2D(self.output_channels, kernel_size=1)
    
    def call(self, x):
        return self.detector(x) # TF uses [B, H, W, C] by default

class SpaceToDepth(tf.keras.layers.Layer):
    def __init__(self, block_size = 2):
        super().__init__()
        self.block_size = block_size
    
    # NHWC
    def call(self, inputs):
        return tf.nn.space_to_depth(inputs, block_size = self.block_size)

class McuYolo(tf.keras.Model):
    def __init__(self, backbone_fn, num_classes = 20, num_anchors = 5):
        super().__init__()
        self.backbone = backbone_fn

        self.passthrough_layer_idx = 12 
        self.final_block_idx = 16

        # extra 3 conv layer: conv1 and conv2
        # tf doesn't explicitly specify the input channel size
        self.conv1 = layers.Conv2D(512, kernel_size=1, activation='relu')
        self.conv2 = layers.Conv2D(512, kernel_size=1, activation='relu')

        # space to depth
        self.space_to_depth = SpaceToDepth() # 10×10×96 -> 5×5×384

        # extra 3 conv layer: conv3
        self.conv3 = layers.Conv2D(512, kernel_size=1, activation='relu')

        # add detection head
        self.det_head = YoloHead(num_classes=num_classes, num_anchors=num_anchors)


    # inputs [B, H, W, C]
    def call(self, inputs, training = False):
        x = inputs 
        passthrough_feat = None

        # traverse backbone up to final_block_idx
        for i in range(self.final_block_idx + 1):
            x = self.backbone.blocks[i](x)
            if i == self.passthrough_layer_idx:
                passthrough_feat = x 
            if i == self.final_block_idx:
                break

        # head conv layers
        x = self.conv1(x) 
        x = self.conv2(x) 

        passthrough = self.space_to_depth(passthrough_feat) 
        x = tf.concat([x, passthrough], axis=-1) # concatenate

        x = self.conv3(x) 

        # final detection 
        return self.det_head(x)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            params.requires_grad = False
        print("Backbone frozen ...")
    
    def unfreeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            params.requires_grad = True
        print("Backbone unfrozen ...")


# Loss function
class Yolov2Loss(tf.keras.losses.Loss):
    def __init__(self, num_classes, anchors, lambda_coord = 5.0, lambda_noobj = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def calculate_iou(self, ground_box, pred_box):
        """
        Compute IoU between two boxes in [cx, cy, w, h] format.
        All coordinates should be in the same unit (e.g., grid cells).
        """

        cx1, cy1, w1, h1 = ground_box
        cx2, cy2, w2, h2 = pred_box

        # convert center format to corner format 
        x1_min = cx1 - w1 / 2
        y1_min = cy1 - h1 / 2
        x1_max = cx1 + w1 / 2
        y1_max = cy1 + h1 / 2
        
        x2_min = cx2 - w2 / 2
        y2_min = cy2 - h2 / 2
        x2_max = cx2 + w2 / 2
        y2_max = cy2 + h2 / 2

        # Compute intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    def calculate_iou_wh(self, ground_box, anchor_box):
        """
        Compute IoU between two boxes using only [w, h] dimensions.
        Assumes both boxes are centered at the same location (e.g., (0, 0)).
        Used for anchor shape matching only.
        """

        _, _, w1, h1 = ground_box
        _, _, w2, h2 = anchor_box

        # Compute intersection
        inter_w = min(w1, w2)
        inter_h = min(h1, h2)
        inter_area = inter_w * inter_h

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    def call(self, predictions, targets):
        # predictions [B, S, S, A*(5+C)]
        # targets [B, S, S, A, 5+C]

        B = tf.shape(predictions)[0]
        S = tf.shape(predictions)[1]
        A = len(self.anchors)
        C = self.num_classes

        # Reshape predictions
        predictions = tf.reshape(B, S, S, A, 5 + C)
        # Extract tx, ty, tw, th, obj_score, class_probs
        tx = predictions[..., 0]
        ty = predictions[..., 1]
        tw = predictions[..., 2]
        th = predictions[..., 3]
        conf_logits = predictions[..., 4]
        class_logits = predictions[..., 5:]

        # Extract targets 
        gx = y_true[..., 0]
        gy = y_true[..., 1]
        gw = y_true[..., 2]
        gh = y_true[..., 3]
        obj_mask = y_true[..., 4] # 1 is object, 0 otherwise
        class_target = y_true[..., 5:]

        # Predicted box decoding
        pred_cx = tf.sigmoid(tx)
        pred_cy = tf.sigmoid(ty)
        pred_w = tf.exp(tw) * self.anchors[:, 0]
        pred_h = tf.exp(th) * self.anchors[:, 1]

        # Ground truth box for IoU 
        iou = tf.zeros_like(pred_cx)
        for b in range(B):
            for a in range(A):
                for i in range(S):
                    for j in range(S):
                        if obj_mask[b, i, j, a] == 1:
                            # Convert into grid cell unit
                            cx_gt = gx[b, i, j, a] + tf.cast(j, tf.float32)
                            cy_gt = gy[b, i, j, a] + tf.cast(i, tf.fload32)

                            cx_pr = pred_cx[b, i, j, a] + tf.cast(j, tf.float32)
                            cy_pr = pred_cy[b, i, j, a] + tf.cast(i, tf.float32)

                            gt_box = [cx_gt, cy_gt, gw[b, i, j, a], gh[b, i, j, a]]
                            pr_box = [cx_pr, cy_pr, pred_w[a], pred_h[a]]

                            # TensorFlow tensors are immutable by default
                            iou = tf.tensor_scatter_nd_update(iou, [[b, i, j, a]], [self.calculate_iou(gt_box, pr_box)])

        # Compute loss terms, reduce_sum: sum elements across dimensions
        loss_xy = tf.reduce_sum(obj_mask * (tf.square(pred_cx - gx) + tf.square(pred_cy - gy)))
        loss_wh = tf.reduce_sum(obj_mask * (tf.square(tf.sqrt(pred_w + 1e-6) - tf.sqrt(gw + 1e-6)) +
                                            tf.square(tf.sqrt(pred_h + 1e-6) - tf.sqrt(gh + 1e-6))))

        conf_pred = tf.sigmoid(conf_logits)
        loss_obj = tf.reduce_sum(obj_mask * tf.square(iou - conf_pred))
        loss_noobj = tf.reduce_sum((1.0 - obj_mask) * tf.square(conf_pred))

        loss_cls = tf.reduce_sum(obj_mask * tf.keras.losses.binary_crossentropy(class_target, class_logits, from_logits=True))

        total_loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_cls

        return total_loss


if __name__ == "__main__":

    with tf.device(DEVICE):
        print("Creating dummy inputs ... ")
        # Dummy input: [B, H, W, C] format in TensorFlow (channels last)
        dummy_input = tf.random.normal([2, 160, 160, 3])

        # Dummy target: [B, S, S, A, 5+C]
        # Assuming your grid S=5, anchors A=5, and classes C=20
        dummy_target = tf.zeros([2, 5, 5, 5, 5 + 20])  # example placeholder

        # Put one box in one grid cell as a positive example
        # Box: [gx, gy, gw, gh, obj=1, one-hot class vector]
        dummy_target = tf.tensor_scatter_nd_update(
            dummy_target,
            indices=[[0, 2, 3, 1]],  # batch 0, grid[2][3], anchor 1
            updates=[tf.concat([
                tf.constant([0.5, 0.5, 0.2, 0.3, 1.0]),  # box + obj
                tf.one_hot(10, 20)                      # class 10
            ], axis=0)]
        )

        # Backbone
        print("Creating a backbone ... ")
        backbone_fn, _, _ = build_model(net_id="mcunet-in4", pretrained=True)

        # Model & Loss
        print("Building a model ... ")
        model = McuYolo(backbone_fn=backbone_fn, num_classes=20, num_anchors=5)
        print("Building a loss function ... ")
        loss_fn = Yolov2Loss(num_classes=20, anchors=tf.constant([[1,2], [2,1], [1.5,1.5], [2,2], [1,1]], dtype=tf.float32))

        # Forward pass
        print("Forward pass ... ")
        y_pred = model(dummy_input, training=False)
        loss = loss_fn(y_pred, dummy_target)

        print("Dummy loss:", float(loss))