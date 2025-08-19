import tensorflow as tf
import torch
import numpy as np
from .tf2_proxyless_net import ProxylessNASNetsTF

class TFObjectDetector:
    def __init__(self, mcunet_config, tf_weights, conf_threshold, image_size=160):
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
        
        return predictions
    
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