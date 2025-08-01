import cv2 
import numpy as np 
import torch 
import torchvision.transforms as transforms
import json
from tqdm import tqdm
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from PIL import Image
from config import VOC_CLASSES, DEVICE
from yolomodel import McuYolo, Yolov2Loss, SpaceToDepth, YoloHead
import os 

class ObjectDetector:
    def __init__(self, model_path, anchors, confidence_threshold=0.5):
        """
        Initialise object detector

        Args:
            model_name: Pre-trained model to use
            confidence_threshold
        """

        self.anchors = anchors
        self.confidence_threshold = confidence_threshold

        # load model 

        # To load later:
        self.model = torch.load(model_path, weights_only=False)        
        self.model.to(DEVICE)
        self.model.eval()

        # Convert anchors to tensor if needed
        if isinstance(self.anchors, list):    
            self.anchors = torch.tensor(self.anchors, dtype=torch.float32).to(DEVICE)


    def model_construct(self):
        self.anchors = self.anchors.to(DEVICE)
        backbone, _, _ = yolomodel.build_model(net_id="mcunet-in4", pretrained=True)

        # Load model and loss 
        self.net = model.McuYolo(backbone_fn=backbone, num_classes=self.num_classes, num_anchors=len(self.anchors)).to(DEVICE)
        self.loss_fn = model.Yolov2Loss(num_classes=self.num_classes, anchors=self.anchors).to(DEVICE)
        self.optimiser = optim.Adam(self.net.parameters(), lr=1e-4)

    def preprocess_image(self, image_path, image_size):
        """
        Preprocess image for detection

        Args:
            image_path: Path to input image
        Returns:
            Preprocess image tensor and original image
        """

        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image and then to tensor
        pil_image = Image.fromarray(image_rgb)

        # Transform to tensor
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

        # Remove batch dimension before converting to PIL
        img_pil = transforms.functional.to_pil_image(image_tensor.squeeze(0).cpu())

        return image_tensor, img_pil
    
    def detect_objects(self, image_path, input_image_size):
        """
        Detect objects in image 

        Args:
            image_path: Path to input image
            input_image_size: Size to resize image for model input
        Returns:
            Dictionary containing bboxes, scores and labels
        """
        # Preprocess image 
        image_tensor, original_image = self.preprocess_image(image_path, input_image_size)

        # Run detection
        with torch.no_grad():
            predictions = self.model(image_tensor)

        
        return predictions, original_image

    def decode_pred(self, pred, num_classes, image_size, conf_thresh = 0.5):
        """
        Decode YOLO predictions into bounding boxes.
        Args:
            pred: Tensor of shape [B, A*(5+C), S, S] 
            num_classes: number of classes
            image_size: pixel width/height of input image
            conf_thresh: objectness threshold

        Returns:
            List[List[box]]: for each batch element, a list of [xmin, ymin, xmax, ymax, confidence]
        """

        batch_size, pred_dim, S, _ = pred.shape  # [B, A*(5+C), S, S]
        A = len(self.anchors)
        C = num_classes

        # Reshape to [B, A, 5+C, S, S] then permute to [B, S, S, A, 5+C]
        pred = pred.reshape(batch_size, A, 5 + C, S, S)
        pred = pred.permute(0, 3, 4, 1, 2)  # [B, S, S, A, 5+C]
        
        # Extract components
        tx = pred[..., 0]
        ty = pred[..., 1]
        tw = pred[..., 2]
        th = pred[..., 3]
        obj_score = torch.sigmoid(pred[..., 4])
        class_scores = torch.softmax(pred[..., 5:], dim=-1)  # Class probabilities

        all_decoded = []

        for b in range(batch_size):
            decoded_boxes = [] 
            for i in range(S):
                for j in range(S):
                    for a in range(A): 

                        conf = obj_score[b, i, j, a]

                        if conf < conf_thresh: 
                            continue
                        
                        # grid unit -> image unit 
                        cx_grid = j + torch.sigmoid(tx[b, i, j, a])
                        cy_grid = i + torch.sigmoid(ty[b, i, j, a])
                        bw_grid = self.anchors[a, 0] * torch.exp(tw[b, i, j, a]) # anchor is tensor
                        bh_grid = self.anchors[a, 1] * torch.exp(th[b, i, j, a])
                        
                        cx = cx_grid / S
                        cy = cy_grid / S
                        bw = bw_grid / S 
                        bh = bh_grid / S 

                        # convert to box 
                        xmin = (cx - bw/2) * image_size
                        ymin = (cy - bh/2) * image_size
                        xmax = (cx + bw/2) * image_size
                        ymax = (cy + bh/2) * image_size

                        # Clamp to image bounds
                        xmin = max(0, min(xmin.item(), image_size))
                        ymin = max(0, min(ymin.item(), image_size))
                        xmax = max(0, min(xmax.item(), image_size))
                        ymax = max(0, min(ymax.item(), image_size))

                        class_probs = class_scores[b, i, j, a]
                        allowed_classes = [6, 13, 1]  
                        class_probs_filtered = class_probs[allowed_classes]
                        best_class_conf, best_class_pos = torch.max(class_probs_filtered, dim=0)
                        best_class_id = allowed_classes[best_class_pos]
                        ## best_class_conf, best_class_id = torch.max(class_probs, dim=0)
                        final_conf = conf * best_class_conf  # Combined confidence
                                
                        ## decoded_boxes.append(
                        ##     [xmin, ymin, xmax, ymax, final_conf.item(), best_class_id.item()])
                        decoded_boxes.append(
                            [xmin, ymin, xmax, ymax, final_conf.item(), best_class_id])

            all_decoded.append(decoded_boxes)

        return all_decoded

    def visualize_single_predictions(self, image_path, image_size, save_path=None, saved_image_idx=None):
        """
        Visualize model predictions on a single image

        Args:
            image_path: Path to input image
            image_size: Size for model input
            save_path: Path to save visualization 
        """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"Created directory: {save_path}")

        # Get predictions
        preds, original_image = self.detect_objects(image_path, image_size)

        # Decode predictions
        decoded_preds = self.decode_pred(
            preds, num_classes=len(VOC_CLASSES), image_size = image_size,
            conf_thresh = self.confidence_threshold
        )

        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        ax.imshow(original_image)

        # Draw bbox for the batch
        pred_boxes = decoded_preds[0] if decoded_preds else []

        pred_boxes = sorted(pred_boxes, key=lambda x:x[4], reverse=True)[:10]

        for pred in pred_boxes:
            xmin, ymin, xmax, ymax, conf, class_id = pred
            w = xmax - xmin
            h = ymax - ymin
                        
            # Draw pred box in red
            rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, 
                edgecolor='red', facecolor='none')
            ax.add_patch(rect)
                
            class_name = VOC_CLASSES[int(class_id)]
            ax.text(xmin, ymin-5, f'Pred: {class_name} ({conf:.2f})', color='red', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round, pad=0.3', facecolor='yellow', alpha=0.7))
            
        ax.set_title(f'Detection Results: {len(pred_boxes)} objects detected')
        ax.axis('off')
                    
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.close(fig)

if __name__ == "__main__":

    anchors = torch.tensor([
        [8.851960381366869 / 32, 20.71491425034285 / 32], 
        [27.376232242192298 / 32, 56.73302518805578 / 32], 
        [42.88177452824786 / 32, 98.24243329589638 / 32], 
        [67.68032082718717 / 32, 132.7704493338952 / 32], 
        [131.16250016574756 / 32, 137.1476847579408 / 32]
    ], dtype=torch.float32)

    detector = ObjectDetector("./yolovoc_150_aug_epoch_80.pkl", anchors, 0.5)

    base_path = "./co3d/car"

    with open(os.path.join(base_path, "manifest_with_occ.json"), 'r') as f:
        manifest = json.load(f)

        for item in tqdm(manifest, desc="Retrieving Images"):
            for filename in [item['image'], item['occ30'], item['occ50'], item['occ70']]:
                bbox = item['bbox']
                # Visualize predictions
                save_path = os.path.join(
                        "./detection_result/car", item['seq_name'] ,f"{os.path.splitext(os.path.basename(filename))[0]}_det.jpg")
                
                detector.visualize_single_predictions(
                    filename, 
                    image_size=160, 
                    save_path=save_path, 
                )
            
