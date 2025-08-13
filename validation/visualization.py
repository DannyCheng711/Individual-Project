import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import ImageDraw
from torchvision import transforms
from torchvision.ops import batched_nms
from dotenv import load_dotenv
from config import VOC_CLASSES, VOC_ANCHORS
from .bbox_utils import decode_pred, target_tensor_to_gt, apply_classwise_nms

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")

def visualize_predictions(val_loader, model, anchors, image_size, num_classes,  conf_thresh=0.5, save_dir=None):
    """
    Visualize model predictions vs ground truth on sample images
    """

    model.eval()
    
    # Convert anchors to tensor if needed
    if isinstance(anchors, list):    
        anchors = torch.tensor(anchors, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(val_loader):
            # only check batch 0
            if batch_idx >= 10:
                break
            
            fig, axes = plt.subplots(2, 8, figsize=(24, 12))
            axes = axes.flatten()
            
            try:
                imgs = imgs.to(DEVICE)
                preds = model(imgs)
                # only visualize the bbox which conf >= 0.5
                # [B, A*(5+C), S, S] 
                decoded_preds = decode_pred(
                    preds, anchors=anchors, num_classes=num_classes, image_size=image_size, conf_thresh=conf_thresh)

                for img_idx in range(len(imgs)):
                    ax = axes[img_idx]
                    img_pil = transforms.functional.to_pil_image(imgs[img_idx].cpu())
                    ax.imshow(img_pil)
                    
                    # Draw ground truth boxes (green)
                    target_tensor = targets[img_idx]
                    gt_xyxy, gt_labels = target_tensor_to_gt(target_tensor, image_size)

                    for k in range(len(gt_labels)):
                        x1, y1, x2, y2 = gt_xyxy[k].tolist()
                        w_box = x2 - x1
                        h_box = y2 - y1
                        class_id = int(gt_labels[k].item())
                        class_name = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else f"C{class_id}"
                        rect = patches.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor='green', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x1, y1-5, f'GT: {class_name}', color='green', fontweight='bold')
                    
                    # Draw prediction boxes (red)
                    pred_boxes = decoded_preds[img_idx]

                    if pred_boxes.shape[0] > 0:
                        boxes = pred_boxes[:, :4]
                        scores = pred_boxes[:, 4]
                        labels = pred_boxes[:, 5].long()
                        keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
                        boxes = boxes[keep]
                        scores = scores[keep]
                        labels = labels[keep]

                         # Get top 5 by confidence
                        if scores.numel() > 0:
                            topk = min(5, scores.size(0))
                            topk_scores, topk_idx = scores.topk(topk)
                            boxes = boxes[topk_idx]
                            scores = scores[topk_idx]
                            labels = labels[topk_idx]

                    for i in range(boxes.shape[0]):
                        xmin, ymin, xmax, ymax = boxes[i].tolist()
                        conf = scores[i].item()
                        class_id = labels[i].item()
                        w = xmax - xmin
                        h = ymax - ymin
                        
                        # Draw pred box in red
                        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, 
                                            edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
                        
                        class_name = VOC_CLASSES[int(class_id)] if int(class_id) < len(VOC_CLASSES) else f"C{int(class_id)}"
                        ax.text(xmin, ymin-5, f'Pred: {class_name} ({conf:.2f})', color='red', fontweight='bold')
                    
                    ax.set_title(f'Image {img_idx}')
                    ax.axis('off')

                plt.tight_layout()
                batch_save_path = os.path.join(save_dir, f"batch_{batch_idx:02d}_predictions.png")
                plt.savefig(batch_save_path, dpi=300, bbox_inches='tight')
                print(f"Prediction visualization saved to {batch_save_path}")
            finally:
                plt.close()


def plot_loss(loss_log, save_path):
    plt.figure(figsize=(10, 6))
    for k in loss_log:
        plt.plot(loss_log[k], label=k)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

