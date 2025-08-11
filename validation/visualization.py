import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import ImageDraw
from torchvision import transforms
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

                    if len(pred_boxes) > 0:
                        nms_pred_boxes = apply_classwise_nms(pred_boxes=pred_boxes, iou_thresh=0.5)
                        # Sort by confidence and keep top 5
                        nms_pred_boxes = sorted(nms_pred_boxes, key=lambda x: -x[4])[:5]
                    else:
                        nms_pred_boxes = []

                    for pred in nms_pred_boxes:
                        xmin, ymin, xmax, ymax, conf, class_id = pred
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

            

def save_preds_image(img_tensor, pred_tensor, epoch, batch_idx, img_idx, save_dir="./vocmain/images_voc", threshold=0.5):
        os.makedirs(save_dir, exist_ok=True)
        img = transforms.functional.to_pil_image(img_tensor.cpu())
        draw = ImageDraw.Draw(img)
        decoded_boxes = decode_pred(
            pred_tensor.unsqueeze(0),
            anchors=VOC_ANCHORS,
            num_classes=len(VOC_CLASSES),
            image_size=160,
            conf_thresh=threshold
        )
        for box in decoded_boxes[0]:
            xmin, ymin, xmax, ymax, conf, class_id = box
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            draw.text((xmin, ymin - 10), f"Class {int(class_id)}: {conf:.2f}", fill="red")
        filename = f"epoch_{epoch}_batch_{batch_idx}_image_{img_idx}.jpg"
        img.save(os.path.join(save_dir, filename))

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

