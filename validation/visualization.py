import os
import torch
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from torchvision.ops import batched_nms
from dotenv import load_dotenv
from config import VOC_CLASSES
from .bboxprep import decode_pred, target_tensor_to_gt


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

                    if pred_boxes is not None and pred_boxes.shape[0] > 0:
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
                    else:
                        print(f"No predictions for image {img_idx}")
                        
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



def plot_pr_curves_comp(recall_vals_list, precision_vals_list, ap_list,
                        save_path=None, title="Precision-Recall Curves Comparison"):

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']

    for i, (recall_vals, precision_vals, ap) in enumerate(zip(recall_vals_list, precision_vals_list, ap_list)):
        color = colors[i % len(colors)]
        if i == 0:
            plt.plot(recall_vals, precision_vals, marker='o', linestyle='-', color=color, 
                    label=f"AP@0.5 = {ap:.4f}", linewidth=2, markersize=4)
        if i == 1:
            plt.plot(recall_vals, precision_vals, marker='o', linestyle='-', color=color, 
                    label=f"WBF-AP@0.5 = {ap:.4f}", linewidth=2, markersize=4)

    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Comparison PR curve saved to {save_path}")


def draw_bboxes_pil(pil_img, bboxes, color=(0, 255, 0), thickness=2, add_scores=False):
    """Draw bounding boxes on a PIL image."""
    try:
        if pil_img.mode!= 'RGB':
            pil_img = pil_img.convert('RGB')
        
        img_array = np.array(pil_img)
        if img_array.size == 0:
            print("Error: Empty image array")
            return pil_img
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    except Exception as e:
        print(f"Error converting image: {e}")
        return pil_img
    
    for box in bboxes:
        if box is None:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if add_scores and len(box) > 4:
            score = box[4]
            text = f"{score:.2f}"
            # Place text inside the box with padding 
            text_x, text_y = x1 + 5, y1 + 15  
            cv2.putText(img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA) 
            
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def visualize_bbox_grid_tensors(images_view1, images_view2,
                                 preds_view1, wbf_view1,
                                 preds_view2, wbf_view2,
                                 save_path=None, max_cols=10):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert tensors to PIL
    images_v1 = []
    images_v2 = []

    for img in images_view1:
        img = img.cpu()
        images_v1.append(transforms.ToPILImage()(img))

    for img in images_view2:
        img = img.cpu()
        images_v2.append(transforms.ToPILImage()(img))

    rows = 4
    total_images = len(images_v1)
    for chunk_idx in range(0, total_images, max_cols):
        start_col = chunk_idx
        end_col = min(chunk_idx + max_cols, total_images)
        cols = end_col - start_col

        rows = 4
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        row_titles = ["View1 Base", "View1 WBF", "View2 Base", "View2 WBF"]

        for col in range(cols):
            actual_col = start_col + col
            for row in range(rows):
                ax = axes[row, col]
                if row in (0, 1):  # view1
                    img = images_v1[actual_col]
                    bboxes = preds_view1[actual_col] if row == 0 else wbf_view1[actual_col]
                    boxes = bboxes[:, :4]
                    scores = bboxes[:, 4]
                    labels = bboxes[:, 5].long()
                    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
                    nms_pred_boxes = bboxes[keep]
                    if nms_pred_boxes.shape[0] > 0:
                        idx = torch.argmax(nms_pred_boxes[:, 4])
                        max_bbox = nms_pred_boxes[idx]
                    else:
                        max_bbox = None

                else:  # view2
                    img = images_v2[actual_col]
                    bboxes = preds_view2[actual_col] if row == 2 else wbf_view2[actual_col]
                    boxes = bboxes[:, :4]
                    scores = bboxes[:, 4]
                    labels = bboxes[:, 5].long()
                    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
                    nms_pred_boxes = bboxes[keep]
                    if nms_pred_boxes.shape[0] > 0:
                        idx = torch.argmax(nms_pred_boxes[:, 4])
                        max_bbox = nms_pred_boxes[idx]
                    else:
                        max_bbox = None

                img_with_boxes = draw_bboxes_pil(img, [max_bbox], add_scores=True)
                ax.imshow(img_with_boxes)
                ax.axis('off')
                if row == 0:
                    ax.set_title(f"Object {actual_col+1}", fontsize=10)
        
        for row in range(rows):
            axes[row, 0].text(-0.1, 0.5, row_titles[row], transform=axes[row,0].transAxes,
            fontsize=12, rotation=90, verticalalignment='center', horizontalalignment='right')
        
        plt.tight_layout()
        base_path = save_path.replace('.png', '')

        if chunk_idx == 0:
            chunk_save_path = save_path
        else:
            chunk_num = (chunk_idx // max_cols) + 1
            chunk_save_path = f"{base_path}_part{chunk_num}.png"

        plt.savefig(chunk_save_path, dpi=300)
        plt.close()
        print(f"Saved visualization grid to {chunk_save_path}")


def visualize_bbox_grid_tensors_single_view(
        images_view1, preds_view1, save_path="./image_result/bbox_grid_single.png", max_cols=10):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert tensors to PIL
    images_v1 = []

    for img in images_view1:
        img = img.cpu()
        images_v1.append(transforms.ToPILImage()(img))

    total_images = len(images_v1)
    for chunk_idx in range(0, total_images, max_cols):
        start_col = chunk_idx
        end_col = min(chunk_idx + max_cols, total_images)
        cols = end_col - start_col

        rows = 1
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

        if cols == 1:
            axes = [axes] # Make it a list for consistent indexing

        for col in range(cols):
            actual_col = start_col + col
            
            ax = axes[col]
            img = images_v1[actual_col]
            bboxes = preds_view1[actual_col] 

            if bboxes:
                nms_bboxes = apply_classwise_nms(pred_boxes=bboxes, iou_thresh=0.5)
                if nms_bboxes:
                    max_bbox= max(nms_bboxes, key=lambda x: x[4])
                    img_with_boxes = draw_bboxes_pil(img, [max_bbox], add_scores=True)
                else:
                    img_with_boxes = img
            else:
                img_with_boxes = img

            ax.imshow(img_with_boxes)
            ax.axis('off')
            ax.set_title(f"Image {actual_col+1}", fontsize=12)
        
        fig.suptitle("Single View Detection Resutls", fontsize=16, y=0.95)
       
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        base_path = save_path.replace('.png', '')
        if chunk_idx == 0:
            chunk_save_path = save_path
        else:
            chunk_num = (chunk_idx // max_cols) + 1
            chunk_save_path = f"{base_path}_part{chunk_num}.png"

        plt.savefig(chunk_save_path, dpi=300)
        plt.close()
        print(f"Saved single view visualization to {chunk_save_path}")

