import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from config import DEVICE, VOC_CLASSES
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np 
import json 

import model 
from validation import decode_pred, evaluate_pred_gt, visualize_bbox_grid_tensors, evaluate_pred_gt_handcraft, plot_pr_curves_comp
from fusion import weighted_boxes_fusion

# 5 views construct a batch
# we have 20 objects, so we have 20 batches for each occlusion rate

curclass = "car"
curclassid = 6

def process_image(image_paths, ground_truths, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    processed_images = []
    resized_gts = []

    for img_path, gt in zip(image_paths, ground_truths):
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size

        image_tensor = transform(image)
        processed_images.append(image_tensor)

        x1, y1, x2, y2, class_id = gt
        scale_x = image_size / orig_width
        scale_y = image_size / orig_height

        resized_gt = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, class_id]
        resized_gts.append([resized_gt])

    all_images = torch.stack(processed_images).to(DEVICE)

    return all_images, resized_gts

# Filter predictions to only keep specific detections
def filter_class_only(decoded_preds, spec_class_id):
    class_only_preds = []

    for pred_list in decoded_preds:
        boxes = []
        for box in pred_list:
            xmin, ymin, xmax, ymax, conf, class_id = box
            if int(class_id) == spec_class_id:
                boxes.append(box)
        class_only_preds.append(boxes)
    
    return class_only_preds

anchors = torch.tensor([
        [8.851960381366869 / 32, 20.71491425034285 / 32], 
        [27.376232242192298 / 32, 56.73302518805578 / 32], 
        [42.88177452824786 / 32, 98.24243329589638 / 32], 
        [67.68032082718717 / 32, 132.7704493338952 / 32], 
        [131.16250016574756 / 32, 137.1476847579408 / 32]
    ], dtype=torch.float32)

detector = model.ObjectDetector("./yolovoc_150_aug_epoch_80.pkl", anchors, 0.5)
base_path = "./co3d/car"

with open(os.path.join(base_path, "manifest_with_occ.json"), 'r') as f:
    manifest = json.load(f)

occlusion_combination = [
    ("occ30", "occ30"),
    ("occ30", "occ50"), 
    # ("occ30", "occ70"), 
    ("occ50", "occ50"),
    # ("occ50", "occ70"), 
    # ("occ70", "occ70")
]

for occ_comb1, occ_comb2 in occlusion_combination:
    print(f"\n ======== Processing {occ_comb1} vs {occ_comb2} ========")

    # Collect all image and ground truthes paths for both views
    image_paths_view1 = []
    image_paths_view2 = []
    all_gt1 = []
    all_gt2 = []

    # Process pairs of images
    for i in range(0, len(manifest) - 1, 2): # Process pairs 
        item1 = manifest[i]
        item2 = manifest[i + 1]

        filename1 = item1[occ_comb1]
        filename2 = item2[occ_comb2]
        
        # Check if files exist
        if os.path.exists(filename1) and os.path.exists(filename2):
            image_paths_view1.append(filename1)
            image_paths_view2.append(filename2)
            gt1 = item1['bbox'] + [curclassid]
            gt2 = item2['bbox'] + [curclassid]
            
            all_gt1.append(gt1)
            all_gt2.append(gt2)

    print(f"Processing {len(image_paths_view1)} image pairs")
        
    # Create batches for bot views (5 images per batch)
    all_images1, resized_gt1 = process_image(image_paths_view1, all_gt1, image_size=160)
    all_images2, resized_gt2 = process_image(image_paths_view2, all_gt2, image_size=160)


    print("Getting predictions for view 1 ...")
    with torch.no_grad():
        preds1 = detector.model(all_images1) # [All images, A*(5+C), S, S]

    print("Getting predictions for view 2 ...")
    # Process bathces for view 2
    with torch.no_grad():
        preds2 = detector.model(all_images2) # [All images, A*(5+C), S, S]

    # Decode predictions for both views 
    print("Decoding predictions...")
    # [ 
    #   [ # images
    #       # detections [x1, x2, y1, y2, conf, classid]]] images, detections
    decoded_preds1 = decode_pred(
        preds1, anchors=anchors, num_classes=len(VOC_CLASSES), 
        image_size=160, conf_thresh= 0.001
    )

    decoded_preds2 = decode_pred(
        preds2, anchors=anchors, num_classes=len(VOC_CLASSES),
        image_size=160, conf_thresh= 0.001
    )

    class_preds1 = filter_class_only(decoded_preds1, curclassid)
    class_preds2 = filter_class_only(decoded_preds2, curclassid)

    # Evaluate occ rate 1 
    print("Evaluating Occlusion Rate 1 ...")
    results1 = evaluate_pred_gt(class_preds1, resized_gt1, device=DEVICE)
    ap_1, recall_1, precision_1 = evaluate_pred_gt_handcraft(class_preds1, resized_gt1, iou_threshold=0.5)
    print("Evaluating Occlusion Rate 2 ...")
    results2 = evaluate_pred_gt(class_preds2, resized_gt2, device=DEVICE)
    ap_2, recall_2, precision_2 = evaluate_pred_gt_handcraft(class_preds2, resized_gt2, iou_threshold=0.5)

    print("Applying Weighted Box Fusion for each class ... ")
    wbf_class_preds = []

    # For each image 
    for i in range(len(class_preds1)):
        combined_boxes = []

        # Integrate all boxes in occ-rate view 1 and 2
        for box1 in class_preds1[i]:
            xmin, ymin, xmax, ymax, conf, class_id = box1
            combined_boxes.append([xmin/160, ymin/160, xmax/160, ymax/160, conf, int(class_id)])
        for box2 in class_preds2[i]:
            xmin, ymin, xmax, ymax, conf, class_id = box2
            combined_boxes.append([xmin/160, ymin/160, xmax/160, ymax/160, conf, int(class_id)])

        if combined_boxes:
            fused_boxes, fused_scores, fused_classes = weighted_boxes_fusion(
                combined_boxes, iou_thr=0.55, n_models=2
            )   
            # convert back to pixel coordinates
            wbf_result = []
            for j in range(len(fused_boxes)):
                box = fused_boxes[j]
                xmin, ymin, xmax, ymax = box[0]*160, box[1]*160, box[2]*160, box[3]*160
                wbf_result.append([xmin, ymin, xmax, ymax, fused_scores[j], int(fused_classes[j])])

            wbf_class_preds.append(wbf_result)
        
        else:
            wbf_class_preds.append([])

    print("Evaluateing WBF ...")

    print("Evaluating WBF Occlusion Rate 1 ...")
    wbf_results1 = evaluate_pred_gt(wbf_class_preds, resized_gt1, device=DEVICE)
    wbf_ap_1, wbf_recall_1, wbf_precision_1 = evaluate_pred_gt_handcraft(wbf_class_preds, resized_gt1, iou_threshold=0.5)
    plot_pr_curves_comp(
        recall_vals_list=[recall_1, wbf_recall_1],
        precision_vals_list=[precision_1, wbf_precision_1],
        ap_list=[ap_1, wbf_ap_1],
        save_path=f"./wbf_result/pr_curve_comp_view1_{occ_comb1}_{occ_comb2}.png"
    )
    print("Evaluating WBF Occlusion Rate 2 ...")
    wbf_results2 = evaluate_pred_gt(wbf_class_preds, resized_gt2, device=DEVICE)
    wbf_ap_2, wbf_recall_2, wbf_precision_2 = evaluate_pred_gt_handcraft(wbf_class_preds, resized_gt2, iou_threshold=0.5)
    plot_pr_curves_comp(
        recall_vals_list=[recall_2, wbf_recall_2],
        precision_vals_list=[precision_2, wbf_precision_2],
        ap_list=[ap_2, wbf_ap_2],
        save_path=f"./wbf_result/pr_curve_comp_view2_{occ_comb1}_{occ_comb2}.png"
    )

    print(f"\n ======== Results for {occ_comb1} vs {occ_comb2} ========")
    print(f"View 1 ({occ_comb1}):")
    print(f" mAP: {results1['map']:.4f}")
    print(f" mAP@50: {results1['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ap_1}")

    print(f"WBF View 1 ({occ_comb1}):")
    print(f" mAP: {wbf_results1['map']:.4f}")
    print(f" mAP@50: {wbf_results1['map_50']:.4f}")
    print(f" hand-craft mAP@50: {wbf_ap_1}")

    print(f"View 2 ({occ_comb2}):")
    print(f" mAP: {results2['map']:.4f}")
    print(f" mAP@50: {results2['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ap_2}")

    print(f"WBF View 2 ({occ_comb2}):")
    print(f" mAP: {wbf_results2['map']:.4f}")
    print(f" mAP@50: {wbf_results2['map_50']:.4f}")
    print(f" hand-craft mAP@50: {wbf_ap_2}")

    


    visualize_bbox_grid_tensors(
        all_images1, all_images2,
        class_preds1, wbf_class_preds,
        class_preds2, wbf_class_preds,
        save_path=f"./wbf_result/bbox_grid_{occ_comb1}_{occ_comb2}.png"
    )

    
    