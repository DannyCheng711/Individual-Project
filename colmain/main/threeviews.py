import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import VOC_ANCHORS, VOC_CLASSES, VOC_CLASS_TO_IDX
import json 
from dotenv import load_dotenv
from models.mcunetYolo.tinynas.nn.proxyless_net import ProxylessNASNets
from models.dethead.mvyolodet import MultiViewMcuYolo
from utils.config_utils import fix_state_dict_keys
from dataset.occlusion.preprocess import process_image, filter_class_only
from validation.bboxprep import decode_pred
from validation.metrics import eval_metric_col, eval_metric_col_handcraft
from validation.visualization import visualize_bbox_grid_tensors, plot_pr_curves_comp
from ..decision_fusion.fusion import weighted_boxes_fusion

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device("cuda")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")
IMAGE_SIZE = 160
eval_class = "car"


# Create single view model 
with open("./quanmain/model_config/mcunetYolo_config.json", "r") as f:
    mcunetYolo_config = json.load(f)

checkpoint = torch.load("./runs/mcunet_S5_res160_pkg_lrdecay/best.pth", map_location='cpu')
wbf_result_path = "./colmain/decision_fusion/resultview3"
fixed_state_dict = fix_state_dict_keys(checkpoint['model'])
pytorch_model = ProxylessNASNets.build_from_config(mcunetYolo_config)
pytorch_model.eval()
pytorch_model.load_state_dict(fixed_state_dict)
pytorch_model.to(DEVICE) 


# Loading Occlusion Data 
base_path = "./dataset/occlusion/co3d/car"
with open(os.path.join(base_path, "manifest_with_occ.json"), 'r') as f:
    manifest = json.load(f)

occlusion_combination = [
    ("occ30", "occ30", "occ30"),
    ("occ30", "occ30", "occ50"), 
    ("occ30", "occ50", "occ50"),
    ("occ50", "occ50", "occ50"),
    # ("occ30", "occ70"), 
    # ("occ50", "occ30"),
    # ("occ50", "occ50"),
    # ("occ50", "occ70"), 
    # ("occ70", "occ30"),
    # ("occ70", "occ50"),
    # ("occ70", "occ70")
]

for occ_comb1, occ_comb2, occ_comb3 in occlusion_combination:
    print(f"\n ======== Processing {occ_comb1} vs {occ_comb2} vs {occ_comb3} ========")

    # Collect all image and ground truthes paths for both views
    image_paths_view1 = []
    image_paths_view2 = []
    image_paths_view3 = []
    all_gt1 = []
    all_gt2 = []
    all_gt3 = []

    # Process pairs of images
    for i in range(0, len(manifest) - 2, 3): # Process pairs 
        item1 = manifest[i]
        if (i + 1) <= len(manifest) - 1 and (i + 2) <= len(manifest) - 1:
            item2 = manifest[i + 1]
            item3 = manifest[i + 2]
        else:
            break

        filename1 = item1[occ_comb1]
        filename2 = item2[occ_comb2]
        filename3 = item3[occ_comb3]
        filename1_path = os.path.join(base_path, filename1)
        filename2_path = os.path.join(base_path, filename2)
        filename3_path = os.path.join(base_path, filename3)
        
        # Check if files exist
        if os.path.exists(filename1_path) and os.path.exists(filename2_path) and os.path.exists(filename3_path):
            image_paths_view1.append(filename1_path)
            image_paths_view2.append(filename2_path)
            image_paths_view3.append(filename3_path)
            gt1 = item1['bbox'] + [VOC_CLASS_TO_IDX[eval_class]]
            gt2 = item2['bbox'] + [VOC_CLASS_TO_IDX[eval_class]]
            gt3 = item3['bbox'] + [VOC_CLASS_TO_IDX[eval_class]]
            
            all_gt1.append(gt1) # one image contains one ground truth
            all_gt2.append(gt2)
            all_gt3.append(gt3)

    print(f"Processing {len(image_paths_view1)} image pairs")

    # Create batches for bot views (5 images per batch)
    all_images1, resized_gt1 = process_image(image_paths_view1, all_gt1, image_size=IMAGE_SIZE)
    all_images2, resized_gt2 = process_image(image_paths_view2, all_gt2, image_size=IMAGE_SIZE)
    all_images3, resized_gt3 = process_image(image_paths_view3, all_gt3, image_size=IMAGE_SIZE)

    all_images1 = all_images1.to(DEVICE)
    all_images2 = all_images2.to(DEVICE)
    all_images3 = all_images3.to(DEVICE)

    print("Getting predictions for view 1 and view 2 and view 3..")
    with torch.no_grad():
        preds1 = pytorch_model(all_images1) # [All images, A*(5+C), S, S]
        preds2 = pytorch_model(all_images2) # [All images, A*(5+C), S, S]
        preds3 = pytorch_model(all_images3) # [All images, A*(5+C), S, S]
        

    # Decode predictions for both views 
    print("Decoding predictions...")

    decoded_preds1 = decode_pred(
        preds1, anchors=torch.tensor(VOC_ANCHORS, dtype= torch.float32), num_classes=len(VOC_CLASSES), 
        image_size=IMAGE_SIZE, conf_thresh= 0.001
    )

    decoded_preds2 = decode_pred(
        preds2, anchors=torch.tensor(VOC_ANCHORS, dtype= torch.float32), num_classes=len(VOC_CLASSES),
        image_size=IMAGE_SIZE, conf_thresh= 0.001
    )

    decoded_preds3 = decode_pred(
        preds3, anchors=torch.tensor(VOC_ANCHORS, dtype= torch.float32), num_classes=len(VOC_CLASSES),
        image_size=IMAGE_SIZE, conf_thresh= 0.001
    )


    class_preds1 = filter_class_only(decoded_preds1, VOC_CLASS_TO_IDX[eval_class])
    class_preds2 = filter_class_only(decoded_preds2, VOC_CLASS_TO_IDX[eval_class])
    class_preds3 = filter_class_only(decoded_preds3, VOC_CLASS_TO_IDX[eval_class])
    
    # Evaluate single view
    print("Evaluateing Single View ...")
    print("Evaluating Occlusion Rate 1 ...")
    results1 = eval_metric_col(class_preds1, resized_gt1, device=DEVICE)
    ap_1, recall_1, precision_1 = eval_metric_col_handcraft(class_preds1, resized_gt1, iou_threshold=0.5)
    print("Evaluating Occlusion Rate 2 ...")
    results2 = eval_metric_col(class_preds2, resized_gt2, device=DEVICE)
    ap_2, recall_2, precision_2 = eval_metric_col_handcraft(class_preds2, resized_gt2, iou_threshold=0.5)
    print("Evaluating Occlusion Rate 3 ...")
    results3 = eval_metric_col(class_preds3, resized_gt3, device=DEVICE)
    ap_3, recall_3, precision_3 = eval_metric_col_handcraft(class_preds3, resized_gt3, iou_threshold=0.5)


    # Evaluate decision fusion (WBF)
    print("Applying Weighted Box Fusion for each class ... ")
    wbf_class_preds1 = []
    wbf_class_preds_2view = []

    # For each image 
    for i in range(len(class_preds1)):
        combined_boxes1 = []
        combined_boxes2view = []

        # Integrate all boxes in occ-rate view 1 and 2 
        # Prepare lists for WBF
        for box1 in class_preds1[i]:
            xmin, ymin, xmax, ymax, conf, class_id = box1
            combined_boxes1.append([
                (xmin/IMAGE_SIZE).item(), (ymin/IMAGE_SIZE).item(), 
                (xmax/IMAGE_SIZE).item(), (ymax/IMAGE_SIZE).item(), 
                conf.item(), int(class_id.item())])
            combined_boxes2view.append([
                (xmin/IMAGE_SIZE).item(), (ymin/IMAGE_SIZE).item(), 
                (xmax/IMAGE_SIZE).item(), (ymax/IMAGE_SIZE).item(), 
                conf.item(), int(class_id.item())])
        for box2 in class_preds2[i]:
            xmin, ymin, xmax, ymax, conf, class_id = box2
            combined_boxes1.append([
                (xmin/IMAGE_SIZE).item(), (ymin/IMAGE_SIZE).item(), 
                (xmax/IMAGE_SIZE).item(), (ymax/IMAGE_SIZE).item(), 
                conf.item(), int(class_id.item())])
            combined_boxes2view.append([
                (xmin/IMAGE_SIZE).item(), (ymin/IMAGE_SIZE).item(), 
                (xmax/IMAGE_SIZE).item(), (ymax/IMAGE_SIZE).item(), 
                conf.item(), int(class_id.item())])
        for box3 in class_preds3[i]:
            xmin, ymin, xmax, ymax, conf, class_id = box3
            combined_boxes1.append([
                (xmin/IMAGE_SIZE).item(), (ymin/IMAGE_SIZE).item(), 
                (xmax/IMAGE_SIZE).item(), (ymax/IMAGE_SIZE).item(), 
                conf.item(), int(class_id.item())])

        # 3 views fusion
        if combined_boxes1:
            fused_boxes1, fused_scores1, fused_classes1 = weighted_boxes_fusion(
                combined_boxes1, iou_thr=0.55, n_models=3
            )   

            # Convert back to pixel coordinates
            wbf_result1 = []
            for j in range(len(fused_boxes1)):
                box = fused_boxes1[j]
                xmin, ymin, xmax, ymax = box[0]*IMAGE_SIZE, box[1]*IMAGE_SIZE, box[2]*IMAGE_SIZE, box[3]*IMAGE_SIZE
                wbf_result1.append([xmin, ymin, xmax, ymax, fused_scores1[j], int(fused_classes1[j])])

            # Convert back to Tensor
            wbf_class_preds1.append(torch.tensor(wbf_result1, dtype=torch.float32, device=DEVICE))
        
        else:
            # Convert back to Tensor
            wbf_class_preds1.append(torch.empty((0, 6), dtype=torch.float32, device=DEVICE))

        # 2 views fusion
        if combined_boxes2view:
            fused_boxes2view, fused_scores2view, fused_classes2view = weighted_boxes_fusion(
                combined_boxes2view, iou_thr=0.55, n_models=3
            )   

            # Convert back to pixel coordinates
            wbf_result2view = []
            for j in range(len(fused_boxes2view)):
                box = fused_boxes2view[j]
                xmin, ymin, xmax, ymax = box[0]*IMAGE_SIZE, box[1]*IMAGE_SIZE, box[2]*IMAGE_SIZE, box[3]*IMAGE_SIZE
                wbf_result2view.append([xmin, ymin, xmax, ymax, fused_scores2view[j], int(fused_classes2view[j])])

            # Convert back to Tensor
            wbf_class_preds_2view.append(torch.tensor(wbf_result2view, dtype=torch.float32, device=DEVICE))
        
        else:
            # Convert back to Tensor
            wbf_class_preds_2view.append(torch.empty((0, 6), dtype=torch.float32, device=DEVICE))


    print("Evaluateing WBF ...")
    print("Evaluating WBF Occlusion Rate 1 ...")
    wbf_results1 = eval_metric_col(wbf_class_preds1, resized_gt1, device=DEVICE)
    wbf_ap_1, wbf_recall_1, wbf_precision_1 = eval_metric_col_handcraft(wbf_class_preds1, resized_gt1, iou_threshold=0.5, device=DEVICE)
    plot_pr_curves_comp(
        recall_vals_list=[recall_1, wbf_recall_1],
        precision_vals_list=[precision_1, wbf_precision_1],
        ap_list=[ap_1, wbf_ap_1],
        save_path= os.path.join(wbf_result_path, f"pr_curve_comp_view1_{occ_comb1}_{occ_comb2}_{occ_comb3}.png")
    )
    wbf2view_results1 = eval_metric_col(wbf_class_preds_2view, resized_gt1, device=DEVICE)
    wbf2view_ap_1, wbf2view_recall_1, wbf2view_precision_1 = eval_metric_col_handcraft(wbf_class_preds_2view, resized_gt1, iou_threshold=0.5, device=DEVICE)
    plot_pr_curves_comp(
        recall_vals_list=[recall_1, wbf2view_recall_1],
        precision_vals_list=[precision_1, wbf2view_precision_1],
        ap_list=[ap_1, wbf2view_ap_1],
        save_path= os.path.join(wbf_result_path, f"pr_curve_comp2view_view1_{occ_comb1}_{occ_comb2}_{occ_comb3}.png")
    )

    print("Evaluating WBF Occlusion Rate 2 ...")
    wbf_results2 = eval_metric_col(wbf_class_preds1, resized_gt2, device=DEVICE)
    wbf_ap_2, wbf_recall_2, wbf_precision_2 = eval_metric_col_handcraft(wbf_class_preds1, resized_gt2, iou_threshold=0.5, device=DEVICE)
    plot_pr_curves_comp(
        recall_vals_list=[recall_2, wbf_recall_2],
        precision_vals_list=[precision_2, wbf_precision_2],
        ap_list=[ap_2, wbf_ap_2],
        save_path= os.path.join(wbf_result_path, f"pr_curve_comp_view2_{occ_comb1}_{occ_comb2}_{occ_comb3}.png")
    )
    wbf2view_results2 = eval_metric_col(wbf_class_preds_2view, resized_gt2, device=DEVICE)
    wbf2view_ap_2, wbf2view_recall_2, wbf2view_precision_2 = eval_metric_col_handcraft(wbf_class_preds_2view, resized_gt2, iou_threshold=0.5, device=DEVICE)
    plot_pr_curves_comp(
        recall_vals_list=[recall_1, wbf2view_recall_2],
        precision_vals_list=[precision_1, wbf2view_precision_2],
        ap_list=[ap_1, wbf2view_ap_2],
        save_path= os.path.join(wbf_result_path, f"pr_curve_comp2view_view2_{occ_comb1}_{occ_comb2}_{occ_comb3}.png")
    )

    print("Evaluating WBF Occlusion Rate 3 ...")
    wbf_results3 = eval_metric_col(wbf_class_preds1, resized_gt3, device=DEVICE)
    wbf_ap_3, wbf_recall_3, wbf_precision_3 = eval_metric_col_handcraft(wbf_class_preds1, resized_gt3, iou_threshold=0.5, device=DEVICE)
    plot_pr_curves_comp(
        recall_vals_list=[recall_3, wbf_recall_3],
        precision_vals_list=[precision_3, wbf_precision_3],
        ap_list=[ap_3, wbf_ap_3],
        save_path= os.path.join(wbf_result_path, f"pr_curve_comp_view3_{occ_comb1}_{occ_comb2}_{occ_comb3}.png")
    )

    print(f"\n ======== Results for {occ_comb1} vs {occ_comb2} vs {occ_comb3} ========")
    print(f"View 1 ({occ_comb1}):")
    print(f" mAP: {results1['map']:.4f}")
    print(f" mAP@50: {results1['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ap_1}")

    print(f"WBF-3 View 1 ({occ_comb1}):")
    print(f" mAP: {wbf_results1['map']:.4f}")
    print(f" mAP@50: {wbf_results1['map_50']:.4f}")
    print(f" hand-craft mAP@50: {wbf_ap_1}")

    print(f"WBF-2 View 1 ({occ_comb1}):")
    print(f" mAP: {wbf2view_results1['map']:.4f}")
    print(f" mAP@50: {wbf2view_results1['map_50']:.4f}")
    print(f" hand-craft mAP@50: {wbf2view_ap_1}")

    print(f"View 2 ({occ_comb2}):")
    print(f" mAP: {results2['map']:.4f}")
    print(f" mAP@50: {results2['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ap_2}")

    print(f"WBF-3 View 2 ({occ_comb2}):")
    print(f" mAP: {wbf_results2['map']:.4f}")
    print(f" mAP@50: {wbf_results2['map_50']:.4f}")
    print(f" hand-craft mAP@50: {wbf_ap_2}")

    print(f"WBF-2 View 2 ({occ_comb1}):")
    print(f" mAP: {wbf2view_results2['map']:.4f}")
    print(f" mAP@50: {wbf2view_results2['map_50']:.4f}")
    print(f" hand-craft mAP@50: {wbf2view_ap_2}")

    print(f"View 3 ({occ_comb3}):")
    print(f" mAP: {results3['map']:.4f}")
    print(f" mAP@50: {results3['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ap_3}")

    print(f"WBF-3 View 3 ({occ_comb3}):")
    print(f" mAP: {wbf_results3['map']:.4f}")
    print(f" mAP@50: {wbf_results3['map_50']:.4f}")
    print(f" hand-craft mAP@50: {wbf_ap_3}")
