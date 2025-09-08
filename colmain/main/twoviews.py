import os
import torch
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
wbf_result_path = "./colmain/decision_fusion/result"
fixed_state_dict = fix_state_dict_keys(checkpoint['model'])
pytorch_model = ProxylessNASNets.build_from_config(mcunetYolo_config)
pytorch_model.eval()
pytorch_model.load_state_dict(fixed_state_dict)
pytorch_model.to(DEVICE) 


# Create feature fusion model 
ff_base_path = "./colmain/feature_fusion/runs/separate_fusion"
ff_result_path = "./colmain/feature_fusion/result"
ff_state_dict = torch.load(os.path.join(ff_base_path, "best.pth"), map_location='cuda')
ff_model = MultiViewMcuYolo(
    num_classes=len(VOC_CLASSES), num_anchors=5, final_ch=320, 
    passthrough_ch=96, mid_ch=512).to(DEVICE)
ff_model.load_state_dict(ff_state_dict['model'])
ff_model.eval()


# Loading Occlusion Data 
base_path = "./dataset/occlusion/co3d/car"
with open(os.path.join(base_path, "manifest_with_occ.json"), 'r') as f:
    manifest = json.load(f)

occlusion_combination = [
    ("occ30", "occ30"),
    ("occ30", "occ50"), 
    ("occ30", "occ70"), 
    ("occ50", "occ30"),
    ("occ50", "occ50"),
    ("occ50", "occ70"), 
    ("occ70", "occ30"),
    ("occ70", "occ50"),
    ("occ70", "occ70")
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
        filename1_path = os.path.join(base_path, filename1)
        filename2_path = os.path.join(base_path, filename2)
        
        # Check if files exist
        if os.path.exists(filename1_path) and os.path.exists(filename2_path):
            image_paths_view1.append(filename1_path)
            image_paths_view2.append(filename2_path)
            gt1 = item1['bbox'] + [VOC_CLASS_TO_IDX[eval_class]]
            gt2 = item2['bbox'] + [VOC_CLASS_TO_IDX[eval_class]]
            
            all_gt1.append(gt1) # one image contains one ground truth
            all_gt2.append(gt2)

    print(f"Processing {len(image_paths_view1)} image pairs")

    # Create batches for bot views (5 images per batch)
    all_images1, resized_gt1 = process_image(image_paths_view1, all_gt1, image_size=IMAGE_SIZE)
    all_images2, resized_gt2 = process_image(image_paths_view2, all_gt2, image_size=IMAGE_SIZE)

    all_images1 = all_images1.to(DEVICE)
    all_images2 = all_images2.to(DEVICE)

    print("Getting predictions for view 1 and view 2...")
    with torch.no_grad():
        preds1 = pytorch_model(all_images1) # [All images, A*(5+C), S, S]
        preds2 = pytorch_model(all_images2) # [All images, A*(5+C), S, S]
        ff_pred, _ = ff_model(all_images1, all_images2)

    # Decode predictions for both views 
    print("Decoding predictions...")
    # [ 
    #   [ # images
    #       # detections [x1, x2, y1, y2, conf, classid]]] images, detections
    decoded_preds1 = decode_pred(
        preds1, anchors=torch.tensor(VOC_ANCHORS, dtype= torch.float32), num_classes=len(VOC_CLASSES), 
        image_size=IMAGE_SIZE, conf_thresh= 0.001
    )

    decoded_preds2 = decode_pred(
        preds2, anchors=torch.tensor(VOC_ANCHORS, dtype= torch.float32), num_classes=len(VOC_CLASSES),
        image_size=IMAGE_SIZE, conf_thresh= 0.001
    )

    ff_decode_pred = decode_pred(
        ff_pred, anchors=torch.tensor(VOC_ANCHORS, dtype= torch.float32), num_classes=len(VOC_CLASSES),
        image_size=IMAGE_SIZE, conf_thresh= 0.001
    )

    class_preds1 = filter_class_only(decoded_preds1, VOC_CLASS_TO_IDX[eval_class])
    class_preds2 = filter_class_only(decoded_preds2, VOC_CLASS_TO_IDX[eval_class])
    ff_class_preds = filter_class_only(ff_decode_pred, VOC_CLASS_TO_IDX[eval_class])
    
    # Evaluate single view
    print("Evaluateing Single View ...")
    print("Evaluating Occlusion Rate 1 ...")
    results1 = eval_metric_col(class_preds1, resized_gt1, device=DEVICE)
    ap_1, recall_1, precision_1 = eval_metric_col_handcraft(class_preds1, resized_gt1, iou_threshold=0.5)
    print("Evaluating Occlusion Rate 2 ...")
    results2 = eval_metric_col(class_preds2, resized_gt2, device=DEVICE)
    ap_2, recall_2, precision_2 = eval_metric_col_handcraft(class_preds2, resized_gt2, iou_threshold=0.5)

    # Evaluate decision fusion (WBF)
    print("Applying Weighted Box Fusion for each class ... ")
    wbf_class_preds = []

    # For each image 
    for i in range(len(class_preds1)):
        combined_boxes = []

        # Integrate all boxes in occ-rate view 1 and 2 
        # Prepare lists for WBF
        for box1 in class_preds1[i]:
            xmin, ymin, xmax, ymax, conf, class_id = box1
            combined_boxes.append([
                (xmin/IMAGE_SIZE).item(), (ymin/IMAGE_SIZE).item(), 
                (xmax/IMAGE_SIZE).item(), (ymax/IMAGE_SIZE).item(), 
                conf.item(), int(class_id.item())])
        for box2 in class_preds2[i]:
            xmin, ymin, xmax, ymax, conf, class_id = box2
            combined_boxes.append([
                (xmin/IMAGE_SIZE).item(), (ymin/IMAGE_SIZE).item(), 
                (xmax/IMAGE_SIZE).item(), (ymax/IMAGE_SIZE).item(), 
                conf.item(), int(class_id.item())])


        if combined_boxes:
            fused_boxes, fused_scores, fused_classes = weighted_boxes_fusion(
                combined_boxes, iou_thr=0.55, n_models=2
            )   

            # Convert back to pixel coordinates
            wbf_result = []
            for j in range(len(fused_boxes)):
                box = fused_boxes[j]
                xmin, ymin, xmax, ymax = box[0]*IMAGE_SIZE, box[1]*IMAGE_SIZE, box[2]*IMAGE_SIZE, box[3]*IMAGE_SIZE
                wbf_result.append([xmin, ymin, xmax, ymax, fused_scores[j], int(fused_classes[j])])

            # Convert back to Tensor
            wbf_class_preds.append(torch.tensor(wbf_result, dtype=torch.float32, device=DEVICE))
        
        else:
            # Convert back to Tensor
            wbf_class_preds.append(torch.empty((0, 6), dtype=torch.float32, device=DEVICE))


    print("Evaluateing WBF ...")
    print("Evaluating WBF Occlusion Rate 1 ...")
    wbf_results1 = eval_metric_col(wbf_class_preds, resized_gt1, device=DEVICE)
    wbf_ap_1, wbf_recall_1, wbf_precision_1 = eval_metric_col_handcraft(wbf_class_preds, resized_gt1, iou_threshold=0.5, device=DEVICE)
    plot_pr_curves_comp(
        recall_vals_list=[recall_1, wbf_recall_1],
        precision_vals_list=[precision_1, wbf_precision_1],
        ap_list=[ap_1, wbf_ap_1],
        save_path= os.path.join(wbf_result_path, f"pr_curve_comp_view1_{occ_comb1}_{occ_comb2}.png")
    )
    print("Evaluating WBF Occlusion Rate 2 ...")
    wbf_results2 = eval_metric_col(wbf_class_preds, resized_gt2, device=DEVICE)
    wbf_ap_2, wbf_recall_2, wbf_precision_2 = eval_metric_col_handcraft(wbf_class_preds, resized_gt2, iou_threshold=0.5, device=DEVICE)
    plot_pr_curves_comp(
        recall_vals_list=[recall_2, wbf_recall_2],
        precision_vals_list=[precision_2, wbf_precision_2],
        ap_list=[ap_2, wbf_ap_2],
        save_path= os.path.join(wbf_result_path, f"pr_curve_comp_view2_{occ_comb1}_{occ_comb2}.png")
    )

    print("Evaluateing FF ...")
    print("Evaluating Feature Fusion Occlusion Rate 1 ...")
    ff_results1 = eval_metric_col(ff_class_preds, resized_gt1, device=DEVICE)
    ff_ap_1, fused_recall_1, fused_precision_1 = eval_metric_col_handcraft(ff_class_preds, resized_gt1, iou_threshold=0.5)
    plot_pr_curves_comp(
        recall_vals_list=[recall_2, wbf_recall_2],
        precision_vals_list=[precision_2, wbf_precision_2],
        ap_list=[ap_2, wbf_ap_2],
        save_path= os.path.join(ff_result_path, f"pr_curve_comp_view1_{occ_comb1}_{occ_comb2}.png")
    )
    print("Evaluating Feature Fusion Occlusion Rate 2 ...")
    ff_results2 = eval_metric_col(ff_class_preds, resized_gt2, device=DEVICE)
    ff_ap_2, fused_recall_2, fused_precision_2 = eval_metric_col_handcraft(ff_class_preds, resized_gt2, iou_threshold=0.5)
    plot_pr_curves_comp(
        recall_vals_list=[recall_2, wbf_recall_2],
        precision_vals_list=[precision_2, wbf_precision_2],
        ap_list=[ap_2, wbf_ap_2],
        save_path= os.path.join(ff_result_path, f"pr_curve_comp_view2_{occ_comb1}_{occ_comb2}.png")
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

    print(f"Feature Fusion View 1 ({occ_comb1}):")
    print(f" mAP: {ff_results1['map']:.4f}")
    print(f" mAP@50: {ff_results1['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ff_ap_1}")

    print(f"View 2 ({occ_comb2}):")
    print(f" mAP: {results2['map']:.4f}")
    print(f" mAP@50: {results2['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ap_2}")

    print(f"WBF View 2 ({occ_comb2}):")
    print(f" mAP: {wbf_results2['map']:.4f}")
    print(f" mAP@50: {wbf_results2['map_50']:.4f}")
    print(f" hand-craft mAP@50: {wbf_ap_2}")

    print(f"Feature Fusion View 2 ({occ_comb1}):")
    print(f" mAP: {ff_results2['map']:.4f}")
    print(f" mAP@50: {ff_results2['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ff_ap_2}")

    visualize_bbox_grid_tensors(
        all_images1, all_images2,
        class_preds1, wbf_class_preds,
        class_preds2, wbf_class_preds,
        save_path= os.path.join(wbf_result_path, f"bbox_grid_{occ_comb1}_{occ_comb2}.png")
    )

    visualize_bbox_grid_tensors(
        all_images1, all_images2,
        class_preds1, ff_class_preds,
        class_preds2, ff_class_preds,
        save_path= os.path.join(ff_result_path, f"bbox_grid_{occ_comb1}_{occ_comb2}.png")
    )


