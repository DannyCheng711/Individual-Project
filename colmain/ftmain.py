import os
import torch
from PIL import Image
from config import VOC_ANCHORS, VOC_CLASSES, VOC_CLASS_TO_IDX
import json 
from dotenv import load_dotenv
from models.mcunetYolo.tinynas.nn.proxyless_net import ProxylessNASNets
from models.dethead.mvyolodet import MultiViewYolo
from utils.config_utils import fix_state_dict_keys
from utils.image_utils import process_image, filter_class_only
from validation.bboxprep import decode_pred
from validation.metrics import eval_metric_col, eval_metric_col_handcraft
from validation.visualization import visualize_bbox_grid_tensors, plot_pr_curves_comp
from .fusion import weighted_boxes_fusion

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device("cuda")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")
IMAGE_SIZE = 160

eval_class = "car"

with open("./nasmain/model_config/mcunetYolo_config.json", "r") as f:
    mcunetYolo_config = json.load(f)

checkpoint = torch.load("./runs/mcunet_S5_res160_pkg_lrdecay/best.pth", map_location='cpu')
print(" Converting PyTorch keys to PyTorch NAS keys...")

fixed_state_dict = fix_state_dict_keys(checkpoint['model'])

# PyTorch YOLO head
pytorch_model = ProxylessNASNets.build_from_config(mcunetYolo_config)
pytorch_model.eval()
pytorch_model.load_state_dict(fixed_state_dict)
pytorch_model.to(DEVICE) 

ft_state_dict = torch.load("./multiview_yolo_final.pth", map_location='cuda')

multiv_model = MultiViewYolo(
    num_classes=len(VOC_CLASSES), num_anchors=5, final_ch=320, 
    passthrough_ch=96, mid_ch=512).to(DEVICE)

# 3. Apply weights to model
multiv_model.load_state_dict(ft_state_dict)

# 4. Switch to eval mode
multiv_model.eval()


base_path = "./colmain/co3d/car"

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
        ft_pred = multiv_model(all_images1, all_images2)


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

    ft_decode_pred = decode_pred(
        ft_pred, anchors=torch.tensor(VOC_ANCHORS, dtype= torch.float32), num_classes=len(VOC_CLASSES),
        image_size=IMAGE_SIZE, conf_thresh= 0.001
    )

    class_preds1 = filter_class_only(decoded_preds1, VOC_CLASS_TO_IDX[eval_class])
    class_preds2 = filter_class_only(decoded_preds2, VOC_CLASS_TO_IDX[eval_class])
    ft_class_preds = filter_class_only(ft_decode_pred, VOC_CLASS_TO_IDX[eval_class])
    
    # Evaluate occ rate 1 
    print("Evaluating Occlusion Rate 1 ...")
    results1 = eval_metric_col(class_preds1, resized_gt1, device=DEVICE)
    ap_1, recall_1, precision_1 = eval_metric_col_handcraft(class_preds1, resized_gt1, iou_threshold=0.5)
    print("Evaluating Occlusion Rate 2 ...")
    results2 = eval_metric_col(class_preds2, resized_gt2, device=DEVICE)
    ap_2, recall_2, precision_2 = eval_metric_col_handcraft(class_preds2, resized_gt2, iou_threshold=0.5)

    
    print("Evaluating Occlusion Rate 1 ...")
    fused_results1 = eval_metric_col(ft_class_preds, resized_gt1, device=DEVICE)
    fused_ap_1, fused_recall_1, fused_precision_1 = eval_metric_col_handcraft(ft_class_preds, resized_gt1, iou_threshold=0.5)
    print("Evaluating Occlusion Rate 2 ...")
    fused_results2 = eval_metric_col(ft_class_preds, resized_gt2, device=DEVICE)
    fused_ap_2, fused_recall_2, fused_precision_2 = eval_metric_col_handcraft(ft_class_preds, resized_gt2, iou_threshold=0.5)


    print(f"\n ======== Results for {occ_comb1} vs {occ_comb2} ========")
    print(f"View 1 ({occ_comb1}):")
    print(f" mAP: {results1['map']:.4f}")
    print(f" mAP@50: {results1['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ap_1}")

    print(f"Feature Map Fused View 1 ({occ_comb1}):")
    print(f" mAP: {fused_results1['map']:.4f}")
    print(f" mAP@50: {fused_results1['map_50']:.4f}")
    print(f" hand-craft mAP@50: {fused_ap_1}")

    print(f"View 2 ({occ_comb2}):")
    print(f" mAP: {results2['map']:.4f}")
    print(f" mAP@50: {results2['map_50']:.4f}")
    print(f" hand-craft mAP@50: {ap_2}")

    print(f"Feature Map Fused View 2 ({occ_comb2}):")
    print(f" mAP: {fused_results2['map']:.4f}")
    print(f" mAP@50: {fused_results2['map_50']:.4f}")
    print(f" hand-craft mAP@50: {fused_ap_2}")

    visualize_bbox_grid_tensors(
        all_images1, all_images2,
        class_preds1, ft_class_preds,
        class_preds2, ft_class_preds,
        save_path=f"./colmain/ft_result/bbox_grid_{occ_comb1}_{occ_comb2}.png"
    )

    