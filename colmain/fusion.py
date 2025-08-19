import numpy as np

def iou(box1, box2):
    """
    Compute IoU between two boxes: [x1, y1, x2, y2]. 
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0 

# Works with Python lists for flexibility.
def weighted_boxes_fusion(pred_boxes, iou_thr=0.55, n_models=2):
    """
    Weighted Boxes Fusion(WBF)

    Args: 
        pred_boxes (list of lists): Each element is [[x1, y1, x2, y2, score, class_id], ...] from one model.
       
        iou_thr (float): IoU threshold for clustering
        n_models (int): Total number of models/devices
        conf_weight (str): 'linear', 'sqrt', 'square' weighting 

    Returns: 
        fused_boxes: Final list of fused boxes.
        fused_scores: Final list of fused scores.
    """

    # Step 1: Sort by conf descending
    all_boxes = sorted(pred_boxes, key=lambda x: -x[4])

    clusters = [] # L: list of clusters (each cluster = list of boxes)
    fused_boxes = [] # F: fused boxes (1 per cluster)
    fused_scores = []
    fused_classes = []

    # Step 2 & 3: Iterate over sorted boxes 
    for x1, y1, x2, y2, score, cls in all_boxes:
        box = [x1, y1, x2, y2]
        matched = False
        for idx, fused in enumerate(fused_boxes):
            # fuse only if same class
            if fused_classes[idx] == cls and iou(box, fused) > iou_thr:
                clusters[idx].append((box, score))
                matched = True
                break
        
        # create a new cluster
        if not matched:
            clusters.append([(box, score)])
            fused_boxes.append(box)
            fused_scores.append(score)
            fused_classes.append(cls)

        # recalculate fused box for each cluster 
        for idx, cluster in enumerate(clusters):
            boxes_arr = np.array([b for b, _ in cluster])
            scores_arr = np.array([s for _, s in cluster])
            weights =scores_arr # weight average
            x1 = np.sum(weights * boxes_arr[:, 0]) / np.sum(weights)
            y1 = np.sum(weights * boxes_arr[:, 1]) / np.sum(weights)
            x2 = np.sum(weights * boxes_arr[:, 2]) / np.sum(weights)
            y2 = np.sum(weights * boxes_arr[:, 3]) / np.sum(weights) 
            fused_boxes[idx] = [x1, y1, x2, y2]
            fused_scores[idx] = np.mean(scores_arr)

    
    # Confidence rescale 
    for idx, cluster in enumerate(clusters):
        T = len(cluster)
        fused_scores[idx] = fused_scores[idx] * min(T, n_models) / n_models

    return fused_boxes, fused_scores, fused_classes


        

