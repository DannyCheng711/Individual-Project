import torch
from torchvision.ops import nms

def target_tensor_to_gt(target_tensor, image_size):
    """
    Convert a YOLO-style target tensor [S, S, A, 5+C] to GT boxes in pixel xyxy format.
    
    target_tensor: Tensor [S, S, A, 5+C]
      tx, ty: offsets in cell (0~1)
      tw, th: width/height in grid units
      obj: objectness (1 if positive)
      one-hot classes after index 5
    image_size: int, assuming square images
    """
    S = target_tensor.shape[0]  # grid size (assume Sx=Sy)
    
    gt_boxes = []
    gt_labels = []

    for i_grid in range(S):
        for j_grid in range(S):
            for a_idx in range(target_tensor.shape[2]):
                if target_tensor[i_grid, j_grid, a_idx, 4] == 1:
                    tx = target_tensor[i_grid, j_grid, a_idx, 0].item()
                    ty = target_tensor[i_grid, j_grid, a_idx, 1].item()
                    tw = target_tensor[i_grid, j_grid, a_idx, 2].item()
                    th = target_tensor[i_grid, j_grid, a_idx, 3].item()

                    cx = (j_grid + tx) / S
                    cy = (i_grid + ty) / S
                    w  = tw / S
                    h  = th / S

                    x_min = (cx - w / 2) * image_size
                    y_min = (cy - h / 2) * image_size
                    x_max = (cx + w / 2) * image_size
                    y_max = (cy + h / 2) * image_size

                    class_vector = target_tensor[i_grid, j_grid, a_idx, 5:]
                    class_id = torch.argmax(class_vector).item()

                    gt_boxes.append([x_min, y_min, x_max, y_max])
                    gt_labels.append(class_id)

    return torch.tensor(gt_boxes, dtype=torch.float32), torch.tensor(gt_labels, dtype=torch.long)

def xyxy_to_cxcywh_norm(xyxy, image_size):
    """
    Convert pixel xyxy -> normalized cxcywh (divide by image_size).
    xyxy: (N,4) tensor
    """
    x1, y1, x2, y2 = xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5 / image_size
    cy = (y1 + y2) * 0.5 / image_size
    w  = (x2 - x1) / image_size
    h  = (y2 - y1) / image_size
    return torch.stack([cx, cy, w, h], dim=-1)

def norm_cxcywh_to_xyxy_pixels(cxcywh, image_size):
    """
    cxcywh in [0,1] (normalized), returns xyxy in pixels (clamped to [0, image_size]).
    cxcywh: (..., 4) tensor
    """
    cx, cy, w, h = cxcywh.unbind(-1)
    x1 = (cx - 0.5 * w) * image_size
    y1 = (cy - 0.5 * h) * image_size
    x2 = (cx + 0.5 * w) * image_size
    y2 = (cy + 0.5 * h) * image_size
    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return torch.clamp(xyxy, min=0.0, max=float(image_size))

def decode_pred(pred, anchors, num_classes, image_size, conf_thresh = 0.0):
    """
    Decode YOLO predictions into bounding boxes.
    Args:
        pred: Tensor of shape [B, A*(5+C), S, S] 
        anchors: tensor of shape [A, 2] in grid units 
        num_classes: number of classes
        image_size: pixel width/height of input image
        conf_thresh: objectness threshold

    Returns:
        List[box]: a list of [xmin, ymin, xmax, ymax, confidence]
    """

    B, _, S, _ = pred.shape  # [B, A*(5+C), S, S]
    A = len(anchors)
    C = num_classes

    # Reshape to [B, A, 5+C, S, S] then permute to [B, S, S, A, 5+C]
    pred = pred.reshape(B, A, 5 + C, S, S)
    pred = pred.permute(0, 3, 4, 1, 2)  # [B, S, S, A, 5+C]
    
    # Extract components
    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    obj_logit = pred[..., 4]
    cls_logits = pred[..., 5:] 

    sig_tx = torch.sigmoid(tx)
    sig_ty = torch.sigmoid(ty)
    exp_tw = torch.exp(tw)
    exp_th = torch.exp(th)
    obj = torch.sigmoid(obj_logit)
    cls_probs = torch.softmax(cls_logits, dim=-1)

    anchors = anchors.to(device=pred.device, dtype=pred.dtype)

    all_decoded = []
    for b in range(B):
        decoded_boxes = [] 
        for i in range(S):
            for j in range(S):
                for a in range(A):
                    # final confidence = objectness * best class prob
                    best_cls_prob, best_cls_id = cls_probs[b, i, j, a].max(dim=-1) 
                    final_conf = obj[b, i, j, a] * best_cls_prob
                    
                    if final_conf.item() < conf_thresh:
                        continue
                    
                    # YOLO param -> normalized cx,cy,w,h
                    cx = (j + sig_tx[b, i, j, a]) / float(S)
                    cy = (i + sig_ty[b, i, j, a]) / float(S)
                    bw = (anchors[a, 0] * exp_tw[b, i, j, a]) / float(S)
                    bh = (anchors[a, 1] * exp_th[b, i, j, a]) / float(S)

                    # shared conversion helper → pixel xyxy (clamped)
                    cxcywh = torch.stack([cx, cy, bw, bh], dim=-1)  # shape (4,)
                    xyxy   = norm_cxcywh_to_xyxy_pixels(cxcywh, image_size) # shape (4,)
                    x1, y1, x2, y2 = map(float, xyxy.tolist())

                    decoded_boxes.append(
                        [x1, y1, x2, y2, float(final_conf.item()), int(best_cls_id.item())])

        all_decoded.append(decoded_boxes)

    return all_decoded

# surpass the bbox which iou_thresh > 0.5
def apply_classwise_nms(pred_boxes, iou_thresh=0.5):
    boxes = torch.tensor([[p[0], p[1], p[2], p[3]] for p in pred_boxes], dtype=torch.float32)
    scores = torch.tensor([p[4] for p in pred_boxes], dtype=torch.float32)
    labels = torch.tensor([p[5] for p in pred_boxes], dtype=torch.long)

    keep = []
    for cls in labels.unique():
        # select the class index for this class
        class_mask = (labels == cls)
        # Shape: (N, 1) -> as_tuple -> tuple of one tensors (with one row) 
        class_indices = torch.nonzero(class_mask, as_tuple=True)[0]

        # Run NMS for this class subset
        keep_idxs = nms(boxes[class_indices], scores[class_indices], iou_thresh)
        keep.append(class_indices[keep_idxs])

    # [tensor([1, 2]), tensor([5, 6, 7]), tensor([9])]
    # -> tensor([1, 2, 5, 6, 7, 9])
    keep = torch.cat(keep)
    return [pred_boxes[i] for i in keep]