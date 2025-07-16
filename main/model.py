from mcunet.mcunet.model_zoo import net_id_list, build_model, download_tflite
from torchsummary import summary 
from config import DEVICE, DATASET_ROOT
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

print(net_id_list)  # the list of models in the model zoo

# detection head
# Input feature map: [B, in_channels, H, W]
class YoloHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1) # 1x1 conv on each cell

    def forward(self, x):
        return self.detector(x).permute(0, 2, 3, 1).contiguous()

# feature map  
class McuYolo(nn.Module):

    def __init__(self, num_classes=20, num_anchors = 5):
        super().__init__()
        self.backbone, _, _ = build_model(net_id="mcunet-in4", pretrained=True)

        self.passthrough_layer_idx = 12 # block 12: 10×10×96
        self.final_block_idx = 16 # block 16: 5×5×320

        # extra 3 conv layer: conv1 and conv2
        self.conv1 = nn.Conv2d(320, 512, kernel_size = 1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size = 1)

        # space to depth
        self.space_to_depth = SpaceToDepth() # 10×10×96 -> 5×5×384

        # Concatenate passthrough + final, reduce channels to 512
        ## self.concat_conv = nn.Conv2d(384 + 512, 512, kernel_size=1)

        # extra 3 conv layer: conv3
        self.conv3 = nn.Conv2d(512 + 384, 512, kernel_size=1)  # concat + compress

        # add detection head
        self.det_head = YoloHead(in_channels = 512, num_classes = num_classes, num_anchors=num_anchors)

    def forward(self, x):

        x = self.backbone.first_conv(x)
        # forward each block manually via for loop 
        # each block is a subclass of nn.Module, and doing block(x) is equivalent to block.forward(x) 
        passthrough_feat = None
        for i in range(self.final_block_idx + 1):
            x = self.backbone.blocks[i](x)
            if i == self.passthrough_layer_idx:
                passthrough_feat = x 

        # head conv layers
        x = self.conv1(x) 
        x = self.conv2(x) 

        passthrough = self.space_to_depth(passthrough_feat) 

        # concatenate
        x = torch.cat([x, passthrough], dim = 1)
        x = self.conv3(x) 

        # final detection 
        return self.det_head(x)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False 
        print("Backbone frozen ... ")

    def unfreeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            params.requires_grad = True
        print("Backbone unfrozen ...")

# using sequential for creating the model 
class SpaceToDepth(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size 

    # x is the previous output
    def forward(self, x):
        batch, channel, height, width = x.size()

        # only continue if both height and width are divisible by the block size
        assert height % self.block_size == 0 and width % self.block_size == 0

        new_channel = channel * (self.block_size ** 2)
        new_height = height // self.block_size
        new_width = width // self.block_size

        # reshape to ensure the conversion strategy is correct 
        x = x.reshape(batch, channel, new_height, self.block_size, new_width, self.block_size)
        x = x.permute(0, 3, 5, 1, 2, 4).reshape(batch, new_channel, new_height, new_width)

        return x

# Loss function
class Yolov2Loss(nn.Module):
    def __init__(self, num_classes, anchors, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors 
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.count = 0

    def calculate_iou(self, ground_box, pred_box):
        """
        Compute IoU between two boxes in [cx, cy, w, h] format.
        All coordinates should be in the same unit (e.g., grid cells).
        """

        cx1, cy1, w1, h1 = ground_box
        cx2, cy2, w2, h2 = pred_box

        # convert center format to corner format 
        x1_min = cx1 - w1 / 2
        y1_min = cy1 - h1 / 2
        x1_max = cx1 + w1 / 2
        y1_max = cy1 + h1 / 2
        
        x2_min = cx2 - w2 / 2
        y2_min = cy2 - h2 / 2
        x2_max = cx2 + w2 / 2
        y2_max = cy2 + h2 / 2

        # Compute intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    def calculate_iou_wh(self, ground_box, anchor_box):
        """
        Compute IoU between two boxes using only [w, h] dimensions.
        Assumes both boxes are centered at the same location (e.g., (0, 0)).
        Used for anchor shape matching only.
        """

        _, _, w1, h1 = ground_box
        _, _, w2, h2 = anchor_box

        # Compute intersection
        inter_w = min(w1, w2)
        inter_h = min(h1, h2)
        inter_area = inter_w * inter_h

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    
    def forward(self, predictions, targets, imgs = None):

        # Step 1: reshape predictions (B, S, S, A*(5+C)) → (B, A, S, S, 5+C)
        batch_size, sh, sw, pred_dim = predictions.shape
        num_anchors = len(self.anchors)

        # Make sure the channel dimension matches
        expected_dim = num_anchors * (5 + self.num_classes)
        assert pred_dim == expected_dim, f"Expected {expected_dim}, got {pred_dim}"

        # Reshape and permute to [B, A, S, S, 5+C]
        predictions= predictions.reshape(batch_size, sh, sw, num_anchors, 5 + self.num_classes)
        predictions = predictions.permute(0, 3, 1, 2, 4).contiguous()  # → [B, A, S, S, 5 + C] and make it continuous in memory location
        
        # Step 2: extract tx, ty, tw, th, obj_score, class_probs
        tx = predictions[..., 0]
        ty = predictions[..., 1]
        tw = predictions[..., 2]
        th = predictions[..., 3]
        obj_score = predictions[..., 4]
        class_probs = predictions[..., 5:]

        # Anchors
        anchor_tensor = torch.tensor(self.anchors, device = DEVICE)
        # widths: tensor([1.0, 2.0, 1.5]), reshape -1 : total number of elements
        anchor_w = anchor_tensor[:, 0].reshape(1, -1, 1, 1) # [[[[1.0]], [[2.0]], [[1.5]]]] 
        # heights: tensor([2.0, 1.0, 1.5])
        anchor_h = anchor_tensor[:, 1].reshape(1, -1, 1, 1)

        # decode predicted box (using tensor)
        # bx, by  in [0,1], within current grid cell
        ## bx = torch.sigmoid(tx) 
        ## by = torch.sigmoid(ty) 
        # bw, bh  in [0,1], grid cell units 
        ## bw = torch.exp(tw) * anchor_w 
        ## bh = torch.exp(th) * anchor_h 
        obj_conf = torch.sigmoid(obj_score) # IOU
        
        
        # Step 3: compute objectness mask (1^obj_ij) and noobj mask
        # [B, A, S, S]
        obj_mask = torch.zeros(
            (batch_size, num_anchors, sh, sw), dtype=torch.bool, device = DEVICE)
        no_obj_mask = torch.ones(
            (batch_size, num_anchors, sh, sw), dtype=torch.bool, device = DEVICE)

        # These store the ground truth targets (same shape as prediction maps)
        ground_gx = torch.zeros((batch_size, num_anchors, sh, sw), device = DEVICE)
        ground_gy = torch.zeros_like(ground_gx, device = DEVICE)
        ground_gw = torch.zeros_like(ground_gx, device = DEVICE)
        ground_gh = torch.zeros_like(ground_gx, device = DEVICE)
        iou_map = torch.zeros_like(ground_gx, device = DEVICE)
        ground_class = torch.zeros((batch_size, num_anchors, sh, sw), dtype=torch.long, device = DEVICE)  # for class labels

        pred_bx = torch.zeros((batch_size, num_anchors, sh, sw), device = DEVICE)
        pred_by = torch.zeros_like(pred_bx, device = DEVICE)
        pred_bw = torch.zeros_like(pred_bx, device = DEVICE)
        pred_bh = torch.zeros_like(pred_bx, device = DEVICE)

        pos_count = 0
        saved_debug_count = 0

        for b in range(batch_size):
            # targets[b]: List[Tensor[num_objects, 5]] 
            # [x_center, y_center, width, height, class_id]
            for gt in targets[b]: 
                gx, gy, gw, gh, gt_cls = gt.tolist() # [0, 1] w.r.t image size

                # grid normalise
                gt_box = [0, 0, gw * sw, gh * sh] # box size in grid units
                anchor_ious = []

                for anchor in self.anchors:
                    anchor_box = [0, 0, anchor[0], anchor[1]]
                    iou = self.calculate_iou_wh(gt_box, anchor_box)
                    anchor_ious.append(iou)

                # assign anchor with highest IoU
                best_a = torch.argmax(torch.tensor(anchor_ious, device = DEVICE)).item() # argmax output is tensor(2), use item() extracts the value
                max_iou = anchor_ious[best_a]
                
                # print(f"[DEBUG] GT: w={gw*sw:.2f}, h={gh*sh:.2f}, Best IOU={max_iou:.3f}, Anchor ID={best_a}")
                if max_iou > 0.5:
                    pos_count += 1
                elif saved_debug_count < 10:
                    img_pil = transforms.functional.to_pil_image(imgs[b].cpu())
                    # visualise_iou_anchor(img_pil, gt, self.anchors[best_a], save_path =  f"./test/anchor_debug_train_{self.count}.jpg")
                    saved_debug_count += 1
                    self.count += 1

                # convert into which grid cell
                gi = int(gx * sw) 
                gj = int(gy * sh) 
                # set masks
                obj_mask[b, best_a, gj, gi] = True
                no_obj_mask[b, best_a, gj, gi] = False

                # Fill target maps (use fractional offset for gx, gy)
                ground_gx[b, best_a, gj, gi] = gx * sw - gi   # offset within cell (units of grid cells)
                ground_gy[b, best_a, gj, gi] = gy * sh - gj
                ground_gw[b, best_a, gj, gi] = gw * sw # unit of grid cell 
                ground_gh[b, best_a, gj, gi] = gh * sh 
                ground_class[b, best_a, gj, gi] = int(gt_cls) # class index in each cell 

                # Reconstruct predicted box at this location
                pred_bx[b, best_a, gj, gi] = torch.sigmoid(tx[b, best_a, gj, gi]) # offset within cell (units of grid cells)
                pred_by[b, best_a, gj, gi]= torch.sigmoid(ty[b, best_a, gj, gi])
                pred_bw[b, best_a, gj, gi] = torch.exp(tw[b, best_a, gj, gi]) * self.anchors[best_a][0] # width in grid cell units
                pred_bh[b, best_a, gj, gi] = torch.exp(th[b, best_a, gj, gi]) * self.anchors[best_a][1] # height in grid cell units

                # Calculate iou, using absolute center coordinate
                cx_gt = ground_gx[b, best_a, gj, gi] + gi    # center x in grid units
                cy_gt = ground_gy[b, best_a, gj, gi] + gj
                cx_pred = pred_bx[b, best_a, gj, gi] + gi
                cy_pred = pred_by[b, best_a, gj, gi] + gj

                iou_map[b, best_a, gj, gi] = self.calculate_iou(
                    [cx_gt, cy_gt, ground_gw[b, best_a, gj, gi], ground_gh[b, best_a, gj, gi]],
                    [cx_pred, cy_pred, pred_bw[b, best_a, gj, gi], pred_bh[b, best_a, gj, gi]]
                )        
        
        print(f"[DEBUG] Positive anchor matches in batch: {pos_count}")
        

        # Step 4: compute individual loss components
        # decode predictions using anchors
        loss_coord_offset = 0.0 
        loss_coord_scale = 0.0

        # loss_coord
        loss_coord_offset = self.lambda_coord * torch.sum(
            obj_mask * ((pred_bx - ground_gx) ** 2 + (pred_by - ground_gy) ** 2)
        )
        loss_coord_scale = self.lambda_coord * torch.sum(
            obj_mask * ((torch.sqrt(pred_bw + 1e-6) - torch.sqrt(ground_gw + 1e-6)) ** 2 +
                        (torch.sqrt(pred_bh + 1e-6) - torch.sqrt(ground_gh + 1e-6)) ** 2)
        )
        
        loss_obj = torch.sum(obj_mask * (iou_map.detach() - obj_conf ) ** 2)
        loss_no_obj = self.lambda_noobj * torch.sum(no_obj_mask * (0.0 - obj_conf) ** 2)

        """
        # convert gt_class_map (class index) into one-hot
        gt_class_onehot = torch.nn.functional.one_hot(
            gt_class_map, num_classes=self.num_classes).float()  # [B, A, S, S, C]
        # pred_class matches shape [B, A, S, S, C]
        pred_class = torch.softmax(class_probs, dim=-1)
        # compute squared error 
        loss_per_class = (pred_class - gt_class_onehot)**2

        obj_mask_expanded = obj_mask.unsqueeze(-1)  # [B, A, S, S, 1]
        
        loss_class = torch.sum(obj_mask_expanded * loss_per_class)
        """

        # Step 6: return total loss

        total_loss = self.lambda_coord * (loss_coord_offset + loss_coord_scale) + loss_obj + self.lambda_noobj * loss_no_obj # + loss_class

        ## return {'total': total_loss, 'coord': loss_coord_offset + loss_coord_scale, 'obj': loss_obj + loss_no_obj , 'cls': loss_class}
        return {'total': total_loss, 'coord': loss_coord_offset + loss_coord_scale, 'obj': loss_obj + loss_no_obj}


# pytorch fp32 model
def get_model():
    model, image_size, description = build_model(
        net_id="mcunet-in4", pretrained=True)  # you can replace net_id with any other option from net_id_list

    print("model image size: " + str(image_size))

    print("====== model structure ======")
    summary(model, input_size=(3, 160, 160), device="cpu")

    print("====== module name ======")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"{name}: {module}")

    print("\n=== Linear Layers ===")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"{name}: {module}")

    return model 


def visualise_iou_anchor(image_pil, gt_box, anchor, save_path=None):

    num_classes = 1 # COCO 80, but we only detect human 
    image_size = 160 # mcunet
    S = 5 

    draw = ImageDraw.Draw(image_pil)
    w, h = image_pil.size
    
    cx, cy, bw, bh, _ = gt_box.tolist()
    xmin = (cx - bw/2) * w
    ymin = (cy - bh/2) * h
    xmax = (cx + bw/2) * w 
    ymax = (cy + bh/2) * h

    draw.rectangle([xmin, ymin, xmax, ymax], outline = "red", width = 2)
    
    aw, ah = anchor 
    anchor_xmin = (cx - aw / 2) * w
    anchor_ymin = (cy - ah / 2) * h
    anchor_xmax = (cx + aw / 2) * w
    anchor_ymax = (cy + ah / 2) * h


    draw.rectangle([anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax], outline="blue", width=1)
    image_pil.save(save_path)

"""
if __name__ == "__main__":
    model = McuYolo(num_classes = 80, num_anchors = 5)
    dummy_input = torch.randn(1, 3, 160, 160)
    output = model(dummy_input)
    
    print(output.shape)  # Should be [1, 5, 5, 125] if VOC

    dummy_input = torch.randn(2, 3, 160, 160)
    dummy_targets = [
        [torch.tensor([0.5, 0.5, 0.2, 0.3, 10], device = DEVICE)],  # image 1: 1 box
        [torch.tensor([0.4, 0.4, 0.1, 0.2, 5], device = DEVICE)]    # image 2: 1 box
    ]

    model = McuYolo(num_classes=80, num_anchors=5)
    loss_fn = Yolov2Loss(num_classes=80, anchors=[[1,2], [2,1], [1.5,1.5], [2,2], [1,1]])

    pred = model(dummy_input)
    loss = loss_fn(pred, dummy_targets)
    print("Dummy loss:", loss.item())
"""

"""
# optimise 
def generate_anchors(scales, ratios):
    # Generates anchor boxes for given scales and aspect ratios.
    anchors = []
    for scale in scales:
        for ratio in ratios:
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchors.append((width, height))
    return np.array(anchors)

# Example: Scales and ratios
scales = [0.1, 0.2, 0.4]
ratios = [0.5, 1, 2]
anchors = generate_anchors(scales, ratios)
print("Anchor Boxes:", anchors)
"""


# data augmentation 
