from mcunet.mcunet.model_zoo import net_id_list, build_model, download_tflite
from torchsummary import summary 
from config import DEVICE, DATASET_ROOT
import torch
import torch.nn as nn 

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

    def calculate_iou(self, gbox, abox):
        w1, h1 = gbox[2], gbox[3]
        w2, h2 = abox[2], abox[3]

        # compute intersection dimensions
        inter_w = min(w1, w2)
        inter_h = min(h1, h2)
        inter_area = inter_w * inter_h

        # computer union
        area1 = w1 * h1
        area2 = w2 * h2 
        union_area = area1 + area2 - inter_area

        # avoid divide by zero
        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    
    def forward(self, predictions, targets):

        # Step 1: reshape predictions (B, S, S, A*(5+C)) → (B, A, S, S, 5+C)
        batch_size, sh, sw, pred_dim = predictions.shape
        num_anchors = len(self.anchors)

        # Make sure the channel dimension matches
        print("prediction2 shape ... ")
        print(predictions.shape)
        
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
        bx = torch.sigmoid(tx) # offset from top-left corner of the cell
        by = torch.sigmoid(ty) # offset from top-left corner of the cell
        bw = torch.exp(tw) * anchor_w # scale respect to anchor box
        bh = torch.exp(th) * anchor_h # scale respect to anchor box
        obj_conf = torch.sigmoid(obj_score) # IOU
        
        # Step 3: compute objectness mask (1^obj_ij) and noobj mask
        # [B, A, S, S]
        obj_mask = torch.zeros(
            (batch_size, num_anchors, sh, sw), dtype=torch.bool, device = DEVICE)
        no_obj_mask = torch.ones(
            (batch_size, num_anchors, sh, sw), dtype=torch.bool, device = DEVICE)

        # These store the ground truth targets (same shape as prediction maps)
        gx_map = torch.zeros((batch_size, num_anchors, sh, sw), device = DEVICE)
        gy_map = torch.zeros_like(gx_map, device = DEVICE)
        gw_map = torch.zeros_like(gx_map, device = DEVICE)
        gh_map = torch.zeros_like(gx_map, device = DEVICE)
        iou_map = torch.zeros_like(gx_map, device = DEVICE)
        gt_class_map = torch.zeros((batch_size, num_anchors, sh, sw), dtype=torch.long, device = DEVICE)  # for class labels

        for b in range(batch_size):
            # targets[b]: List[Tensor[num_objects, 5]] 
            # [x_center, y_center, width, height, class_id]
            for gt in targets[b]: 
                gx, gy, gw, gh, gt_cls = gt.tolist()

                gi = int(gx * sw) # gx is (0,1) w.r.t. image size # which cell x-axis
                gj = int(gy * sh) 

                gt_box = torch.tensor([0, 0, gw, gh], device = DEVICE) # assume centered at (0, 0), with size gw x gh
                anchor_ious = []

                for anchor in self.anchors: 
                    anchor_box = torch.tensor([0, 0, anchor[0], anchor[1]], device = DEVICE) # use centered at (0, 0) to align coordination
                    iou = self.calculate_iou(gt_box, anchor_box)
                    anchor_ious.append(iou)

                # assign anchor with highest IoU
                best_a = torch.argmax(torch.tensor(anchor_ious, device = DEVICE)).item() # argmax output is tensor(2), use item() extracts the value

                # set masks
                obj_mask[b, best_a, gj, gi] = True
                no_obj_mask[b, best_a, gj, gi] = False

                # Fill target maps (use fractional offset for gx, gy)
                gx_map[b, best_a, gj, gi] = gx * sw - gi   # fractional offset within cell
                gy_map[b, best_a, gj, gi] = gy * sh - gj
                gw_map[b, best_a, gj, gi] = gw
                gh_map[b, best_a, gj, gi] = gh
                gt_class_map[b, best_a, gj, gi] = int(gt_cls)

                # Reconstruct predicted box at this location
                pred_bx = bx[b, best_a, gj, gi] + gi
                pred_by = by[b, best_a, gj, gi] + gj
                pred_bw = bw[b, best_a, gj, gi]
                pred_bh = bh[b, best_a, gj, gi]
                pred_box = torch.tensor([pred_bx, pred_by, pred_bw, pred_bh], device = DEVICE) #, device=device)

                true_box = torch.tensor([gx * sw, gy * sh, gw * sw, gh * sh], device = DEVICE)
                iou_map[b, best_a, gj, gi] = self.calculate_iou(true_box, pred_box)
                        
        
        # Step 4: compute individual loss components
        # decode predictions using anchors
        loss_coord_offset = 0.0 
        loss_coord_scale = 0.0

        # loss_coord
        loss_coord_offset = self.lambda_coord * torch.sum(
            obj_mask * ((bx - gx_map) ** 2 + (by - gy_map) ** 2)
        )
        loss_coord_scale = self.lambda_coord * torch.sum(
            obj_mask * ((torch.sqrt(bw + 1e-6) - torch.sqrt(gw_map + 1e-6)) ** 2 +
                        (torch.sqrt(bh + 1e-6) - torch.sqrt(gh_map + 1e-6)) ** 2)
        )
        
        loss_obj = torch.sum(obj_mask * (iou_map.detach() - obj_conf ) ** 2)
        loss_no_obj = self.lambda_noobj * torch.sum(no_obj_mask * (0.0 - obj_conf) ** 2)

        ## loss_class = torch.sum(
        ##    obj_mask * 
        ## )

        # Step 6: return total loss
        total_loss = loss_coord_offset + loss_coord_scale + loss_obj + loss_no_obj ## + loss_class

        return total_loss


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

"""
# Annotation convertion 
def convert_to_yolo_format(width, height, bbox):
    # Converts absolute bounding box to YOLO format.
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return [x_center, y_center, box_width, box_height]
""" 
# data augmentation 
