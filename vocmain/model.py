import torch
import torch.nn as nn 
import torch.nn.functional as F
from config import DEVICE, VOC_ROOT

# detection head
# Input feature map: [B, in_channels, H, W]
class YoloHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_channels = num_anchors * (5 + num_classes)
        self.detector = nn.Conv2d(in_channels = in_channels, out_channels=self.output_channels, kernel_size=1) # 1x1 conv on each cell

    def forward(self, x):
        # Input: [B, C, H, W] -> Output: [B, A*(5+C), H, W]
        return self.detector(x) 

# feature map  
class McuYolo(nn.Module):

    def __init__(self, backbone_fn, num_classes=20, num_anchors=5):
        super().__init__()
        self.backbone = backbone_fn

        self.passthrough_layer_idx = 12 # block 12: 10×10×96
        self.final_block_idx = 16 # block 16: 5×5×320

        # extra 3 conv layer: conv1 and conv2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1),  # Update in_channels
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # space to depth
        self.space_to_depth = SpaceToDepth() # 10×10×96 -> 5×5×384

        # Concatenate passthrough + final, reduce channels to 512
        ## self.concat_conv = nn.Conv2d(384 + 512, 512, kernel_size=1)

        # extra 3 conv layer: conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512 + 384, out_channels=512, kernel_size=1),  # Update in_channels
            nn.ReLU(inplace=True)
        )

        # add detection head
        self.det_head = YoloHead(in_channels = 512, num_classes = num_classes, num_anchors=num_anchors)

    def forward(self, inputs, training = False):

        x = inputs 
        passthrough_feat = None

        # First pass through the initial convolution layer
        x = self.backbone.first_conv(x)
    
        # forward each block manually via for loop 
        # each block is a subclass of nn.Module, and doing block(x) is equivalent to block.forward(x) 
        for i in range(self.final_block_idx + 1):
            x = self.backbone.blocks[i](x)
            if i == self.passthrough_layer_idx:
                passthrough_feat = x 
            if i == self.final_block_idx:
                break

        # head conv layers
        x = self.conv1(x) 
        x = self.conv2(x) 

        passthrough = self.space_to_depth(passthrough_feat) 

        # concatenate
        x = torch.cat([x, passthrough], dim = 1)  # concatenate along channel dimension
        x = self.conv3(x) 

        # final detection 
        return self.det_head(x)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False 
        print("Backbone frozen ... ")

    def unfreeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True
        print("Backbone unfrozen ...")

# using sequential for creating the model 
class SpaceToDepth(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size 

    # inputs is the previous output
    def forward(self, inputs):
        # Input: [B, C, H, W]
        B, C, H, W = inputs.size()

        out_C = C * (self.block_size ** 2)
        out_H = H // self.block_size
        out_W = W // self.block_size

         # Reshape: [B, C, H, W] -> [B, C, H//block_size, block_size, W//block_size, block_size]
        inputs = inputs.reshape(B, C, out_H, self.block_size, out_W, self.block_size)

        # Permute and reshape to get final output: [B, C*block_size^2, H//block_size, W//block_size]
        inputs = inputs.permute(0, 1, 3, 5, 2, 4).contiguous()
        inputs = inputs.reshape(B, out_C, out_H, out_W)

        return inputs

# Loss function
class Yolov2Loss(nn.Module):
    def __init__(self, num_classes, anchors, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors 
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

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
        inter_xmin = torch.max(x1_min, x2_min)
        inter_ymin = torch.max(y1_min, y2_min)
        inter_xmax = torch.min(x1_max, x2_max)
        inter_ymax = torch.min(y1_max, y2_max)

        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0.0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0.0)
        inter_area = inter_w * inter_h

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else torch.tensor(0.0)
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
        inter_w = torch.min(w1, w2)
        inter_h = torch.min(h1, h2)
        inter_area = inter_w * inter_h

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        iou = inter_area / union_area if union_area > 0 else torch.tensor(0.0)
        return iou

    
    def forward(self, predictions, targets, imgs = None):

        # predictions [B, A*(5+C), S, S]
        # targets [B, S, S, A, 5+C]
        B, pred_dim, S, _ = predictions.shape
        A = len(self.anchors)
        C = self.num_classes

        # Reshape and permute to [B, S, S, A, 5+C]
        predictions= predictions.reshape(B, A, 5+C, S, S)
        predictions = predictions.permute(0, 3, 4, 1, 2) # [B, S, S, A, 5+C]
        
        # Extract tx, ty, tw, th, obj_score, class_probs
        tx = predictions[..., 0]
        ty = predictions[..., 1]
        tw = predictions[..., 2]
        th = predictions[..., 3]
        conf_logits = predictions[..., 4]
        class_logits = predictions[..., 5:]

        # Extract targets 
        gx = targets[..., 0]
        gy = targets[..., 1]
        gw = targets[..., 2]
        gh = targets[..., 3]
        obj_mask = targets[..., 4]  # 1 is object, 0 otherwise
        class_target = targets[..., 5:]

        # Predicted box decoding
        pred_cx = torch.sigmoid(tx)
        pred_cy = torch.sigmoid(ty)

        # anchors is tensor
        pred_w = torch.exp(tw) * self.anchors[:, 0].view(1, 1, 1, -1)
        pred_h = torch.exp(th) * self.anchors[:, 1].view(1, 1, 1, -1)

    
        iou = torch.zeros_like(pred_cx)
        for b in range(B):
            for a in range(A):
                for i in range(S):
                    for j in range(S):
                        if obj_mask[b, i, j, a] == 1:
                            # Convert into grid cell unit
                            cx_gt = gx[b, i, j, a] + float(j)
                            cy_gt = gy[b, i, j, a] + float(i)

                            cx_pr = pred_cx[b, i, j, a] + float(j)
                            cy_pr = pred_cy[b, i, j, a] + float(i)

                            gt_box = [cx_gt, cy_gt, gw[b, i, j, a], gh[b, i, j, a]]
                            pr_box = [cx_pr, cy_pr, pred_w[b, i, j, a], pred_h[b, i, j, a]]

                            # PyTorch tensors are mutable, so we can assign directly
                            iou[b, i, j, a] = self.calculate_iou(gt_box, pr_box)

        loss_xy = torch.sum(obj_mask * (torch.square(pred_cx - gx) + torch.square(pred_cy - gy)))
        loss_wh = torch.sum(obj_mask * (
            torch.square(torch.sqrt(pred_w + 1e-6) - torch.sqrt(gw + 1e-6)) +
            torch.square(torch.sqrt(pred_h + 1e-6) - torch.sqrt(gh + 1e-6))))

        conf_pred = torch.sigmoid(conf_logits)
        loss_obj = torch.sum(obj_mask * torch.square(iou - conf_pred))
        loss_noobj = torch.sum((1.0 - obj_mask) * torch.square(conf_pred))

        # Fix the class loss calculation by expanding obj_mask to match class dimensions
        obj_mask_expanded = obj_mask.unsqueeze(-1).expand(-1, -1, -1, -1, C) # expand with value 1 
        class_loss_per_element = F.binary_cross_entropy_with_logits(class_logits, class_target, reduction='none')
        loss_cls = torch.sum(obj_mask_expanded * class_loss_per_element)

        total_loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_cls
        # return total_loss
        return {'total': total_loss, 'coord': loss_xy + loss_wh, 'class': loss_cls, 'obj': loss_obj + loss_noobj}

if __name__ == "__main__":
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    
    print("Creating dummy inputs ... ")
    # Dummy input: [B, C, H, W] format in PyTorch (channels first)
    dummy_input = torch.randn(2, 3, 160, 160).to(device)

    # Dummy target: [B, S, S, A, 5+C]
    # Assuming your grid S=5, anchors A=5, and classes C=20
    dummy_target = torch.zeros(2, 5, 5, 5, 5 + 20).to(device)

    # Put one box in one grid cell as a positive example
    # Box: [gx, gy, gw, gh, obj=1, one-hot class vector]
    dummy_target[0, 2, 3, 1, :5] = torch.tensor([0.5, 0.5, 0.2, 0.3, 1.0])  # box + obj
    dummy_target[0, 2, 3, 1, 5 + 10] = 1.0  # class 10

    # Backbone
    print("Creating a backbone ... ")
    backbone_fn, _, _ = build_model(net_id="mcunet-in4", pretrained=True)

    # Model & Loss
    print("Building a model ... ")
    model = McuYolo(backbone_fn=backbone_fn, num_classes=20, num_anchors=5).to(device)
    print("Building a loss function ... ")
    anchors = torch.tensor([[1, 2], [2, 1], [1.5, 1.5], [2, 2], [1, 1]], dtype=torch.float32).to(device)
    loss_fn = Yolov2Loss(num_classes=20, anchors=anchors)

    # Forward pass
    print("Forward pass ... ")
    model.eval()
    with torch.no_grad():
        y_pred = model(dummy_input)
        loss = loss_fn(y_pred, dummy_target)

    print("Dummy loss:", loss)