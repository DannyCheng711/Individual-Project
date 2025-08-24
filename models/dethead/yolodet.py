# Standard library
import os
# Third-party
import torch
import torch.nn as nn 
import torch.nn.functional as F
from dotenv import load_dotenv
import torchvision.models as models

from models.mcunet.mcunet.model_zoo import build_model


load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
VOC_ROOT = os.getenv("VOC_ROOT")


class McunetTaps(nn.Module):
    """Return (passthrough, final) from MCUNet: blocks[12] -> blocks[16]."""
    # e.g., block 12: 10×10×96, block 16: 5×5×320 (for 160 input)
    def __init__(self, backbone, passthrough_idx=12, final_idx=16):
        super().__init__()
        self.backbone = backbone
        self.passthrough_idx = passthrough_idx
        self.final_idx = final_idx

    def forward(self, x):
        x = self.backbone.first_conv(x)
        passthrough = final = None
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i == self.passthrough_idx:  
                passthrough = x
            if i == self.final_idx: 
                final = x
                break
        return passthrough, final


class MobilenetV2Taps(nn.Module):
    """
    Return (passthrough, final) from MCUNet: blocks[13] -> blocks[16].
    Use blcocks[12] (stride-16, ~14x14x32) as passthrough, features[16] (stride-32, ~7x7x112) as final. 
    """
   
    def __init__(self, backbone, passthrough_idx=12, final_idx=16):
        super().__init__()
        self.backbone = backbone
        self.passthrough_idx = passthrough_idx
        self.final_idx = final_idx

    def forward(self, x):
        x = self.backbone.first_conv(x)
        passthrough = final = None
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i == self.passthrough_idx:
                passthrough = x
            if i == self.final_idx:
                final = x
                break
        return passthrough, final


class ResNet18Taps(nn.Module):
    """Return (passthrough, final) feature maps from ResNet-18."""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # keep blocks separately
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # -> 56×56
        self.layer1 = resnet.layer1   # -> 56×56
        self.layer2 = resnet.layer2   # -> 28×28
        self.layer3 = resnet.layer3   # -> 14×14, 256ch
        self.layer4 = resnet.layer4   # -> 7×7, 512ch

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        passthrough = self.layer3(x)   # passthrough
        final = self.layer4(passthrough)   # final
        return passthrough, final
 
class McuYolo(nn.Module):
    def __init__(self, taps, num_classes=20, num_anchors=5, final_ch=320, passthrough_ch=96, mid_ch=512, s2d_r=2):
        super().__init__()
        self.taps = taps 

        # extra 3 conv layer: conv1 and conv2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=final_ch, out_channels=mid_ch, kernel_size=1),  # Update in_channels
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # space to depth
        self.space_to_depth = SpaceToDepth() # 10×10×96 -> 5×5×384
        fuse_in = mid_ch + passthrough_ch * (s2d_r ** 2)

        # extra 3 conv layer: conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=fuse_in, out_channels=mid_ch, kernel_size=1),  # Update in_channels
            nn.ReLU(inplace=True)
        )

        # add detection head 
        # Output: [B, A*(5+C), H, W]
        self.det_head = nn.Conv2d(in_channels = mid_ch, out_channels= num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, inputs):

        x = inputs 
        passthrough_feat, final_feat = self.taps(x)

        # head conv layers
        x = self.conv1(final_feat) 
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

class SpaceToDepth(nn.Module):
    """Space-to-Depth for NCHW that matches tf.nn.space_to_depth (NHWC) channel order."""
    def __init__(self, r=2):
        super().__init__()
        self.r = r

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        r = self.r
        assert H % r == 0 and W % r == 0
        Ho, Wo = H // r, W // r
        # Move the subpixel dims in front of channels (subpixel-major), then flatten
        # (B,C,Ho,r,Wo,r) -> (B,r,r,C,Ho,Wo) -> (B, C*r*r, Ho, Wo)
        x = x.view(B, C, Ho, r, Wo, r).permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(B, C * r * r, Ho, Wo)
    

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

                            gt_box = [cx_gt, cy_gt, gw[b, i, j, a], gh[b, i, j, a]] # grid unit
                            pr_box = [cx_pr, cy_pr, pred_w[b, i, j, a], pred_h[b, i, j, a]] # grid unit

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

        """
        Yolov1
        lambda_coord = 5.0
        lambda_obj = 5.0
        lambda_noobj = 0.5
        """
        total_loss = 5.0 * (loss_xy + loss_wh) + 5.0 * loss_obj + 0.5 * loss_noobj + loss_cls
        # total_loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_cls
        # return total_loss
        return {'total': total_loss, 'coord': loss_xy + loss_wh, 'class': loss_cls, 'obj': loss_obj + loss_noobj}

