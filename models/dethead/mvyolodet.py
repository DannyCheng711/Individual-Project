# Standard library
import os
# Third-party
import torch
import torch.nn as nn 

from .yolodet import McunetTaps, SpaceToDepth
from models.mcunet.mcunet.model_zoo import build_model

class McunetBackbone(nn.Module):
    def __init__(self, passthrough_idx=12, final_idx=16, pretrained=True):
        super().__init__()
        backbone, _, _ = build_model(net_id="mcunet-in4", pretrained=pretrained)
        self.taps = McunetTaps(backbone, passthrough_idx=passthrough_idx, final_idx=final_idx)

    def forward(self, x):
        # returns (passthrough, final)
        return self.taps(x)


class CrossModalityAlignmentBlock(nn.Module):
    """Simple learnable fusion block"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, feat_a, feat_b):
        # Concatenate and learn fusion weights
        concat_feat = torch.cat([feat_a, feat_b], dim=1)
        return self.relu(self.bn(self.conv(concat_feat)))
    

class FeatureFusion(nn.Module):
    def __init__(self, final_ch=320, passthrough_ch=96):
        super().__init__()
        self.final_alignment = CrossModalityAlignmentBlock(final_ch)
        self.passthrough_alignment = CrossModalityAlignmentBlock(passthrough_ch)

    
    def forward(self, local_final, local_passthrough, peer_final=None, peer_passthrough=None, detach_peer=True, fallback_zeros=True):

        # peer features 
        if peer_final is None or peer_passthrough is None:
            print(" the peer feature map is lost ")
            if fallback_zeros:
                peer_final = self._zeros_like(local_final)
                peer_passthrough = self._zeros_like(local_passthrough)
            else:
                peer_final = local_final
                peer_passthrough = local_passthrough
     
        # do not update the gradient in its peer 
        if detach_peer:
            peer_final = peer_final.detach()
            peer_passthrough = peer_passthrough.detach()

        # align / fuse 
        fused_final = self.final_alignment(local_final, peer_final)
        fused_passthrough = self.passthrough_alignment(local_passthrough, peer_passthrough)
    

        return fused_final, fused_passthrough


class Yolov2HeadShared(nn.Module):
    def __init__(self, num_classes=20, num_anchors=5, final_ch=320, passthrough_ch=96, mid_ch=512, s2d_r=2):
        super().__init__()
        
        # Detection head
        self.conv1 = nn.Sequential(
            nn.Conv2d(final_ch, mid_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.space_to_depth = SpaceToDepth()
        # Note: passthrough_ch * 2 because we concatenate two passthrough features
        fuse_in = mid_ch + passthrough_ch * (s2d_r ** 2)
        # print("Expected fuse_in:", fuse_in)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(fuse_in, mid_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.det_head = nn.Conv2d(mid_ch, num_anchors * (5 + num_classes), kernel_size=1)

    @torch.no_grad()
    # if the feature map from another model is lost
    def _zeros_like(self, t):  # utility for fallback
        return torch.zeros_like(t)


    def forward(self, fused_final, fused_passthrough):
        
        # head conv layers
        x = self.conv1(fused_final) 
        x = self.conv2(x) 

        passthrough = self.space_to_depth(fused_passthrough) 

        # concatenate
        x = torch.cat([x, passthrough], dim = 1)  # concatenate along channel dimension
        x = self.conv3(x) 

        # final detection 
        return self.det_head(x)


class MultiViewMcuYolo(nn.Module):
    def __init__(self, num_classes=20, num_anchors=5, final_ch=320, passthrough_ch=96, mid_ch=512, s2d_r=2):
        super().__init__()
        # Two per-view backbones (you can tie them if you really want identical weights)
        self.backbone_a = McunetBackbone()
        self.backbone_b = McunetBackbone()

        # Fusion and shared head
        self.fuse = FeatureFusion(final_ch=final_ch, passthrough_ch=passthrough_ch)
        self.head = Yolov2HeadShared(num_classes=num_classes, num_anchors=num_anchors,
                                     final_ch=final_ch, passthrough_ch=passthrough_ch,
                                     mid_ch=mid_ch, s2d_r=s2d_r)

    def forward(self, img_a, img_b, feature_pass_mode=False):
        """
        feature_pass_mode=False: centralized training (grad through both backbones)
        feature_pass_mode=True : simulate devices (no grad on peer features)
        """
        if feature_pass_mode:
            # Local features (grad), peer features (no-grad -> detach)
            passthrough_a, final_a = self.backbone_a(img_a)
            with torch.no_grad():
                passthrough_b, final_b = self.backbone_b(img_b)
            fused_final, fused_pt = self.fuse(final_a, passthrough_a, final_b, passthrough_b,
                                              detach_peer=True, fallback_zeros=False)
        else:
            # Full grad through everything
            passthrough_a, final_a = self.backbone_a(img_a)
            passthrough_b, final_b = self.backbone_b(img_b)
            fused_final, fused_pt = self.fuse(final_a, passthrough_a, final_b, passthrough_b,
                                              detach_peer=False, fallback_zeros=False)

        preds = self.head(fused_final, fused_pt)
        # Return also raw features 
        return preds, (passthrough_a, final_a, passthrough_b, final_b)