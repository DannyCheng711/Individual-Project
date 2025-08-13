import torch 
import torch.nn as nn
from ...utils import MyModule 

# Create a Custom Detection Head Compatible with YoloClassifier
class McuYoloDetectionHead(MyModule):
    def __init__(self, num_classes=20, num_anchors=5):
        super().__init__()
    
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU6(inplace=True)
        )
        self.space_to_depth = SpaceToDepth()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512 + 384, out_channels=512, kernel_size=1),
            nn.ReLU6(inplace=True)
        )
        self.det_head = nn.Conv2d(in_channels = 512, out_channels= num_anchors * (5 + num_classes), kernel_size=1)

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, final_features):
        """Standard forward when intermeidate features aren't available"""
        x = self.conv1(final_features)
        x = self.conv2(x)
        return self.det_head(x)
        
    def forward_with_intermediate(self, final_features, intermediate_features):
        """Full pipeline with intermediate features"""
        passthrough_feat = intermediate_features.get('passthrough')
        final_feat = intermediate_features.get('final')

        if final_feat is not None:
            x = final_feat
        else:
            x = final_features

        x = self.conv1(x) # 5x5x320 -> 5x5x512
        x = self.conv2(x) # 5x5x512 -> 5x5x512

        if passthrough_feat is not None:
            print("[PYTORCH] Conv3 is being used!") 
            passthrough = self.space_to_depth(passthrough_feat)
            x = torch.cat([x, passthrough], dim=1)
            x = self.conv3(x)
        else:
            print("[PYTORCH] Conv3 is skipped!")   
        return self.det_head(x)
    

    @property
    def config(self):
        return {
            'name': McuYoloDetectionHead.__name__,
            'num_classes': self.num_classes,
            'num_anchors': self.num_anchors, 
        }
    
    @staticmethod
    def build_from_config(config):
        return McuYoloDetectionHead(
            num_classes=config.get('num_classes', 20),
            num_anchors=config.get('num_anchors', 5)
        )


# using sequential for creating the model 
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
    
