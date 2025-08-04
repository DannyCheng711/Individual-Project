import torch 
import torch.nn as nn
from ...utils import MyModule 

# Create a Custom Detection Head Compatible with YoloClassifier
class McuYoloDetectionHead(MyModule):
    def __init__(self, num_classes=20, num_anchors=5):
        super().__init__()
    
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.space_to_depth = SpaceToDepth()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512 + 384, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.det_head = YoloHead(in_channels = 512, num_classes = num_classes, num_anchors = num_anchors)

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
            passthrough = self.space_to_depth(passthrough_feat)
            x = torch.cat([x, passthrough], dim=1)
            x = self.conv3(x)

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


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size
    

    def forward(self, inputs):
        B, C, H, W = inputs.size()
        out_C = C * (self.block_size ** 2)
        out_H = H // self.block_size
        out_W = W // self.block_size

        inputs = inputs.reshape(B, C, out_H, self.block_size, out_W, self.block_size)
        inputs = inputs.permute(0, 1, 3, 5, 2, 4).contiguous()
        inputs = inputs.reshape(B, out_C, out_H, out_W)
        return inputs


# delete here later !!!
class YoloHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_channels = num_anchors * (5 + num_classes)
        self.detector = nn.Conv2d(
            in_channels=in_channels, out_channels = self.output_channels, kernel_size=1)
        
    def forward(self, x):
        return self.detector(x)
    

