# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import json

import torch.nn as nn
from .layers import *
from ...utils import download_url, MyNetwork, MyModule

__all__ = ['ProxylessNASNets', 'MobileInvertedResidualBlock', 'YOLOClassifier']


def proxyless_base(net_config=None, n_classes=None, bn_param=None, dropout_rate=None,
                   local_path='~/.torch/proxylessnas/'):
    assert net_config is not None, 'Please input a network config'
    if 'http' in net_config:
        net_config_path = download_url(net_config, local_path)
    else:
        net_config_path = net_config
    net_config_json = json.load(open(net_config_path, 'r'))

    if n_classes is not None:
        net_config_json['classifier']['out_features'] = n_classes
    if dropout_rate is not None:
        net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    if bn_param is not None:
        net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)


class YOLOClassifier(MyModule):

    def __init__(self, layer1=None, layer2=None, isConv=True, need_intermediate_features=False):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.isConv = isConv
        self.needs_intermediate_features = need_intermediate_features

    def forward(self, x, intermediate_features=None):
        if self.layer1 is not None:
            if self.needs_intermediate_features and intermediate_features is not None:
                if hasattr(self.layer1, 'forward_with_intermediate'):
                    x = self.layer1.forward_with_intermediate(x, intermediate_features) # forward with intermediate feat?
                else:
                    x = self.layer1(x)
            else:
                x = self.layer1(x)

        if self.layer2 is not None:
            x = self.layer2(x)
        # if self.isConv:
        #     x = x.permute(0, 2, 3, 1)
        # else:
        #     x = x.reshape(-1, 7, 7, 30) # Traditional YOLO head v1
        return x

    @property
    def config(self):
        return {
            'name': YOLOClassifier.__name__,
            'layer1': self.layer1.config if self.layer1 is not None else None,
            'layer2': self.layer2.config if self.layer2 is not None else None,
            'isConv': self.isConv,
            'needs_intermediate_features': self.needs_intermediate_features,
        }

    @staticmethod
    def build_from_config(config):
        layer1 = set_layer_from_config(config['layer1'])
        layer2 = set_layer_from_config(config['layer2'])
        isConv = config['isConv']
        needs_intermediate = config.get('need_intermediate_features', False)
        return YOLOClassifier(layer1, layer2, isConv, needs_intermediate)


class ProxylessNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super().__init__()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer # currently, it is None
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)

        # Check if YoloHead needs intermediate features 
        intermediate_features = {}
        needs_intermediate = (hasattr(self.classifier, 'needs_intermediate_features') and 
                            self.classifier.needs_intermediate_features)


        for i, block in enumerate(self.blocks):
            x = block(x)
            # Store intermediate features 
            if needs_intermediate:
                if i == 12:
                    intermediate_features['passthrough'] = x
                if i == 16:
                    intermediate_features['final'] = x

        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)

        if needs_intermediate:
            x = self.classifier(x, intermediate_features)
        else:
            x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': None if self.feature_mix_layer is None else self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):

        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])

        if config['classifier']['name'] == 'YOLOClassifier':
            classifier = YOLOClassifier.build_from_config(config['classifier'])
        else:
            # Initial loading
            classifier = set_layer_from_config(config['classifier'])


        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))
        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

