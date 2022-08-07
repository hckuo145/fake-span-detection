import torch
import torch.nn as nn

try:
    from .sublayer import *
    from .feature  import *
    from .pooling  import *
except:
    from sublayer import *
    from feature  import *
    from pooling  import *

from typing import Tuple, List, Any


class SENet(nn.Module):
    def __init__(self, layers:List[int], channels:List[int]=[16, 16, 32, 64, 128]) -> None:
        super(SENet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=(2, 1), padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

        self.inplanes = channels[0]
        self.layer1 = self._make_layer(SEBlock, channels[1], layers[0])
        self.layer2 = self._make_layer(SEBlock, channels[2], layers[1], stride=(2, 1))
        self.layer3 = self._make_layer(SEBlock, channels[3], layers[2], stride=(2, 1))
        self.layer4 = self._make_layer(SEBlock, channels[4], layers[3], stride=(2, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block:Any, planes:int, blocks:int, stride:Tuple[int, ...]=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class MainModel(nn.Module):
    def __init__(self, frontend, backbone, pooling, attn_heads, duration):
        super(MainModel, self).__init__()

        _sample = torch.randn(1, int(duration * 16000))

        self.frontend  = globals()[frontend['name']](**frontend['args'])

        _sample = self.frontend(_sample)
        _batch, _n_features, _n_frames = _sample.size()

        self.frontendnorm = nn.InstanceNorm1d(_n_features)
        self.backbone     = globals()[backbone['name']](**backbone['args'])

        _sample = _sample.unsqueeze(1)
        _sample = self.backbone(_sample)
        _batch, _n_channels, _n_features, _n_frames = _sample.size()
        
        _n_features *= _n_channels

        self.attention  = TransformerLayer(_n_features, attn_heads, _n_channels)
        self.spandetect = nn.Linear(_n_features, 2)
        
        try:
            self.pooling = globals()[pooling]()
        except:
            self.pooling = globals()[pooling](_n_features)
        
        if 'Statistics' in pooling:
            _n_features *= 2
        
        self.antispoof = nn.Linear(_n_features, 2)


    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        frontend_feature = self.frontend(x)
        frontend_feature = self.frontendnorm(frontend_feature)
        frontend_feature = frontend_feature.unsqueeze(1)

        backbone_feature = self.backbone(frontend_feature)
        backbone_feature = backbone_feature.flatten(start_dim=1, end_dim=2)
        backbone_feature = backbone_feature.permute(0, 2, 1).contiguous()

        attention_feature = self.attention(backbone_feature)
        pos_output = self.spandetect(attention_feature)

        pooling_feature = self.pooling(attention_feature)
        cls_output = self.antispoof(pooling_feature)

        return cls_output, pos_output



if __name__ == '__main__':
    frontend = {
        'name': 'MelSpectrogram',
        'args': {
            'n_fft'     : 512,
            'n_mels'    : 80,
            'hop_length': 128
        }
    }
    
    backbone = {
        'name': 'SENet',
        'args': {
            'layers'  : [3, 4, 6, 3],
            'channels': [16, 16, 32, 64, 128]
        }
    }
    
    pooling = 'Temporal_Average_Pooling'
    
    attn_heads = 8
    duration   = 4.

    x = torch.randn(10, int(duration * 16000))
    
    m = MainModel(frontend, backbone, pooling, attn_heads, duration)
    cls, pos = m(x)
    print(m)
    print(cls.size(), pos.size())

    pooling = 'Temporal_Statistics_Pooling'

    m = MainModel(frontend, backbone, pooling, attn_heads, duration)
    cls, pos = m(x)
    print(cls.size(), pos.size())

    pooling = 'Self_Attentive_Pooling'

    m = MainModel(frontend, backbone, pooling, attn_heads, duration)
    cls, pos = m(x)
    print(cls.size(), pos.size())
    
    pooling = 'Attentive_Statistics_Pooling'
    
    m = MainModel(frontend, backbone, pooling, attn_heads, duration)
    cls, pos = m(x)
    print(cls.size(), pos.size())