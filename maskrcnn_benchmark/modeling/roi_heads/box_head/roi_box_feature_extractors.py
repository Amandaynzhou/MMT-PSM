# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
import torch
from maskrcnn_benchmark.modeling.utils import GradientReversal,GRL
from maskrcnn_benchmark.layers import Conv2d
@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
        )

        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


class MaskRCNNFPNAdaptor(nn.Module):

    def __init__(self, cfg):
        super(MaskRCNNFPNAdaptor, self).__init__()
        channel = [256, 256, 256, 256, 256]
        if cfg.MT.T_ADAPT is not True:
            out_channel = channel
        else:
            out_channel = [128,128,128, 128,128]
        self.adapter_1 = self._init_adaptor(channel[0],out_channel[0])
        self.adapter_2 = self._init_adaptor(channel[1],out_channel[1])
        self.adapter_3 = self._init_adaptor(channel[2],out_channel[2])
        self.adapter_4 = self._init_adaptor(channel[3],out_channel[3])
        self.adapter_5 = self._init_adaptor(channel[4],out_channel[4])

    def _init_adaptor(self, s_channel, t_channel):
        adaptor = Conv2d( s_channel, t_channel, 1, 1, 0)
        nn.init.kaiming_uniform_(adaptor.weight, a=1)
        nn.init.constant_(adaptor.bias, 0)
        return adaptor

    def forward(self, features_s):
        # adapt
        features1 = self.adapter_1(features_s[0])
        features2 = self.adapter_2(features_s[1])
        features3 = self.adapter_3(features_s[2])
        features4 = self.adapter_4(features_s[3])
        features5 = self.adapter_5(features_s[4])
        output = [features1,features2,features3,features4,features5]

        return output

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.pooler = pooler
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.cfg = cfg

        for l in [self.fc6, self.fc7]:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x, proposals, filp = False, istrain = False):
        # pooler only support cuda:0!!!
        device =x[0].device.index
        if device!=0:
            x = [p.to('cuda:0') for p in x]
            proposals = [p.to('cuda:0') for p in proposals]
        x = self.pooler(x, proposals)
        if device!=0:
            x =x.to('cuda:1')

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))

        x = F.relu(self.fc7(x))
        if self.cfg.MODEL.ROI_BOX_HEAD.DO > 0:
            # todo train for teacher
            x = F.dropout(x,p=self.cfg.MODEL.ROI_BOX_HEAD.DO,training=istrain)
        return x

def make_roi_box_feature_extractor(cfg):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg)
