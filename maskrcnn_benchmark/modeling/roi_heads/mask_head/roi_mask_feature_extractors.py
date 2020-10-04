# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F
import torch
from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.layers import Conv2d
import pdb
class PRCNNFeatureExtractor(nn.Module):
    '''
    second stage extractor for paper: Cell Segmentation Proposal Network for Microscopy Image Analysis
    '''
    def __init__(self, cfg):
        super(PRCNNFeatureExtractor, self).__init__()
        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        self.conv1 = Conv2d(3,32, 3, stride=1, padding=1)
        self.conv2 = Conv2d(32,32, 3, stride=1, padding=1)
        self.conv3 = Conv2d(32,64, 3, stride=1, padding=1)
        self.conv4 = Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv5 = Conv2d(64,128, 3, stride=1, padding=1)
        self.conv6 = Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv7  = Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv8 = Conv2d(256, 256, 3, stride=1, padding=1)
        # pdb.set_trace()
        self.pooler1 = Pooler(
            output_size=(25, 25),
            scales=(1.,),
            sampling_ratio=sampling_ratio,
        )
        self.p1 = nn.MaxPool2d(3,2,1)
        self.pooler2 =Pooler(
            output_size=(25, 25),
            scales=(0.5,),
            sampling_ratio=sampling_ratio,
        )
        self.p2 = nn.MaxPool2d(3, 2, 1)
        self.pooler3 =Pooler(
            output_size=(25, 25),
            scales=(0.25,),
            sampling_ratio=sampling_ratio,
        )
        self.p3 = nn.MaxPool2d(3, 2, 1)
        self.pooler4 =Pooler(
            output_size=(25, 25),
            scales=(0.125,),
            sampling_ratio=sampling_ratio,
        )



        self.posconv1 = Conv2d(480,256, 3, stride=1, padding=1)
        self.posconv2 = Conv2d(256, 32, 3, stride=1, padding=1)
        for layer in [self.conv1, self.conv2,self.conv3, self.conv4,self.conv5, self.conv6,self.conv7, self.conv8,
                      self.posconv1, self.posconv2]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0)

    def forward(self, x, proposals):

        pre_feature = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = self.pooler1([x], proposals)
        x = self.p1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x2 = self.pooler2([x], proposals)
        x = self.p2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x3 = self.pooler3([x], proposals)
        x = self.p3(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x4 = self.pooler4([x], proposals)
        # concate
        num = x1.shape[0]
        concate_feature = []
        for i in range(num):
            concate_feature.append(torch.cat([x1[i], x2[i],x3[i], x4[i]], 0)[None,:,:,:])
        concate_feature = torch.cat(concate_feature, 0)
        # pdb.set_trace()
        concate_feature = self.posconv1(concate_feature)
        concate_feature = F.relu(concate_feature)
        concate_feature = self.posconv2(concate_feature)
        return concate_feature, pre_feature


class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.pooler = pooler

        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        # pdb.set_trace()

        device = x[0].device.index
        if device != 0:
            x = [p.to('cuda:0') for p in x]
            proposals = [p.to('cuda:0') for p in proposals]
        x = self.pooler(x, proposals)
        if device != 0:
            x = x.to('cuda:1')
        # x = self.pooler(x, proposals)
        pre_feature = x
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x, pre_feature

class MaskRCNNFPNAdaptor(nn.Module):

    def __init__(self, cfg):
        super(MaskRCNNFPNAdaptor, self).__init__()
        channel = [256, 256, 256, 256, 256]

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.pooler = pooler

        self.adapter_1 = self._init_adaptor(channel[0],channel[0])
        self.adapter_2 = self._init_adaptor(channel[1],channel[1])
        self.adapter_3 = self._init_adaptor(channel[2],channel[2])
        self.adapter_4 = self._init_adaptor(channel[3],channel[3])
        self.adapter_5 = self._init_adaptor(channel[4],channel[4])

    def _init_adaptor(self, s_channel, t_channel):
        adaptor = Conv2d( s_channel, t_channel, 1, 1, 1)
        nn.init.kaiming_normal_(adaptor.weight, mode="fan_out",
                                nonlinearity="relu")
        nn.init.constant_(adaptor.bias, 0)
        return adaptor

    def forward(self, features_s):
        # adapt
        features_s[0] = self.adapter_1(features_s[0])
        features_s[1] = self.adapter_2(features_s[1])
        features_s[2] = self.adapter_3(features_s[2])
        features_s[3] = self.adapter_4(features_s[3])
        features_s[4] = self.adapter_5(features_s[4])
        # for layer_name in self.blocks:
        #     features_s = F.relu(getattr(self, layer_name)(features_s))
        return features_s




class DeeperExtractor(nn.Module):
    def __init__(self, cfg):
        super(DeeperExtractor, self).__init__()
        self.mask_fcn1 = Conv2d(257, 256, 3, 1, 1)
        self.mask_fcn2 = Conv2d(256, 256, 3, 1, 1)
        self.mask_fcn3 = Conv2d(256, 256, 3, 1, 1)
        self.conv5_mask = Conv2d(256, cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL, 3, 1, 1)
        for l in [self.mask_fcn1, self.mask_fcn2, self.mask_fcn3, self.conv5_mask, ]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)
    def forward(self, x):
        x, mask = x
        # import  pdb;pdb.set_trace()
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)
        x = F.relu(self.mask_fcn1(x))
        x = F.relu(self.mask_fcn2(x))
        x = F.relu(self.mask_fcn3(x))
        x = F.relu(self.conv5_mask(x))
        return x

_ROI_MASK_FEATURE_EXTRACTORS = {
    "ResNet50Conv5ROIFeatureExtractor": ResNet50Conv5ROIFeatureExtractor,
    "MaskRCNNFPNFeatureExtractor": MaskRCNNFPNFeatureExtractor,
    "PRCNNFeatureExtractor":PRCNNFeatureExtractor,
    'DeeperExtractor':DeeperExtractor,
    'MaskRCNNFPNAdaptor':MaskRCNNFPNAdaptor
}


def make_roi_mask_feature_extractor(cfg):
    func = _ROI_MASK_FEATURE_EXTRACTORS[cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)

def make_mask_adapt_layer(cfg):
    return MaskRCNNFPNAdaptor(cfg)