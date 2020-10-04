# Wriiten by yanning zhou.

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d,ConvTranspose2d


class RoiAlignMaskFeatureExtractor(nn.Module):
    """
    Relation Mask head feature extractor.
    """
    # _Type = [('RoiPool',), ('Mask',), ('RoiPool', 'Mask')]
    def __init__(self, cfg):
        super(RoiAlignMaskFeatureExtractor, self).__init__()
        input_channels = 257
        self.mask_fcn1 = Conv2d(input_channels, 256, 3, 1, 1)
        self.mask_fcn2 = Conv2d(256, 256, 3, 1, 1)
        self.mask_fcn3 = Conv2d(256, 256, 3, 1, 1)
        if cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL == 1:
            self.conv5_mask = ConvTranspose2d(256, 256, 2, 2, 0)
            self.mask_fcn_logits = Conv2d(256, 1, 3, 1, 1)
            for l in [self.mask_fcn1, self.mask_fcn2, self.mask_fcn3,  self.conv5_mask, self.mask_fcn_logits]:
                nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(l.bias, 0)
        else:
            self.mask_fcn_logits = None
            self.conv5_mask = Conv2d(256, 16, 3, 1, 1)
            for l in [self.mask_fcn1, self.mask_fcn2, self.mask_fcn3,  self.conv5_mask,]:
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
        if self.mask_fcn_logits:
            x = self.mask_fcn_logits(x)
        return x


class SameSizeRoiAlignMaskFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(SameSizeRoiAlignMaskFeatureExtractor, self).__init__()
        input_channels = 257
        self.mask_fcn1 = Conv2d(input_channels, 256, 3, 1, 1)
        self.mask_fcn2 = Conv2d(256, 256, 3, 1, 1)
        self.mask_fcn3 = Conv2d(256, 256, 3, 1, 1)
        self.conv5_mask = Conv2d(256, cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL, 3, 1, 1)
        for l in [self.mask_fcn1, self.mask_fcn2, self.mask_fcn3,  self.conv5_mask,]:
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



class RoiAlignFeatureExtractor(nn.Module):

    def __init__(self, cfg):
        input_channels = 256
        self.mask_fcn1 = Conv2d(input_channels, 256, 3, 1, 1)
        self.mask_fcn2 = Conv2d(256, 256, 3, 1, 1)
        self.mask_fcn3 = Conv2d(256, 256, 3, 1, 1)
        self.conv5_mask = ConvTranspose2d(256, 256, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(256, 1, 1, 1, 0)

        for l in [self.mask_fcn1, self.mask_fcn2, self.mask_fcn3, self.mask_fcn4, self.conv5_mask, self.mask_fcn_logits]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = F.relu(self.mask_fcn1(x))
        x = F.relu(self.mask_fcn2(x))
        x = F.relu(self.mask_fcn3(x))
        x = F.relu(self.conv5_mask(x))
        x = self.mask_fcn_logits(x)
        return x


class MaskFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(MaskFeatureExtractor, self).__init__()
        print('init MaskFeatureExtractor')
    def forward(self, x):
        (roi, mask ) = x
        return mask

class SameFeatureMask(nn.Module):
    def __init__(self,cfg):
        super(SameFeatureMask, self).__init__()
    def forward(self, x):
        return x

class DeepFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(DeepFeatureExtractor, self).__init__()
        input_channels = 256
        self.mask_fcn1 = Conv2d(input_channels, 256, 3, 1, 1)
        self.mask_fcn2 = Conv2d(256, 256, 3, 1, 1)
        self.mask_fcn3 = Conv2d(256, 256, 3, 1, 1)
        self.conv5_mask = Conv2d(256, cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL, 3, 1, 1)
        for l in [self.mask_fcn1, self.mask_fcn2, self.mask_fcn3, self.conv5_mask, ]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x, mask = x
        # import  pdb;pdb.set_trace()
        # mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        # x = torch.cat((x, mask_pool), 1)
        x = F.relu(self.mask_fcn1(x))
        x = F.relu(self.mask_fcn2(x))
        x = F.relu(self.mask_fcn3(x))
        x = F.relu(self.conv5_mask(x))
        return x


_RELATION_MASK_FEATURE_EXTRACTORS = {
    "RoiAlignFeatureExtractor": RoiAlignFeatureExtractor,
    "MaskFeatureExtractor": MaskFeatureExtractor,
    "RoiAlignMaskFeatureExtractor": RoiAlignMaskFeatureExtractor,
    "SameFeatureMask": SameFeatureMask,
    "SameSizeRoiAlignMaskFeatureExtractor": SameSizeRoiAlignMaskFeatureExtractor,
    "DeepFeatureExtractor": DeepFeatureExtractor
}


def make_relation_mask_feature_extractor(cfg):
    func =  _RELATION_MASK_FEATURE_EXTRACTORS[cfg.MODEL.RELATION_MASK.FEATURE_EXTRACTOR]
    return func(cfg)



class ShapeBuffer(nn.Module):

    def __init__(self, cfg):
        super(ShapeBuffer, self).__init__()
        self.center = nn.Parameter(torch.Tensor(int(2 * cfg.MODEL.RELATION_MASK.CENTER_PER_CLASS), 1, 28, 28), requires_grad= True)
        nn.init.xavier_normal_(self.center)
    def _set_center(self, means):
        self.center.data = means

    def forward(self):
        return self.center

    def __repr__(self):
        return self.__class__.__name__ + '(center)'