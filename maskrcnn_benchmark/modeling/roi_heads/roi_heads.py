# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .maskiou_head.maskiou_head import build_roi_maskiou_head

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box, class_logits, box_regression = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            _, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)
        return x, detections, losses, class_logits, box_regression


#### Rewrite two roi head, so that the learnable nms can be add between them####
class BoxROIHeads(torch.nn.ModuleDict):
    def __init__(self, cfg , boxheads):
        super(BoxROIHeads, self).__init__(boxheads)
        self.cfg = cfg.clone()
    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box, class_logits, box_regression = self.box(features, proposals, targets)
        losses.update(loss_box)
        return x, detections, losses, class_logits, box_regression

    def forward_student(self,features, proposals, class_logits_t):
        return self.box.forward_student(features, proposals,class_logits_t)


    def forward_teacher(self,feature_tuple, proposals, teacher_infer):
        losses = {}
        x, detections, loss_box, class_logits, box_regression = \
            self.box.forward_teacher(feature_tuple, proposals,
                                    targets = teacher_infer)
        losses.update(loss_box)
        return x, detections, losses, class_logits, box_regression


class MaskROIHeads(torch.nn.ModuleDict):
    def __init__(self, cfg, maskheads):
        super(MaskROIHeads, self).__init__(maskheads)
        self.cfg = cfg
    def forward(self, losses, features, detections,
                targets = None, images = None):

        _, detections, loss_mask = self.mask(features, detections,
                                              targets,
                                             images)
        losses.update(loss_mask)
        return detections, losses


def build_roi_heads(cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads

def  box_roi_heads(cfg, relation = True):
    return BoxROIHeads(cfg, [("box", build_roi_box_head(cfg, relation=relation))])

def mask_roi_heads(cfg, is_student = False):
    return MaskROIHeads(cfg, [("mask", build_roi_mask_head(cfg, is_student))])