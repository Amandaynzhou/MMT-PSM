# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.utils.miscellaneous import batch_boxlist_hflip,_hflip


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, relation = True):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.use_realation_nms =relation and  cfg.MODEL.RELATION_NMS.USE_RELATION_NMS
        self.cfg = cfg
    def set_teacher_mode(self, mode):
        self.mode = mode

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals, istrain =
        self.training)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        if not self.training:
            # if not learn nms, then uncomment the following line
            if not self.use_realation_nms:
                proposals = self.post_processor((class_logits, box_regression), proposals)
            return (
                x,
                proposals,
                {},
                class_logits,
                box_regression
            )
        loss_classifier, loss_box_reg = self.loss_evaluator(
        [class_logits], [box_regression])
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier,
                 loss_box_reg=loss_box_reg),
            class_logits,
            box_regression
        )

    def _forward_single(self, proposals, targets, feats_list, istrain = False):
        if targets is not None:
            proposals = self.loss_evaluator.subsample(proposals, targets)
        proposals_B = batch_boxlist_hflip(proposals)
        feats = []
        logits = []
        regressions = []
        for i, feat in enumerate(feats_list):
            if i % 2 == 0:
                x = self.feature_extractor(feat, proposals, istrain=istrain)
            else:
                x = self.feature_extractor(feat, proposals_B, istrain=istrain)
            class_logit, box_regression = self.predictor(x)
            feats.append(x)
            logits.append(class_logit)
            regressions.append(box_regression)
        # extract feature

        return feats,logits,regressions,proposals

    def forward_teacher(self, feature_tuple, proposals, targets):
        feats, logits, regressions, proposals = \
            self._forward_single(proposals,targets,feature_tuple,istrain= False)
        return (feats, proposals, {}, logits, regressions)

    def forward_student(self, features, proposals, class_logits_t):

        feats, logits, regressions, _ = self._forward_single(proposals, targets=None, feats_list=features, istrain=True)
        if self.cfg.MT.ODKD:
            cls_loss = self.loss_evaluator.evaluateODKD(logits, proposals, class_logits_t)
        elif self.cfg.MT.CLS_LOSS>0:
            cls_loss = self.loss_evaluator.evaluatePSM(logits, class_logits_t, proposals )
        else:
            cls_loss = torch.zeros((1,)).to(logits[0].device)#
            # placeholder
        loss_dict = dict(mt_classifier=cls_loss)
        return loss_dict


def build_roi_box_head(cfg, relation = True):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, relation=relation)
