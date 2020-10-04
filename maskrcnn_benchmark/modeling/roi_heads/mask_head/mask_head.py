# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.miscellaneous import batch_hfilp,batch_boxlist_hflip
from .roi_mask_feature_extractors import \
    make_roi_mask_feature_extractor, DeeperExtractor,make_mask_adapt_layer
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor,make_roi_mask_generator
from .loss import make_roi_mask_loss_evaluator
from maskrcnn_benchmark.modeling.relation.mask_relation_module import MaskRelationRefineNet
import pdb

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    if boxes[0].has_field("labels"):
        label_field = 'labels'
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, is_student):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg)
        self.predictor = make_roi_mask_predictor(cfg)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.mask_generator = make_roi_mask_generator(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)
        if cfg.MODEL.RELATION_MASK:
            self.mask_relation_module = MaskRelationRefineNet(cfg, self.predictor)
        if cfg.MT.HINT:
            self.feat_adapt_module =make_mask_adapt_layer(cfg)
        self.mode = None
    def set_teacher_mode(self,mode):
        self.mode = mode
    def forward(self, features, proposals,
                targets=None , images = None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.

        """

        if self.training :
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        batch_size = [len(f) for f in proposals]
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            if self.cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR == "PRCNNFeatureExtractor":
                x, pre_feature = self.feature_extractor(images.tensors, proposals)
            else:
                x, pre_feature = self.feature_extractor(features, proposals)

        mask_logits_1 = self.predictor(x)
        if self.training:
            loss_mask_1=  self.loss_evaluator(proposals,
                                               mask_logits_1,
                                               targets)
        if len(batch_size)>1:
            mask_logits_1 = torch.split(mask_logits_1,batch_size)
        else:
            mask_logits_1 = [mask_logits_1]
        if self.cfg.MODEL.RELATION_MASK.USE_RELATION:
            feature = x
            if len(batch_size)>1:
                feature = torch.split(feature,batch_size)
            else:
                feature = [feature]
            mask_logits_2_list = []
            proposals_list = []
            targets_list = []
            relation_loss_list = []
            if targets == None:
                targets = [None ] * len(batch_size)
            for f,m,p,t in zip(feature,mask_logits_1,proposals,
                               targets):
                mask_logits_2, p, t, relation_loss = \
                    self.mask_relation_module((f,m,p,t))
                mask_logits_2_list.append(mask_logits_2)
                proposals_list.extend(p)
                targets_list.append(t)
                relation_loss_list.append(relation_loss)
            # pdb.set_trace()
            proposals = proposals_list
            mask_logits_2 = mask_logits_2_list
            if len(mask_logits_2) > 1:
                mask_logits_2 = torch.cat(mask_logits_2)
            else:
                mask_logits_2 = mask_logits_2[0]

        if not self.training:
            if self.cfg.MODEL.RELATION_MASK.USE_RELATION:
                mask_logits = mask_logits_2
            else:
                mask_logits = mask_logits_1
            if self.mode is None or self.mode =='train': # normal phase, inference
                result = self.post_processor(mask_logits, proposals)
            else:
                # teacher, test mode to generate segmentation mask
                # todo: generate the mask in teacher-train mode with augmented images.
                result = self.mask_generator(mask_logits, proposals)
            return x, result, {}

        # loss
        else:
            if self.cfg.MODEL.RELATION_MASK.USE_RELATION:
                loss_mask_2 = self.loss_evaluator(proposals,
                                                   mask_logits_2,
                                                      targets)
                if self.cfg.MODEL.RELATION_MASK.DEEP_SUPER:
                    loss_seg = 0.5 * (loss_mask_1 + loss_mask_2)
                else:
                    loss_seg = loss_mask_2
            else:
                loss_seg = loss_mask_1

            return x, all_proposals, dict(loss_seg=loss_seg)


def build_roi_mask_head(cfg, is_student = False):
    return ROIMaskHead(cfg, is_student)


