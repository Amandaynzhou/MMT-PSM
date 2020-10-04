# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler)
from maskrcnn_benchmark.modeling.utils import cat

class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,cfg = None):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cfg =  cfg

        if cfg is not None:
            self.temp = cfg.MT.TEMP
            if cfg.MT.CLS_LOSS_TYPE == 'bce':
                self.cls_loss_type = 'ce'
            elif cfg.MT.CLS_LOSS_TYPE =='wbce':
                self.cls_loss_type = 'wce'
            else:
                self.cls_loss_type = cfg.MT.CLS_LOSS_TYPE

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(["labels"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        # scores = []
        # pdb.set_trace()
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
        return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(
            labels)
        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
                labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(
                pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals
        labels = cat(
            [proposal.get_field("labels") for proposal in proposals],
            dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in
             proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1)
        box_loss = box_loss / labels.numel()
        return classification_loss, box_loss

    def _mean_var_logits(self, logits):
        t_logits = torch.stack(logits)
        t_logits = torch.mean(t_logits, dim = 0)
        if self.cfg.MT.CLS_LOSS_TYPE == 'bce':
            logits = [F.softmax(l, dim = 1) for l in logits]
        logits = torch.stack(logits)
        m_logit = torch.mean(logits, dim = 0)
        v_logit = torch.std(logits,dim=0)

        return m_logit, v_logit, t_logits

    def _mean_var_regression(self,regressions):
        for i, reg in enumerate(regressions):
            if i % 2 == 1:
                reg[:, 0::4] = - reg[:, 0::4]

        regressions = torch.stack(regressions)
        m_regression = torch.mean(regressions, dim=0)
        v_regression = torch.std(regressions, dim=0)
        return m_regression, v_regression

    def evaluatePSM(self,class_logits, class_logits_t,proposals):

        labels = cat([proposal.get_field("labels")for proposal in proposals], dim=0)
        m_logit_t, v_logit_t, m_logit_supervised = self._mean_var_logits(class_logits_t)
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        sampled_neg_inds_subset = torch.nonzero(labels == 0).squeeze(1)
        v_logit_t_p = v_logit_t[sampled_pos_inds_subset]
        v_logit_t_n = v_logit_t[sampled_neg_inds_subset]
        v_logit_t_p = v_logit_t_p.sum(-1)
        v_logit_t_n = v_logit_t_n.sum(-1)
        m_logit_supervised_p = m_logit_supervised[
            sampled_pos_inds_subset]
        m_logit_supervised_n = m_logit_supervised[
            sampled_neg_inds_subset]
        # currently we do not use regression loss
        cls_loss = []
        for i, class_logit in enumerate(class_logits):
            if self.cfg.MT.RANK_FILTER > 0:
                # choose pos:neg = 2:1
                # if hard_negative:  chooses hard negative and give
                #  class weight= CLS_BALANCE_WEIGHT
                # else: class weight = 1.
                if self.cfg.MT.HARD_NEG:
                    subsample_cls_msk_n = torch.argsort(v_logit_t_n,
                                              descending=self.cfg.MT.HARD_NEG)
                else:
                    subsample_cls_msk_n = torch.randperm(v_logit_t_n.shape[0])
                subsample_cls_msk_n_remain = subsample_cls_msk_n[:
            min(subsample_cls_msk_n.shape[0], int(v_logit_t_p.shape[0] / 2))]
                m_logit_supervised_n_remain = m_logit_supervised_n[
                    subsample_cls_msk_n_remain]
                p_n_number = [m_logit_supervised_p.shape[0],
                              m_logit_supervised_n_remain.shape[0]]
                m_logit_t = torch.cat([m_logit_supervised_p,
                                       m_logit_supervised_n_remain])


                class_logits_p = class_logit[sampled_pos_inds_subset]
                class_logits_n = class_logit[sampled_neg_inds_subset][
                    subsample_cls_msk_n_remain]
                class_logit_s = torch.cat(
                    [class_logits_p, class_logits_n])
            else:
                # use all samples
                class_logit_s = class_logit
                p_n_number = None

            loss = self.cls_loss(class_logit_s, m_logit_t.clone(),
                                 p_n_number, cls_balance_weight=1
                if not self.cfg.MT.HARD_NEG else self.cfg.MT.CLS_BALANCE_WEIGHT)
            cls_loss.append(loss)
        cls_loss = torch.mean(torch.stack(cls_loss), dim=0)
        return cls_loss

    def evaluateODKD(self, logits,proposals,class_logits_t):
        # ODKD means the framework do not use k-augmented samples
        # and estimate the certainty for further loss calculation.
        # instead, the image is augmented once and feed into the
        # student to calculate the loss with all the samples
        # here the weight is set 1.5, which is same as we used in PSM
        logits = logits[0]
        class_logits_t = class_logits_t[0]
        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        device = logits.device
        sampled_neg_inds_subset = torch.nonzero(labels == 0).squeeze(1)
        log_probs = F.log_softmax(logits, dim=1)
        teacher = F.softmax(class_logits_t, dim=1)
        weight =  torch.ones((labels.shape[0])).to(device)
        weight[sampled_neg_inds_subset]  = 1.5
        loss = (- teacher.detach() * log_probs * weight[:, None]).mean()
        return loss

    # help functions
    def _acc_cls_pos(self,gt, logits, thread):
        p =  (gt!=logits)[:int(thread * gt.shape[0])].sum().cpu().numpy()/(thread* gt.shape[0])
        return p

    def _dist(self,gt, regression, thread):
        size = int(gt.shape[0] * thread)
        dist = F.mse_loss(regression[:size].clone(),    gt[:size].clone())
        return dist

    def cls_loss(self, logit, teacher, p_n_number= None,
                 cls_balance_weight= 1.):
        if self.cls_loss_type == 'kl':
            teacher =F.softmax(teacher,dim=1)
            logit = F.log_softmax(logit,dim=1)
            loss = F.kl_div(logit, teacher)
        elif self.cls_loss_type == 'mse':
            loss = F.mse_loss(logit, teacher.detach())
        else:
            log_probs = F.log_softmax(logit, dim = 1)
            teacher = F.softmax(teacher, dim=1)
            if self.cfg.MT.SHARPEN:
                teacher = sharpen(teacher, temp=self.temp)
            if p_n_number is None:
                loss = (- teacher.detach() * log_probs).mean(0).sum()/3
            else:
                # pdb.set_trace()
                weight = torch.ones(logit.shape[0]).to(logit.device)
                weight[p_n_number[0]:] = cls_balance_weight
                loss = (- teacher.detach() * log_probs * weight[:,None]).mean(0).sum() / 3
        return loss

def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,top_k =
        cfg.MODEL.ROI_BOX_HEAD.K_HEAD
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)

    loss_evaluator = FastRCNNLossComputation(matcher,
                                         fg_bg_sampler,
                                         box_coder,
                                         cfg = cfg)

    return loss_evaluator

def sharpen(p,temp = 0.5):
    pt = p ** (1 / temp)
    targets_u = pt / pt.sum(dim=1, keepdim=True)
    targets_u = targets_u.detach()
    return targets_u
