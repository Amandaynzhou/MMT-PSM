# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
import numpy as np
import pdb
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
            is_teacher= False,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.is_teacher = is_teacher

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness,
                                       box_regression,i):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        objectness = objectness.permute(0, 2, 3, 1).reshape(N, -1)
        objectness = objectness.sigmoid()
        box_regression = box_regression.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2)
        box_regression = box_regression.reshape(N, -1, 4)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        # import pdb;pdb.set_trace()
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]
        # batch_ = batch_idx.expand([N ,pre_nms_top_n])

        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)
        # import pdb;pdb.set_trace()
        result = []
        for j, (proposal, score, im_shape, topk_id, ) in enumerate(
                zip(
                proposals, objectness, image_shapes, topk_idx, )):
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            boxlist.add_field("objectness", score)
            if self.is_teacher:
                # boxlist.add_field("bid", batch_[j])
                boxlist.add_field("box_reg", box_regression[j])
                boxlist.add_field("rpn_topk", topk_id)
                boxlist.add_field("rpn_ancher_level", torch.tensor([i]
        *topk_id.shape[0],  device = device))
            boxlist = boxlist.clip_to_image(remove_empty=False)

            boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist = boxlist_nms(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for i, (a, o, b) in enumerate( zip(anchors, objectness, \
                box_regression)):

            sampled_boxes.append(
                self.forward_for_single_feature_map(a, o, b, i ))

        boxlists = list(zip(*sampled_boxes))
        # import pdb;pdb.set_trace()

        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        # todo: check if teacher-train mode can add these gts
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists
    #
    def forward_imitation(self, anchors, objectness,targets, s):
        # code to find bbox region in FFI method
        canvas_list = []
        for j , (a_per_img, o_per_img,tar) in enumerate(zip(anchors,objectness,targets)):
            anchor_list= []
            for i, (a , o )in enumerate( zip(a_per_img,o_per_img)):
                iou = boxlist_iou(a,tar)
                iou[iou<=s] = 0.
                preserve = torch.nonzero(iou)[:,0]
                preserve_anchor = a[preserve].clip_to_image()
                anchor_list.append(preserve_anchor)
            anchor_list = cat_boxlist(anchor_list)
            canvas = torch.zeros(a.size).to(device=o_per_img.device)
            for bbox in anchor_list.bbox:
                bbox = bbox.long()
                canvas[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            canvas_list.append(canvas)
        return canvas_list

    def collect_batch_single_feature_idx(self, teacher_sample_list,
                                         targets):
        batch_idx = []
        # batch_ori_idx = []
        sort_target = []
        for i in range(5):
            feature_idxs = []
            targets_list = []
            # pdb.set_trace()
            for per_img_sample, per_img_target in zip(
                    teacher_sample_list, targets):
                feature_idxs.append(
                per_img_sample[1][(per_img_sample[0] == i).nonzero().squeeze()])
                targets_list.append(
                  per_img_target[(per_img_sample[0] == i).nonzero().squeeze()]
                )
            batch_idx.append(feature_idxs)
            sort_target.append(targets_list)
            # batch_ori_idx.append(ori_idx)
        return batch_idx, sort_target



    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # TODO resolve this difference and make it consistent. It should be per image,
        # and not per batch
        if self.training:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train,
                           is_teacher = False):
    # if is_teacher == True:
    #     # FORCE TRUE
    #     is_train=True
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        is_teacher=is_teacher,

    )
    return box_selector
