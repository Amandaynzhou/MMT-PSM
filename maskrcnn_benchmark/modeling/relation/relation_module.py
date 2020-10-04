import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, boxlist_iou
from maskrcnn_benchmark.modeling.python_nms import cyto_nms
import pdb
# F.kl_div
class RelationModule(nn.Module):
    ''' Multi-head self attention relation module'''
    def __init__(self, appearance_feature_dim=1024, geo_feature_dim = 64 ,
                 fc_dim=(64, 16), group=16, dim=(1024, 1024, 1024), topk  = 10, iou_method  = 'b'):
        super(RelationModule, self).__init__()
        self.fc_dim = fc_dim
        self.dim_group =   (int(dim[0] / group), int(dim[1] / group), int( dim[2] / group))
        self.dim = dim
        self.group = group
        self.WG = nn.Linear(geo_feature_dim, fc_dim[1], bias=True)
        # 1024, 1024
        self.WK = nn.Linear(appearance_feature_dim, dim[1], bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, dim[0], bias=True)
        # self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.fc_dim[1] * appearance_feature_dim, dim[2], 1, groups=group  )
        self.topk  = topk
        self.iou_method = iou_method
        assert fc_dim[1] == group, 'Check the dimensions in attention!'

    def forward(self, f_a, position_embedding, iou):
        # f_a: [num_rois, num_fg_classes, feat_dim]
        N, num_fg_class, feat_dim = f_a.size()
        # f_a = f_a.transpose(0, 1)
        f_a = f_a.permute(1,0,2)
        # f_a_reshape [num-roi*num-fg-cls, feat-dim]
        f_a_reshape = f_a.contiguous().view( N*num_fg_class, feat_dim)
        # [num_fg_classes * num_rois * num_rois, fc_dim[0]]
        position_embedding = position_embedding.view(-1,self.fc_dim[0])
        w_g = self.relu(self.WG(position_embedding))
        w_k = self.WK(f_a_reshape)
        # [ num_rpi, 16, 64 ]
        w_k = w_k.view(-1, N, self.group ,self.dim_group[1])
        w_k = w_k.permute(0,2,3,1)
        # k_data_batch, [num_fg_classes * group, dim_group[1], num_rois,]
        w_k = w_k.contiguous().view(-1,  self.dim_group[1], N )
        w_q = self.WQ(f_a_reshape)
        w_q = w_q.view(-1, N,self.group, self.dim_group[0])
        w_q = w_q.transpose( 1, 2)
        # q_data_batch, [num_fg_classes * group, num_rois, dim_group[0]]
        w_q = w_q.contiguous().view(-1, N, self.dim_group[0] )
        # aff, [num_fg_classes * group, num_rois, num_rois]
        aff = (1.0 / math.sqrt(float(self.dim_group[1]))) * torch.bmm(w_q, w_k)
        w_g = w_g.view(-1, N, N, self.fc_dim[1])
        #  [num_fg_classes, fc_dim[1], num_rois, num_rois]
        w_g = w_g.permute(0,3,1,2)
        #  [num_fg_classes * fc_dim[1], num_rois, num_rois]
        w_g = w_g.contiguous().view(-1 ,N ,N)
        if iou is not None:
            iou = torch.cat([iou[0][None, :, :], iou[1][None, :, :]])
            if self.iou_method == 's':
                log_iou = torch.log(iou + 1)
            elif self.iou_method == 'h':
                log_iou = torch.log(torch.clamp(iou, min = 1e-6))
            else:
                iou[iou>=1e-6] = 1
                log_iou = torch.log(torch.clamp(iou, min = 1e-6))
            log_iou = log_iou[:,None,:,:].repeat(1, self.group,1,1).view(-1, N, N)
            w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + aff + log_iou
        else:
            w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + aff
        top_k = min(N, self.topk)
        w_mn_topk, indices = torch.topk(w_mn, top_k, dim=2,
                                        largest=True, sorted=True)
        res = torch.zeros_like(w_mn).to(w_mn.device)
        w_mn_topk = F.softmax(w_mn_topk, dim = 2)
        w_mn_topk = res.scatter(2, indices, w_mn_topk)
        w_mn_topk = w_mn_topk.view(num_fg_class, -1, N)
        output = torch.bmm(w_mn_topk, f_a)
        output = output.view(num_fg_class, self.fc_dim[1], N, feat_dim)
        output = output.permute(1, 3, 2, 0)
        # output_t_reshape, [1, fc_dim[1] * feat_dim, num_rois, num_fg_classes]
        output = output.contiguous().view(1, self.fc_dim[1] * feat_dim, N, -1)
        # [1, 128, nroi, cls]
        output = self.conv1(output)
        output = output.squeeze()
        output = output.permute(1,2,0)
        return output
#


class DuplicationRemovalNetwork(nn.Module):
    def __init__(self, cfg, is_teacher=False, ):
        super(DuplicationRemovalNetwork, self).__init__()
        self.cfg = cfg.clone()
        # if reg_iou = True, then this network is used to regress
        # the iou to the GT. if not True, this predict
        # true-object/duplicate
        self.reg_iou = self.cfg.MODEL.RELATION_NMS.REG_IOU
        self.first_n = cfg.MODEL.RELATION_NMS.FIRST_N
        self.NMS_thread = cfg.MODEL.RELATION_NMS.THREAD
        self.nms_rank_fc = nn.Linear(cfg.MODEL.RELATION_NMS.ROI_FEAT_DIM, cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM, bias=True)
        self.roi_feat_embedding_fc = nn.Linear(cfg.MODEL.RELATION_NMS.ROI_FEAT_DIM, cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM, bias=True)
        self.target_thresh = cfg.MODEL.RELATION_NMS.THREAD
        self.geo_feature_dim = cfg.MODEL.RELATION_NMS.GEO_FEAT_DIM

        if cfg.MODEL.RELATION_NMS.USE_IOU:
            self.geo_feature_dim = int(self.geo_feature_dim/4 * 5)
        self.relation_module = RelationModule(cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM,
                                                  geo_feature_dim=self.geo_feature_dim,
                                                  fc_dim= (self.geo_feature_dim, 16),
                                                  group=cfg.MODEL.RELATION_NMS.GROUP,
                                                  dim=cfg.MODEL.RELATION_NMS.HID_DIM,
                                                  topk = cfg.MODEL.RELATION_NMS.TOPK,
                                                  iou_method= cfg.MODEL.RELATION_NMS.IOU_METHOD)

        self.nms_fg_weight = torch.tensor([1., cfg.MODEL.RELATION_NMS.WEIGHT])
        self.mt_fg_weight = torch.tensor([1.,10.])
        self.alpha = cfg.MODEL.RELATION_NMS.ALPHA
        self.gamma = cfg.MODEL.RELATION_NMS.GAMMA
        self.boxcoder = BoxCoder(weights=(10., 10., 5., 5.))
        self.class_agnostic = cfg.MODEL.RELATION_NMS.CLASS_AGNOSTIC
        self.fg_class = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES -1
        self.classifier =  nn.Linear(128, len(self.target_thresh), bias= True)
        self.relu1 = nn.ReLU(inplace=True)
        self.fg_thread = cfg.MODEL.RELATION_NMS.FG_THREAD
        self.detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
        self.nms = cfg.MODEL.RELATION_NMS.POS_NMS
        self.nms_loss_type = cfg.MT.NMS_LOSS_TYPE
        self.mode = None

    def set_teacher_mode(self, mode):
        self.mode = mode

    def forward(self, x):
        appearance_feature, proposals, cls_score, box_reg, targets = x
        self.device = appearance_feature.device
        appearance_feature = appearance_feature
        cls_score = cls_score
        box_reg = box_reg

        with torch.no_grad():
            sorted_boxlists = self.prepare_ranking(cls_score,
                                                   box_reg,
                                                   proposals,
                                                   targets,
                                                   reg_iou=self.reg_iou)
        # concate value from different images
        boxes_per_image = [len(f) for f in proposals]
        idxs =  [f.get_field('sorted_idx') for f in sorted_boxlists]
        scores = torch.cat([f.get_field('scores') for f in sorted_boxlists])
        bboxes = torch.cat([f.bbox.reshape(-1,self.fg_class,4) for f in sorted_boxlists])
        objectness = torch.cat([f.get_field('objectness').reshape(-1, self.fg_class) for f in sorted_boxlists])
        all_scores = torch.cat([f.get_field('all_scores') for f in
             sorted_boxlists])

        # add iou information
        image_sizes = [f.size for f in sorted_boxlists]
        sorted_boxes_per_image = [[*f.shape][0] for f in idxs]
        appearance_feature = self.roi_feat_embedding_fc(appearance_feature)
        appearance_feature = appearance_feature.split(boxes_per_image, dim=0)
        sorted_features  = []
        nms_rank_embedding = []
        for id, feature, box_per_image in zip(idxs, appearance_feature, boxes_per_image):
            feature = feature[id]
            size = feature.size()
            if size[0] <= self.first_n:
                first_n = size[0]
            else:
                first_n = self.first_n
            sorted_features.append(feature)
            #[rank_dim * batch , feat_dim]
            nms_rank_embedding.append( extract_rank_embedding(first_n, self.cfg.MODEL.RELATION_NMS.ROI_FEAT_DIM, device = feature.device))
        #  [first_n * batchsize, num_fg_classes, 128]
        sorted_features = torch.cat(sorted_features, dim= 0)
        nms_rank_embedding = torch.cat(nms_rank_embedding, dim = 0)
        nms_rank_embedding = self.nms_rank_fc(nms_rank_embedding)
        sorted_features = sorted_features + nms_rank_embedding[:,None,:]

        boxes_cls_1 = BoxList(bboxes[ :, 0, :], image_sizes[0])
        boxes_cls_2 = BoxList(bboxes[ :, 1, :], image_sizes[0])
        iou_1 = boxlist_iou(boxes_cls_1, boxes_cls_1)
        iou_2 = boxlist_iou(boxes_cls_2, boxes_cls_2)
        if self.cfg.MODEL.RELATION_NMS.USE_IOU:
            iou = [iou_1, iou_2]
        else:
            iou = None
        nms_position_matrix = extract_multi_position_matrix(bboxes, None, self.geo_feature_dim, 1000,
                                                            clswise = self.cfg.MODEL.RELATION_NMS.CLS_WISE_RELATION,)
        nms_attention_1 = self.relation_module(sorted_features,  nms_position_matrix, iou)
        sorted_features = sorted_features + nms_attention_1
        sorted_features = self.relu1(sorted_features)
        # [first_n * num_fg_classes, 128]
        sorted_features = sorted_features.view(-1, self.cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM)
        sorted_features = self.classifier(sorted_features)
        # logit_reshape, [first_n, num_fg_classes, num_thread]
        sorted_features = sorted_features.view(-1, self.fg_class, len(self.target_thresh))
        if not self.reg_iou:
            sorted_features = torch.sigmoid(sorted_features)
        scores = torch.cat([scores[:,:,None]]*len(self.target_thresh), dim = -1)
        loss_dict = {}
        if self.training:
            if self.reg_iou:
                # when use regression donot do sorted_features = scores * sorted_features
                reg_label = torch.cat([f.get_field('labels_iou_reg') for f in sorted_boxlists])
                reg_label = reg_label.to(scores.device)
                reg_label = reg_label.type(torch.cuda.FloatTensor)
                sorted_features = sorted_features.to(scores.device)
                sorted_features = sorted_features.type(torch.cuda.FloatTensor)
                if reg_label.shape is not None:
                    reg_iou_loss = F.mse_loss(reg_label,sorted_features)
                else:
                    reg_iou_loss = torch.tensor(0.).to(scores.device)
                loss_dict['nms_loss'] = reg_iou_loss
            else:
                sorted_features = scores * sorted_features
                labels = torch.cat([f.get_field('labels') for f in sorted_boxlists])

                labels = labels.to(scores.device)
                labels = labels.type(torch.cuda.FloatTensor)

                # WEIGHTED NMS
                nms_loss = F.binary_cross_entropy(scores*sorted_features,labels)
                loss_dict['nms_loss']=nms_loss
            return None, loss_dict
        else:
            input_scores = scores
            if self.reg_iou:
                scores = sorted_features* (scores>self.fg_thread).float()
            else:
                scores = sorted_features * scores
            scores = self.merge_multi_thread_score_test(scores)
            scores = scores.split(sorted_boxes_per_image, dim = 0)
            bboxes = bboxes.split(sorted_boxes_per_image, dim = 0)
            input_scores = input_scores.split(sorted_boxes_per_image, dim = 0)
            objectness = objectness.split(sorted_boxes_per_image, dim = 0)
            all_scores = all_scores.split(sorted_boxes_per_image,
                                          dim = 0)
            result = []
            for i_score, score, bbox, obj, image_size, prob_boxhead \
                    in zip(
                                                    input_scores,
                                                        scores,
                                                        bboxes,
                                                    objectness,
                                                    image_sizes, all_scores):
                result_per_image = []
                # for nuclei
                index = (score[:, 1] >= self.fg_thread).nonzero()[:,
                        0]
                # cls_scores = i_score[index, i,0]
                cls_scores = score[index, 1]
                cls_scores_all = prob_boxhead[index, 1]
                cls_boxes = bbox[index, 1, :]
                cls_obj = obj[index, 1]

                boxlist_for_class = BoxList(cls_boxes, image_size,
                                            mode='xyxy')

                boxlist_for_class.add_field('scores', cls_scores)
                boxlist_for_class.add_field('objectness', cls_obj)
                boxlist_for_class.add_field('all_scores',
                                            cls_scores_all)
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, 0.5, score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field("labels",
                                            torch.full((
                                                num_labels,),
                                                2,
                                                dtype=torch.int64).to(
                                                self.device))
                result_per_image.append(boxlist_for_class)
                index = (score[:, 0 ] >=
                         self.fg_thread).nonzero()[:,0]
                # cls_scores = i_score[index, i,0]
                cls_scores = score[index,0]
                # pdb.set_trace()

                cls_scores_all = prob_boxhead[index, 0]
                cls_boxes = bbox[index, 0, :]
                cls_obj = obj[index, 0 ]

                boxlist_for_class = BoxList(cls_boxes, image_size, mode='xyxy')
                # Pos greedy NMS if POS_NMS!=-1
                # boxlist_for_class.add_field('idx', index)
                boxlist_for_class.add_field('scores', cls_scores)
                boxlist_for_class.add_field('objectness', cls_obj)
                boxlist_for_class.add_field('all_scores', cls_scores_all)
                # pdb.set_trace()
                if self.nms:
                    # for nuclei
                    boxlist_for_class = boxlist_nms(
                        boxlist_for_class, self.nms, score_field="scores"
                    )
                # pdb.set_trace()
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field("labels",
                                            torch.full((
                                                num_labels,),
                                                1,
                                                dtype=torch.int64).to(self.device))
                result_per_image.append(boxlist_for_class)
                result_per_image = cat_boxlist(result_per_image)
                number_of_detections = len(result_per_image)

                # Limit to max_per_image detections **over all classes**
                if number_of_detections > self.detections_per_img > 0:
                    cls_scores = result_per_image.get_field("scores")
                    image_thresh, _ = torch.kthvalue(
                        cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
                    )
                    keep = cls_scores >= image_thresh.item()
                    keep = torch.nonzero(keep).squeeze(1)
                    result_per_image = result_per_image[keep]
                result.append(result_per_image)

            return result, {}

    def prepare_reg_label(self, sorted_boxes, sorted_score, targets):
        '''

        :param sorted_boxes: [ first n, fg_cls_num, 4]
        :param indice: [first n, fg_cls_num]
        :param sorted_score: [first n, fg_cls_num]
        :param targets: Boxlist obj
        :return: label [first n, num_thread * fg_cls_num]
        '''
        TO_REMOVE = 1
        labels = targets.get_field('labels')

        # output = np.zeros((sorted_boxes.shape[0].numpy(),))
        # pdb.set_trace()
        # output_list = []
        output_reg_list = []
        for i in range(self.fg_class):
            cls_label_indice = torch.nonzero(labels == (i+1))
            cls_target_bbox = targets.bbox[cls_label_indice[:,0]]

            # todo: avoid None gt situation
            num_valid_gt = len(cls_label_indice)

            if num_valid_gt == 0:

                output = np.zeros(([*sorted_boxes.shape][0],len(self.target_thresh)))
                # output_reg = output.copy()
                # output_list.append(output)
                output_reg_list.append(output)
            else:
                output_list_per_class = []
                output_reg_list_per_class = []
                eye_matrix = np.eye(num_valid_gt)
                score_per_class = sorted_score[:, i: i + 1].cpu().numpy()
                boxes = sorted_boxes[:, i, :]
                boxes = boxes.view(-1, 4)
                area1 = (boxes[:, 2] - boxes[:, 0] + TO_REMOVE) * (boxes[:, 3] - boxes[:, 1] + TO_REMOVE)
                area2 = (cls_target_bbox[:, 2] - cls_target_bbox[:, 0] + TO_REMOVE) * (cls_target_bbox[:, 3] - cls_target_bbox[:, 1] + TO_REMOVE)
                lt = torch.max( boxes[:,None,:2],cls_target_bbox[:, :2])  # [N,M,2]
                rb = torch.min( boxes[:,None,2:],cls_target_bbox[:, 2:])  # [N,M,2]
                wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
                inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
                # [num_gt, first_n]
                iou = inter / (area1[:, None] + area2 - inter)
                iou = iou.cpu().numpy()
                try:
                    for thresh in self.target_thresh:
                        # pdb.set_trace()
                        output_reg = np.max(iou, 1)
                        # todo: temp comment
                        overlap_mask = (iou > thresh)
                        overlap_iou = iou * overlap_mask
                        valid_bbox_indices = np.where(overlap_mask)[0]
                        overlap_score = np.tile(score_per_class, (1, num_valid_gt))
                        overlap_score *= overlap_mask
                        max_overlap_indices = np.argmax(iou, axis=1)
                        max_overlap_mask = eye_matrix[max_overlap_indices]
                        overlap_score *= max_overlap_mask
                        overlap_iou =overlap_iou * max_overlap_mask
                        max_score_indices = np.argmax(overlap_score, axis=0)
                        max_overlap_iou = overlap_iou[max_score_indices, np.arange(overlap_score.shape[1])]
                        # output = np.zeros(([*sorted_boxes.shape][0],))
                        output_reg = np.zeros(([*sorted_boxes.shape][0],))
                        output_idx, inter_1, inter_2  = np.intersect1d(max_score_indices, valid_bbox_indices,return_indices=True)
                        # output[output_idx] = 1
                        output_reg[output_idx] = max_overlap_iou[inter_1]
                        # output_list_per_class.append(output)
                        output_reg_list_per_class.append(output_reg)
                except:
                    pdb.set_trace()
                # output_per_class = np.stack(output_list_per_class, axis=-1)
                output_reg_per_class  =np.stack(output_reg_list_per_class, axis=-1)
                # pdb.set_trace()
                # output_list.append(output_per_class.view())
                output_reg_list.append(output_reg_per_class)

        # output =  np.stack(output_list, axis=1).astype(np.float32, copy=False)
        output_reg = np.stack(output_reg_list, axis=1).astype(np.float32, copy=False)
        return output_reg
        # return (output, output_reg)

    def prepare_label(self, sorted_boxes, sorted_score, targets):
        '''

        :param sorted_boxes: [ first n, fg_cls_num, 4]
        :param indice: [first n, fg_cls_num]
        :param sorted_score: [first n, fg_cls_num]
        :param targets: Boxlist obj
        :return: label [first n, num_thread * fg_cls_num]
        '''
        TO_REMOVE = 1
        labels = targets.get_field('labels')

        # output = np.zeros((sorted_boxes.shape[0].numpy(),))

        output_list = []
        for i in range(self.fg_class):
            cls_label_indice = torch.nonzero(labels == (i+1))
            cls_target_bbox = targets.bbox[cls_label_indice[:,0]]

            # todo: avoid None gt situation
            num_valid_gt = len(cls_label_indice)

            if num_valid_gt == 0:

                output = np.zeros(([*sorted_boxes.shape][0],len(self.target_thresh)))
                output_list.append(output)
            else:
                output_list_per_class = []
                eye_matrix = np.eye(num_valid_gt)
                score_per_class = sorted_score[:, i: i + 1].cpu().numpy()
                boxes = sorted_boxes[:, i, :]
                boxes = boxes.view(-1, 4)
                area1 = (boxes[:, 2] - boxes[:, 0] + TO_REMOVE) * (boxes[:, 3] - boxes[:, 1] + TO_REMOVE)
                area2 = (cls_target_bbox[:, 2] - cls_target_bbox[:, 0] + TO_REMOVE) * (cls_target_bbox[:, 3] - cls_target_bbox[:, 1] + TO_REMOVE)
                lt = torch.max( boxes[:,None,:2],cls_target_bbox[:, :2])  # [N,M,2]
                rb = torch.min( boxes[:,None,2:],cls_target_bbox[:, 2:])  # [N,M,2]
                wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
                inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
                # [num_gt, first_n]
                iou = inter / (area1[:, None] + area2 - inter)
                iou = iou.cpu().numpy()

                for thresh in self.target_thresh:
                    overlap_mask = (iou > thresh)
                    valid_bbox_indices = np.where(overlap_mask)[0]
                    overlap_score = np.tile(score_per_class, (1, num_valid_gt))
                    overlap_score *= overlap_mask
                    max_overlap_indices = np.argmax(iou, axis=1)
                    max_overlap_mask = eye_matrix[max_overlap_indices]
                    overlap_score *= max_overlap_mask
                    max_score_indices = np.argmax(overlap_score, axis=0)
                    output = np.zeros(([*sorted_boxes.shape][0],))
                    output[np.intersect1d(max_score_indices, valid_bbox_indices)] = 1
                    output_list_per_class.append(output)
                output_per_class = np.stack(output_list_per_class, axis=-1)
                output_list.append(output_per_class)
        output =  np.stack(output_list, axis=1).astype(np.float32, copy=False)
        return output

    def prepare_ranking(self, cls_score, box_regression,
                        proposals, targets, reg_iou = False):
        '''
        :param score:[num_per_img*batchsize, class]
        :param proposal: list of boxlist
        :return:
        '''
        # if is not train, targets is None which should be set into a none list

        boxes_per_image = [len(box) for box in proposals]
        concat_boxes = torch.cat([a.bbox for a in proposals], dim=0)
        image_shapes = [box.size for box in proposals]
        objectness = [f.get_field('objectness') for f in proposals]
        proposals = self.boxcoder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        proposals = proposals.split(boxes_per_image, dim=0)
        cls_score = cls_score.split(boxes_per_image, dim=0)
        results = []
        if self.training:
        # if idx_t is None:
            for prob, boxes_per_img, image_shape,  target,  obj in zip(
                    cls_score, proposals, image_shapes, targets, objectness):

                boxlist = self.filter_results(boxes_per_img,
                                  target, prob,
                                  image_shape,
                                  self.fg_class+1, obj, reg_iou)

                results.append(boxlist)
        else:
            # test do not have target
            for prob, boxes_per_img, image_shape, obj in zip(
                    cls_score, proposals, image_shapes, objectness
            ):
                boxlist = self.filter_results(boxes_per_img, None, prob, image_shape,  self.fg_class+1, obj,reg_iou=reg_iou)
                results.append(boxlist)

        return results

    def filter_results(self, boxes,  targets, scores, image_shape ,
                       num_classes , obj, reg_iou = False):
        """return the sorted boxlist and sorted idx
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        # boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        #[n_roi, 4, cls]
        # boxes = boxlist.bbox.reshape(-1, 4, num_classes)

        boxes = boxes.reshape(-1, 4 * num_classes)
        scores = scores.reshape(-1, num_classes)
        # pdb.set_trace()
        if scores.shape[0] == 0:
            pdb.set_trace()
        cat_boxes = []
        for j in range(1,num_classes):
            # skip class 0, because it is the background class
            cls_boxes = boxes[:, j* 4 : (j + 1) * 4]
            cat_boxes.append(cls_boxes)
        boxes = torch.cat([bbox[:,:,None] for bbox in cat_boxes], dim = 2)
        # scores =  torch.cat([s for s in cat_score])
        scores = scores[:,1:]
        ori_scores = scores
        num_roi = boxes.shape[0]
        if num_roi<= self.first_n:
            first_n = num_roi
            # pdb.set_trace()
        else:
            first_n = self.first_n



        sorted_scores, indices = torch.topk(scores, first_n, dim= 0, largest = True, sorted = True)

        if obj.shape[0]<first_n:
            indices = indices[:obj.shape[0]]
            sorted_scores = sorted_scores[:obj.shape[0]]
        if indices.shape[1] !=2:
            pdb.set_trace()
        cp_s = ori_scores.clone().cpu().numpy()
        cp_o = obj.clone().cpu().numpy()
        box = boxes.clone().cpu().numpy()
        ori_scores = ori_scores[indices]
        sorted_obj = obj[indices]
        sorted_boxes = boxes[indices]



        if sorted_boxes.shape[0] == 0:
            pdb.set_trace()

        if self.class_agnostic:
            # [first_n, num_fg_class, 4]
            sorted_boxes = torch.squeeze(sorted_boxes, dim = -1)
        else:
            try:
                mask = torch.arange(0, num_classes - 1).to(device=self.device)
            except:
                pdb.set_trace()
            try:
                mask = mask.view(1,-1,1,1).expand(first_n, num_classes-1, 4 , 1)
            except:
                pdb.set_trace()
            sorted_boxes = torch.gather(sorted_boxes, dim = 3, index = mask).squeeze(dim = 3)
        if self.training:
            labels = self.prepare_label(sorted_boxes,
                                        sorted_scores, targets)
            labels_cls = torch.from_numpy(labels).to(sorted_scores.device)
            if reg_iou:
                labels_reg = self.prepare_reg_label(sorted_boxes, sorted_scores,
                                     targets)
                labels_reg = torch.from_numpy(labels_reg).to(sorted_scores.device)
        sorted_boxes = sorted_boxes.view(first_n * (num_classes-1), -1)
        sorted_obj = sorted_obj.view(first_n * (num_classes-1))
        boxlist = BoxList(sorted_boxes,  image_shape , mode="xyxy",)
        boxlist.add_field('sorted_idx', indices)
        boxlist.add_field('objectness', sorted_obj)
        boxlist.extra_fields['scores'] =sorted_scores
        boxlist.extra_fields["all_scores"] = ori_scores
        # boxlist.extra_fields[""]
        if self.training:
            if reg_iou:
                boxlist.extra_fields['labels_iou_reg'] = labels_reg
            else:
                boxlist.extra_fields['labels'] = labels_cls
        boxlist = boxlist.clip_to_image(remove_empty=False)
        return boxlist

    def merge_multi_thread_score_test(self, scores):
        if self.cfg.MODEL.RELATION_NMS.MERGE_METHOD == -1:
            scores = torch.mean(scores, -1)
        elif self.cfg.MODEL.RELATION_NMS.MERGE_METHOD == -2:
            scores = torch.max(scores, -1)
        else:
            idx = self.cfg.MODEL.RELATION_NMS.MERGE_METHOD
            idx = min(max(idx, 0), len(self.target_thresh))
            scores = scores[:, :, idx]
        return scores

### help function ###
def extract_rank_embedding(rank_dim, feat_dim, wave_length=1000, device = 'cpu'):
    """ Extract rank embedding
    Args:
        rank_dim: maximum of ranks
        feat_dim: dimension of embedding feature
        wave_length:
    Returns:
        embedding: [rank_dim, feat_dim]
    """
    rank_range = torch.arange(0, rank_dim).to(device).float()
    feat_range = torch.arange(feat_dim / 2).to(device)
    dim_mat = feat_range / (feat_dim / 2)
    dim_mat = 1. / (torch.pow(wave_length, dim_mat))
    dim_mat = dim_mat.view(1, -1)
    rank_mat = rank_range.view(-1, 1)
    mul_mat = rank_mat * dim_mat
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    embedding = embedding.to(device)
    return embedding

def extract_multi_position_matrix(boxes,iou, dim_g, wave_len, clswise = False):
    if iou is not None:
        iou = torch.cat([iou[0][None,:,:], iou[1][None,:,:]], 0)[:,:,:,None]
    boxes = boxes.permute(1, 0, 2)

    if  clswise:
        # [cls * nroi, 1, 4]
        boxes = boxes.reshape(1, -1, 4)
    x_min, y_min, x_max, y_max = torch.chunk(boxes, 4, dim=2)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = cx - cx.permute(0,2,1)
    delta_x = torch.clamp(torch.abs(delta_x/ w), min = 1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.permute(0,2,1)
    delta_y = torch.clamp(torch.abs(delta_y/ h), min = 1e-3)
    delta_y = torch.log(delta_y)

    delta_w = w / w.permute(0,2,1)
    delta_w = torch.log(delta_w)

    delta_h = h / h.permute(0,2,1)
    delta_h = torch.log(delta_h)

    size = delta_h.size()

    delta_x = delta_x.view(size[0], size[1],size[2], 1)
    delta_y = delta_y.view(size[0], size[1],size[2], 1)
    delta_w = delta_w.view(size[0], size[1],size[2], 1)
    delta_h = delta_h.view(size[0], size[1],size[2], 1)
    # clsn, nrio, nrio, 4

    if iou is not None:
        position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h, iou), -1)
    else:
        position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
    dev = 10 if iou is not None else 8
    # pdb.set_trace()
    feat_range = torch.arange(dim_g / dev).to(boxes.device)
    dim_mat = feat_range / (dim_g / dev)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))
    dim_mat = dim_mat.view(1, 1, 1, 1,-1)
    if iou is not None:
        position_mat = position_mat.view(size[0], size[1], size[2], 5, -1)
    else:
        position_mat = position_mat.view(size[0], size[1], size[2], 4, -1)
    position_mat = 100. * position_mat
    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], size[1], size[2],-1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    return embedding
