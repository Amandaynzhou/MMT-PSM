from torch import nn
import torch
import  numpy as np
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.relation.relation_mask_feature_extractor import make_relation_mask_feature_extractor,SameSizeRoiAlignMaskFeatureExtractor
from maskrcnn_benchmark.modeling.relation.relation_module import RelationModule,\
    cat_boxlist,extract_rank_embedding
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
import torch.nn.functional as F
import math
from maskrcnn_benchmark.utils.cuda_kmeans import lloyd
from maskrcnn_benchmark.layers import Conv2d,ConvTranspose2d
import pdb
from maskrcnn_benchmark.modeling.roi_heads.mask_head.loss import vgg16
import cv2
class MaskRelationRefineNet(nn.Module):
    def __init__(self, cfg, predictor):
        super(MaskRelationRefineNet, self).__init__()
        self.cfg = cfg.clone()
        hide_dim = (784,) if cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL == 1 else (int(14 * 14 * 16),)
        self.output_channel = 784 if cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL ==1 else int(14 * 14 * 16)
        self.relation_hw = 14 if cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL !=1 else 28
        self.appearance_feature_extractor = make_relation_mask_feature_extractor(cfg)
        self.prepare_sort_by_cluster = False
        self.num_center_per_class = 1
        if self.cfg.MODEL.RELATION_MASK.IOU_COOR ==  True and self.geo_feature_dim == 4:
            self.geo_feature_dim = 5
        if cfg.MODEL.RELATION_MASK.IOU_COOR and self.geo_feature_dim > 5:
            self.geo_feature_dim = int(self.geo_feature_dim/4 * 5)
        self.boxcoder = BoxCoder(weights=(10., 10., 5., 5.))
        # self.class_agnostic = cfg.MODEL.RELATION_NMS.CLASS_AGNOSTIC
        self.fg_class = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES - 1
        # in_channel = int(16 * 14 * 14) if cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL!=1 else ()
        self.classifier = nn.Conv2d(cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL, 3, 1)
        # self.classifier = nn.Linear(128, int(cfg.MODEL.ROI_MASK_HEAD.RESOLUTION * cfg.MODEL.ROI_MASK_HEAD.RESOLUTION), bias=True)

        if cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL != 1:
            self.deconv_1 = ConvTranspose2d( 16, 16, 2, 2, 0)

        # self.detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
        self.iter = 0
        if self.cfg.MODEL.RELATION_MASK.TYPE == 'CAM':
            self.relation_module = CAM_Module(128)
        elif self.cfg.MODEL.RELATION_MASK.TYPE == 'CIAM':
            self.relation_module = CIAM_Module(cfg)
        if self.cfg.MODEL.RELATION_MASK.SAME_PREDICTOR:
            self.predictor = predictor
        else:
            self.deconv_1 = ConvTranspose2d(cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL,
                                            cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL, 2, 2, 0)
            self.classifier = nn.Conv2d(cfg.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL, 3, 1, 1, 0)

    def forward(self, x):
        '''
        Only support batch = 1
        :param x:
        :return:
        '''

        features_RoI, features_mask, proposals, targets  = x
        # binarized
        select_feature_mask, \
        sorted_feature_roi, \
        sorted_feature_mask, \
        distances,\
        sorted_proposals, \
        bboxes, \
        label_index,\
        center,\
        distance_before_refine = self.prepare_msk_relation(features_RoI, features_mask, proposals,)
        if self.cfg.MODEL.RELATION_MASK.TYPE == 'CAM':
            for i in range(self.fg_class):
                feature = select_feature_mask[i]
                select_feature_mask[i] = self.relation_module(feature[None,:,:,:])[0,:,:,:]
            select_feature_mask = torch.cat(select_feature_mask)
            index = torch.arange(select_feature_mask.shape[0], device=select_feature_mask.device)
            sorted_feature_mask[index, label_index] = select_feature_mask[:,:,:]
            return sorted_feature_mask, sorted_proposals, targets, None
            # distance_before_refine = torch.cat(distance_before_refine)
        if self.cfg.MODEL.RELATION_MASK.TYPE == 'CIAM':
            cls_length = [select_feature_mask[i].shape[0] for i in range(self.fg_class)]
            select_feature_mask = torch.cat(select_feature_mask, dim = 0)
            sorted_feature_roi = torch.cat(sorted_feature_roi, dim = 0)
            select_feature_mask = F.sigmoid(select_feature_mask)
            sorted_feature = self.appearance_feature_extractor(( sorted_feature_roi,select_feature_mask[:,None,:,:]))
            if self.cfg.MODEL.RELATION_MASK.FEATURE_EXTRACTOR == 'SameFeatureMask':
                sorted_feature, select_feature_mask = sorted_feature
                sorted_feature = torch.split(sorted_feature, cls_length)
            else:
                sorted_feature = torch.split(sorted_feature, cls_length)
            relation_feature = []
            for i in range(self.fg_class):
                feature = sorted_feature[i]
                if feature.shape[0]!=0:
                    relation_feature.append(self.relation_module(feature))
            relation_feature = torch.cat(relation_feature)
            if self.cfg.MODEL.RELATION_MASK.SAME_PREDICTOR \
                    and self.cfg.MODEL.RELATION_MASK.FEATURE_EXTRACTOR in\
                    [ "SameSizeRoiAlignMaskFeatureExtractor", "SameFeatureMask", 'DeepFeatureExtractor']:
                relation_feature = self.predictor(relation_feature)
            else:
                relation_feature = self.deconv_1(relation_feature)
                relation_feature = F.relu(relation_feature)
                relation_feature = self.classifier(relation_feature)
            index = torch.arange(relation_feature.shape[0], device=select_feature_mask.device)
            sorted_feature_mask[index] = relation_feature
            return sorted_feature_mask, sorted_proposals, targets, None

    def prepare_msk_relation(self, featureROI, feature_mask, proposal):

        gt_labels = proposal.get_field('labels')
        obj_score = proposal.get_field('objectness')
        label_index = [torch.nonzero(gt_labels == (i+1) )[:,0] for i in range(self.fg_class) ]
        objectness = []
        sorted_feature_roi = []
        select_feature_mask = []
        sorted_feature_mask = []
        bboxes = []
        boxlist = []
        for labels, index in enumerate(label_index):
            # get the class feature and box
            cls_feature_mask = feature_mask[index, labels + 1]
            all_cls_feature_mask = feature_mask[index]
            cls_feature_roi = featureROI[index]
            cls_bbox = proposal.bbox[index]
            cls_obj_score = obj_score[index]
            # sort box and feature
            sort_object_score, sort_index = torch.sort(cls_obj_score, descending=True)
            index = index[sort_index]
            cls_feature_mask = cls_feature_mask[sort_index]
            all_cls_feature_mask = all_cls_feature_mask[sort_index]
            cls_feature_roi = cls_feature_roi[sort_index]
            cls_bbox = cls_bbox[sort_index]
            select_feature_mask.append(cls_feature_mask)
            sorted_feature_mask.append(all_cls_feature_mask)
            sorted_feature_roi.append(cls_feature_roi)
            bboxes.append(cls_bbox)
            objectness.append(sort_object_score)
            cls_boxlist = []
            for i in index.tolist():
                cls_boxlist.append(proposal[i:i + 1])
            if len(index) != 0:
                # sometimes it only have one class box
                boxlist.append(cat_boxlist(cls_boxlist))
        boxlist = [cat_boxlist(boxlist)]
        sorted_feature_mask = torch.cat(sorted_feature_mask, 0)
        return select_feature_mask,\
               sorted_feature_roi, \
               sorted_feature_mask,\
               None,\
               boxlist,\
               bboxes,\
               gt_labels, \
               None, \
               None

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # pdb.set_trace()
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CIAM_Module(nn.Module):
    def __init__(self, cfg,):
        super(CIAM_Module, self).__init__()
        self.norm = cfg.MODEL.RELATION_MASK.NORM
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.topk = cfg.MODEL.RELATION_MASK.TOPK
        self.prenorm = cfg.MODEL.RELATION_MASK.PRE_NORM
    def forward(self, x):
        '''

        :param x: (n,c,h,w) n instance feature map
        :return:
        '''
        # pre-norm



        n_instance, C, H, W = x.size()
        if n_instance>self.topk:
            topk = self.topk
        else:
            topk = n_instance
        if self.prenorm:
            normfeature = torch.norm(x.view(n_instance, -1), 2,-1)
            channel_wize_feature = x/normfeature[:,None,None,None]
            # channel_wize_feature = normalize_tensor(x)
        else:
            channel_wize_feature = x
        channel_wize_feature = channel_wize_feature.permute(1,0,2,3)
        proj_query = channel_wize_feature.contiguous().view(C, n_instance, -1)
        proj_key = channel_wize_feature.contiguous().view(C,n_instance,-1).permute(0,2,1)
        # [c, n, n ]
        energy = torch.bmm(proj_query, proj_key)
        # normalize
        if self.norm == 1:
            channel_weight = torch.abs(torch.sum(energy.view(C,-1), dim = 1))
            channel_weight = channel_weight/torch.max(channel_weight)
            norm_energy = energy * channel_weight[:, None, None].expand_as(energy)
        elif self.norm == 2:
            norm_energy = normalize_tensor(energy[None,:,:,:])
            norm_energy = norm_energy[0,:,:,:]
        else:
            norm_energy = energy
        norm_energy = torch.max(norm_energy, -1, keepdim=True)[0].expand_as(norm_energy) - norm_energy
        norm_energy = torch.mean(norm_energy, 0)
        attention = self.softmax(norm_energy)
        proj_value = x.view(1, n_instance, -1)
        out = torch.bmm(attention[None,:,:], proj_value)
        out = out.view(n_instance, C, H, W)
        out = self.gamma * out + x
        return out

def normalize_tensor(in_feat,eps=1e-10):
    # norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3]).repeat(1,in_feat.size()[1],1,1)
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3])
    return in_feat/(norm_factor.expand_as(in_feat)+eps)
