import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import functional as F
import torch_scatter as ts
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, boxlist_iou
from maskrcnn_benchmark.modeling.losses import balanced_BCE,\
    classification_loss, weighted_BCE,combined_BCE,smooth_l1
from maskrcnn_benchmark.modeling.python_nms import cyto_nms
from sklearn.cluster import MeanShift
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
        # pdb.set_trace()

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
        # todo: check
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

        # scaled_dot = torch.sum((w_k*w_q),-1 )
        # scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_g = w_g.view(-1, N, N, self.fc_dim[1])
        #  [num_fg_classes, fc_dim[1], num_rois, num_rois]
        w_g = w_g.permute(0,3,1,2)
        #  [num_fg_classes * fc_dim[1], num_rois, num_rois]
        w_g = w_g.contiguous().view(-1 ,N ,N)
        # w_a = scaled_dot.view(N,N)
        # pdb.set_trace()
        # log_iou = torch.log(torch.clamp(iou, min =  1e-6)) if iou is not None else 0
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
        # aff_softmax, [num_fg_classes * fc_dim[1], num_rois, num_rois]
        # only care about the relation with topk to remove noise information
        # pdb.set_trace()
        # try:
        top_k = min(N, self.topk)
        w_mn_topk, indices = torch.topk(w_mn, top_k, dim=2,
                                        largest=True, sorted=True)
        # except:
        # pdb.set_trace()
        res = torch.zeros_like(w_mn).to(w_mn.device)

        w_mn_topk = F.softmax(w_mn_topk, dim = 2)
        w_mn_topk = res.scatter(2, indices, w_mn_topk)
        w_mn_topk = w_mn_topk.view(num_fg_class, -1, N)
        # f_a: [num_fg_classes , num_rois, num_feat]
        # [num_fg_cls,fc_dim[1] * num_rois, num_feat ]
        output = torch.bmm(w_mn_topk, f_a)
        # [num_fg_cls,fc_dim[1] , num_rois, num_feat ]
        output = output.view(num_fg_class, self.fc_dim[1], N, feat_dim)
        # [fc_dim[1], num_feat, num_rois, cls]
        output = output.permute(1, 3, 2, 0)
        # output_t_reshape, [1, fc_dim[1] * feat_dim, num_rois, num_fg_classes]
        output = output.contiguous().view(1, self.fc_dim[1] * feat_dim, N, -1)
        # [1, 128, nroi, cls]
        output = self.conv1(output)
        output = output.squeeze()
        output = output.permute(1,2,0)

        return output
#



class ClsWiseRelationModule(RelationModule):
    '''
    Class-specific relation: e.g. the relation matrix w_g, w_k, w_q is different for
    cls1 to cls2, cls2 to cls1, cls1 to cls1, cls2 to cls2
    '''
    def __init__(self, appearance_feature_dim=1024, geo_feature_dim = 64 ,
                 fc_dim=(64, 16), group=16, dim=(1024, 1024, 1024), nb_cls = 2, topk = 90):
        super(ClsWiseRelationModule, self).__init__(appearance_feature_dim, geo_feature_dim, fc_dim, group, dim)

        self.WG =nn.ModuleList( [nn.Linear(geo_feature_dim, fc_dim[1], bias=True) for _ in range(int(nb_cls * nb_cls))])
        self.WK =nn.ModuleList( [nn.Linear(appearance_feature_dim, dim[1], bias=True) for _ in range(int(nb_cls * nb_cls))])
        self.WQ =nn.ModuleList( [nn.Linear(appearance_feature_dim, dim[0], bias=True) for _ in range(int(nb_cls * nb_cls))])
        self.relu = nn.ModuleList( [nn.ReLU(inplace=True) for _ in range(int(nb_cls * nb_cls))])
        self.topk = topk
    def forward(self, f_a, position_embedding):
        # f_a: [num_rois, num_fg_classes, feat_dim]
        # pdb.set_trace()

        N, num_fg_class, feat_dim = f_a.size()
        # f_a = f_a.transpose(0, 1)
        # [cls, nroi, feat_dim ]
        f_a = f_a.permute(1, 0, 2)
        f_a_cls = torch.split(f_a, 1 , dim = 0)
        f_a_cls = [f.squeeze() for f in f_a_cls]
        # pdb.set_trace()
        w_mn_11 = self._clswise_relation_op(f_a_cls[0], f_a_cls[0], position_embedding[:, :N, :N, :], 0)
        w_mn_12 =self._clswise_relation_op(f_a_cls[0], f_a_cls[1], position_embedding[:, :N , N:, :], 1)
        w_mn_21 = self._clswise_relation_op(f_a_cls[1], f_a_cls[0], position_embedding[:,N:, :N, :], 2)
        w_mn_22 =self._clswise_relation_op(f_a_cls[1], f_a_cls[1], position_embedding[:, N: , N:, :], 3)

        w_mn_sub1 = torch.cat([w_mn_11,w_mn_12], dim = 2)
        w_mn_sub2 = torch.cat([w_mn_21, w_mn_22], dim = 2)
        #[fc_dim[1], cls*nroi, cls*nroi]
        w_mn = torch.cat([w_mn_sub1, w_mn_sub2], dim =1)
        # normalized
        # w_mn = w_mn/2
        # wmn
        w_mn[:, N:,:N] = 0
        w_mn[:, :N,N:] = 0

        #[1, fc_dim[1]*cls*nroi, cls*nroi]
        w_mn = w_mn.view(1, -1, N * num_fg_class)
        f_a_reshape = f_a.contiguous().view(1 ,  N * num_fg_class, feat_dim)
        # [1,fc_dim[1]*cls*nroi, feat_dim ]
        output = torch.bmm(w_mn, f_a_reshape)
        output = output.view(self.fc_dim[1], num_fg_class, N, feat_dim)
        output = output.permute(0, 3 , 2 , 1)
        # output_t_reshape, [1, fc_dim[1] * feat_dim, num_rois, num_fg_classes]
        output = output.contiguous().view(1, self.fc_dim[1] * feat_dim, N, -1)
        # [1, 128, nroi, cls]
        output = self.conv1(output)
        output = output.squeeze()
        output = output.permute(1, 2, 0)
        return output

    def _clswise_relation_op(self, f_k, f_q, position_embedding, idx):
        '''
        :param f_k: key featrue [ 1, n_roi, feat_dim]
        :param f_q: query feature [1, n_roi, feat_dim]
        :param position_embedding:
        :param idx:
        :return:
        '''
        # pdb.set_trace()
        device = position_embedding.device
        N, feat_dim = f_k.size()
        # f_a_reshape = f_a [num-roi*num-fg-cls, feat-dim]
        # [ num_rois * num_rois, fc_dim[0]]
        position_embedding = position_embedding.contiguous().view(-1, self.fc_dim[0])
        w_g = self.relu[idx](self.WG[idx](position_embedding))
        w_k = self.WK[idx](f_k)
        # [ num_rpi, 16, 64 ]
        w_k = w_k.view(-1, N, self.group, self.dim_group[1])
        # todo: check
        w_k = w_k.permute(0, 2, 3, 1)
        # k_data_batch, [ group, dim_group[1], num_rois,]
        w_k = w_k.contiguous().view(-1, self.dim_group[1], N)
        w_q = self.WQ[idx](f_q)
        w_q = w_q.view(-1, N, self.group, self.dim_group[0])
        w_q = w_q.transpose(1, 2)
        # q_data_batch, [num_fg_classes * group, num_rois, dim_group[0]]
        w_q = w_q.contiguous().view(-1, N, self.dim_group[0])
        w_mn_appearance = (1.0 / math.sqrt(float(self.dim_group[1]))) * torch.bmm(w_q, w_k)
        w_g = w_g.view(-1, N, N, self.fc_dim[1])
        #  [1, fc_dim[1], num_rois, num_rois]
        w_g = w_g.permute(0,3,1,2)
        #  [1 * fc_dim[1], num_rois, num_rois]
        w_g = w_g.contiguous().view(-1 ,N ,N)
        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_mn_appearance

        w_mn_topk, indices = torch.topk(w_mn, self.topk, dim=2, largest=True, sorted=True)
        res = torch.zeros_like(w_mn).to(device)
        w_mn_topk = F.softmax(w_mn_topk, dim = 2)
        w_mn_topk = res.scatter(2, indices, w_mn_topk)
        # w_mn = F.softmax(w_mn, dim=2)
        return w_mn_topk

class AppearanceIntraRelationModule(ClsWiseRelationModule):
    def __init__(self, appearance_feature_dim=1024, geo_feature_dim=64,
                 fc_dim=(64, 16), group=16, dim=(1024, 1024, 1024), topk = 90):
        super(AppearanceIntraRelationModule, self).__init__(appearance_feature_dim, geo_feature_dim, fc_dim, group, dim)

        self.WG = nn.ModuleList([nn.Linear(geo_feature_dim, fc_dim[1], bias=True) for _ in range(2)])
        self.WK = nn.ModuleList([nn.Linear(appearance_feature_dim, dim[1], bias=True) for _ in range(2)])
        self.WQ = nn.ModuleList([nn.Linear(appearance_feature_dim, dim[0], bias=True) for _ in range(2)])
        self.relu = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(2)])
        self.topk = topk
    def forward(self, f_a, position_embedding):
        # pdb.set_trace()
        N, num_fg_class, feat_dim = f_a.size()
        # f_a = f_a.transpose(0, 1)
        f_a = f_a.permute(1, 0, 2)
        # f_a_reshape [num-roi*num-fg-cls, feat-dim]
        f_a_cls = torch.split(f_a, 1, dim=0)
        f_a_cls = [f.squeeze() for f in f_a_cls]
        w_mn_12 = self._clswise_relation_op(f_a_cls[0], f_a_cls[1], position_embedding[:, :N, N:, :], 0)
        w_mn_21 = self._clswise_relation_op(f_a_cls[1], f_a_cls[0], position_embedding[:, N:, :N, :], 1)
        # pdb.set_trace()
        # w_mn = torch.cat([w_mn_12, w_mn_21], dim=0)
        # w_mn = w_mn.view(1, int(N *2), -1)
        f_a = f_a.contiguous().view(1,int(N *2), feat_dim )
        # pdb.set_trace()
        output_1 = torch.bmm(w_mn_12.view(1, -1, N), f_a[:,N:,:])
        output_2 = torch.bmm(w_mn_21.view(1, -1, N), f_a[:,:N,:])
        output = torch.cat([output_1, output_2])
        # pdb.set_trace()
        # [num_fg_cls,fc_dim[1] , num_rois, num_feat ]
        output = output.view(2, self.fc_dim[1], N, feat_dim)
        # [fc_dim[1], num_feat, num_rois, cls]
        output = output.permute(1, 3, 2, 0)
        # output_t_reshape, [1, fc_dim[1] * feat_dim, num_rois, num_fg_classes]
        output = output.contiguous().view(1, self.fc_dim[1] * feat_dim, N, -1)
        # [1, 128, nroi, cls]
        output = self.conv1(output)
        output = output.squeeze()
        output = output.permute(1, 2, 0)
        return output


class IntraRelationModule(ClsWiseRelationModule):
    def __init__(self, appearance_feature_dim=1024, geo_feature_dim=64,
                 fc_dim=(64, 16), group=16, dim=(1024, 1024, 1024), topk = 90):
        super(IntraRelationModule, self).__init__(appearance_feature_dim, geo_feature_dim, fc_dim, group, dim)

        self.WG = nn.ModuleList([nn.Linear(geo_feature_dim, fc_dim[1], bias=True) for _ in range(2)])
        self.WK = nn.ModuleList([nn.Linear(appearance_feature_dim, dim[1], bias=True) for _ in range(2)])
        self.WQ = nn.ModuleList([nn.Linear(appearance_feature_dim, dim[0], bias=True) for _ in range(2)])
        self.relu = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(2)])
        self.topk = topk
    def forward(self, f_a, position_embedding):
        N, feat_dim = f_a.size()
        # f_a = f_a.transpose(0, 1)
        # [cls, nroi, feat_dim ]
        # f_a = f_a.permute(1, 0, 2)
        # pdb.set_trace()
        f_a = torch.cat([f_a[None,:,:], f_a[None, :,:]])
        f_a_cls = torch.split(f_a, 1, dim=0)
        f_a_cls = [f.squeeze() for f in f_a_cls]
        # pdb.set_trace()





        w_mn_12 = self._clswise_relation_op(f_a_cls[0], f_a_cls[1], position_embedding[:, :N, N:, :], 0)
        w_mn_21 = self._clswise_relation_op(f_a_cls[1], f_a_cls[0], position_embedding[:, N:, :N, :], 1)
        # pdb.set_trace()
        # w_mn = torch.cat([w_mn_12, w_mn_21], dim=0)
        # w_mn = w_mn.view(1, int(N *2), -1)
        f_a = f_a.view(1,int(N *2), feat_dim )
        # pdb.set_trace()
        output_1 = torch.bmm(w_mn_12.view(1, -1, N), f_a[:,N:,:])
        output_2 = torch.bmm(w_mn_21.view(1, -1, N), f_a[:,:N,:])
        output = torch.cat([output_1, output_2])
        # pdb.set_trace()
        # [num_fg_cls,fc_dim[1] , num_rois, num_feat ]
        output = output.view(2, self.fc_dim[1], N, feat_dim)
        # [fc_dim[1], num_feat, num_rois, cls]
        output = output.permute(1, 3, 2, 0)
        # output_t_reshape, [1, fc_dim[1] * feat_dim, num_rois, num_fg_classes]
        output = output.contiguous().view(1, self.fc_dim[1] * feat_dim, N, -1)
        # [1, 128, nroi, cls]
        output = self.conv1(output)
        output = output.squeeze()
        output = output.permute(1, 2, 0)
        return output


class DuplicationRemovalNetwork(nn.Module):
    def __init__(self, cfg, is_teacher=False, ):
        super(DuplicationRemovalNetwork, self).__init__()
        self.cfg = cfg.clone()
        self.reg_iou = self.cfg.MODEL.RELATION_NMS.REG_IOU
        self.reg_iou_mask = cfg.MODEL.RELATION_NMS.REG_IOU_MSK
        self.discriminative = cfg.MODEL.RELATION_NMS.D_LOSS
        self.is_teacher= is_teacher
        # pdb.set_trace()
        self.first_n = cfg.MODEL.RELATION_NMS.FIRST_N
        self.NMS_thread = cfg.MODEL.RELATION_NMS.THREAD
        self.nms_rank_fc = nn.Linear(cfg.MODEL.RELATION_NMS.ROI_FEAT_DIM, cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM, bias=True)
        self.roi_feat_embedding_fc = nn.Linear(cfg.MODEL.RELATION_NMS.ROI_FEAT_DIM, cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM, bias=True)
        self.target_thresh = cfg.MODEL.RELATION_NMS.THREAD
        self.geo_feature_dim = cfg.MODEL.RELATION_NMS.GEO_FEAT_DIM

        if cfg.MODEL.RELATION_NMS.USE_IOU:
            self.geo_feature_dim = int(self.geo_feature_dim/4 * 5)
        if not cfg.MODEL.RELATION_NMS.CLS_WISE_RELATION :
            self.relation_module = RelationModule(cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM,
                                                  geo_feature_dim=self.geo_feature_dim,
                                                  fc_dim= (self.geo_feature_dim, 16),
                                                  group=cfg.MODEL.RELATION_NMS.GROUP,
                                                  dim=cfg.MODEL.RELATION_NMS.HID_DIM,
                                                  topk = cfg.MODEL.RELATION_NMS.TOPK,
                                                  iou_method= cfg.MODEL.RELATION_NMS.IOU_METHOD)
        else:
            self.relation_module = ClsWiseRelationModule(cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM,
                                                  geo_feature_dim=cfg.MODEL.RELATION_NMS.GEO_FEAT_DIM,
                                                  fc_dim= (self.geo_feature_dim, 16),
                                                  group=cfg.MODEL.RELATION_NMS.GROUP,
                                                  dim=cfg.MODEL.RELATION_NMS.HID_DIM)

        if cfg.MODEL.RELATION_NMS.MUTRELATION:
            if cfg.MODEL.RELATION_NMS.APPEARANCE_INTER:
                self.inter_relation = AppearanceIntraRelationModule(cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM,
                                                      geo_feature_dim=cfg.MODEL.RELATION_NMS.GEO_FEAT_DIM,
                                                      fc_dim=cfg.MODEL.RELATION_NMS.FC_DIM,
                                                      group=cfg.MODEL.RELATION_NMS.GROUP,
                                                      dim=cfg.MODEL.RELATION_NMS.HID_DIM,
                                                      topk= cfg.MODEL.RELATION_NMS.TOPK )
            else:
                self.inter_relation = IntraRelationModule(cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM,
                                                      geo_feature_dim=cfg.MODEL.RELATION_NMS.GEO_FEAT_DIM,
                                                      fc_dim=cfg.MODEL.RELATION_NMS.FC_DIM,
                                                      group=cfg.MODEL.RELATION_NMS.GROUP,
                                                      dim=cfg.MODEL.RELATION_NMS.HID_DIM,
                                                      topk= cfg.MODEL.RELATION_NMS.TOPK )
        if self.discriminative:
            self.relation_module_d =  RelationModule(cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM,
                                                  geo_feature_dim=self.geo_feature_dim,
                                                  fc_dim= (self.geo_feature_dim, 16),
                                                  group=cfg.MODEL.RELATION_NMS.GROUP,
                                                  dim=cfg.MODEL.RELATION_NMS.HID_DIM,
                                                  topk = cfg.MODEL.RELATION_NMS.TOPK,
                                                  iou_method= cfg.MODEL.RELATION_NMS.IOU_METHOD)
            self.classifier_d = nn.Linear(128, 8, bias=True)
            self.d_loss = Discriminative_Loss()


            # self.relu2 = nn.ReLU(inplace=True)

            # self.inter_classifier = nn.Linear(128, len(self.target_thresh), bias= True)
            # self.ir_conv = nn.Linear(128, len(self.target_thresh), bias= True)
        # self.nms_logit_fc = nn.Linear(cfg.MODEL.RELATION_NMS.ROI_FEAT_DIM,1,bias=True)
        self.nms_fg_weight = torch.tensor([1., cfg.MODEL.RELATION_NMS.WEIGHT])
        self.mt_fg_weight = torch.tensor([1.,10.])
        self.alpha = cfg.MODEL.RELATION_NMS.ALPHA
        self.gamma = cfg.MODEL.RELATION_NMS.GAMMA
        self.boxcoder = BoxCoder(weights=(10., 10., 5., 5.))
        self.class_agnostic = cfg.MODEL.RELATION_NMS.CLASS_AGNOSTIC
        self.fg_class = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES -1
        if cfg.MODEL.RELATION_NMS.MUTRELATION and cfg.MODEL.RELATION_NMS.CONCAT:
            self.classifier = nn.Linear(int(128 * 2), len(self.target_thresh), bias= True)
        else:
            self.classifier =  nn.Linear(128, len(self.target_thresh), bias= True)
        self.relu1 = nn.ReLU(inplace=True)
        # self.BCEloss = nn.BCELoss()
        # self.class_weight = torch.FloatTensor([1, 3]).cuda()
        self.fg_thread = cfg.MODEL.RELATION_NMS.FG_THREAD
        self.detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

        if cfg.MODEL.RELATION_NMS.POS_NMS != -1:
            self.nms = cfg.MODEL.RELATION_NMS.POS_NMS
        else:
            self.nms = None
        # self.relu2 = nn.ReLU(inplace=True)
        # self.refined_matcher = make_refined_matcher(cfg)
        self.student_loss = cfg.MT.NMS_LOSS_TYPE
        self.mode = None
    def set_teacher_mode(self, mode):
        self.mode = mode

    def forward(self, x, is_student = False,idx_t =None,stratergy2=False):

        # import pdb;pdb.set_trace()
        if  is_student and stratergy2:
            # pdb.set_trace()
            appearance_feature, proposals, cls_score, box_reg,\
            nms_idx, nms_result_t, nms_score, = x
            appearance_feature = appearance_feature
            cls_score = cls_score
            box_reg = box_reg
            device = appearance_feature.device
            self.device = device
            with torch.no_grad():
                sorted_boxlists = self.prepare_ranking_student(cls_score,
                                                       box_reg,
                                                       proposals,
                                                       nms_idx,
                                                       nms_result_t,
                                                       nms_score,
                                                       is_student,)


        else:
            appearance_feature, proposals, cls_score, box_reg, targets = x
            device = appearance_feature.device
            appearance_feature = appearance_feature
            cls_score = cls_score
            box_reg = box_reg
            self.device =device
            # pdb.set_trace()
            with torch.no_grad():
                sorted_boxlists = self.prepare_ranking(cls_score,
                                                       box_reg,
                                                       proposals,
                                                       targets,
                                                       is_student,
                                                       reg_iou=self.reg_iou)
        # concate value from different images
        # pdb.set_trace()
        boxes_per_image = [len(f) for f in proposals]
        # pdb.set_trace()
        idxs =  [f.get_field('sorted_idx') for f in sorted_boxlists]
        scores = torch.cat([f.get_field('scores') for f in sorted_boxlists])
        bboxes = torch.cat([f.bbox.reshape(-1,self.fg_class,4) for f in sorted_boxlists])
        objectness = torch.cat([f.get_field('objectness').reshape(-1, self.fg_class) for f in sorted_boxlists])
        all_scores = torch.cat([f.get_field('all_scores') for f in
             sorted_boxlists])
        # sort_idx = torch.cat([f.get_field('sorted_idx') for f in
        #      sorted_boxlists])
        # add iou information
        image_sizes = [f.size for f in sorted_boxlists]
        # pdb.set_trace()
        sorted_boxes_per_image = [[*f.shape][0] for f in idxs]
        appearance_feature = self.roi_feat_embedding_fc(appearance_feature)
        appearance_feature = appearance_feature.split(boxes_per_image, dim=0)
        sorted_features  = []
        nms_rank_embedding = []
        # pdb.set_trace()
        for id, feature, box_per_image in zip(idxs, appearance_feature, boxes_per_image):
            feature = feature[id]
            # pdb.set_trace()
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
        # pdb.set_trace()
        if not self.cfg.MODEL.RELATION_NMS.MUTRELATION:

            sorted_features = sorted_features + nms_rank_embedding[:,None,:]

            boxes_cls_1 = BoxList(bboxes[ :, 0, :], image_sizes[0])
            boxes_cls_2 = BoxList(bboxes[ :, 1, :], image_sizes[0])
            iou_1 = boxlist_iou(boxes_cls_1, boxes_cls_1)
            iou_2 = boxlist_iou(boxes_cls_2, boxes_cls_2)
            if self.cfg.MODEL.RELATION_NMS.USE_IOU:
                iou = [iou_1, iou_2]
            else:
                iou = None
            # pdb.set_trace()
            nms_position_matrix = extract_multi_position_matrix(bboxes, None, self.geo_feature_dim, 1000,
                                                                clswise = self.cfg.MODEL.RELATION_NMS.CLS_WISE_RELATION,)
            nms_attention_1 = self.relation_module(sorted_features,  nms_position_matrix, iou)
            if self.discriminative:
                nms_attention_2 = self.relation_module_d(sorted_features,  nms_position_matrix, iou)
                d_features = sorted_features + nms_attention_2
                # d_features = nms_attention_2
                d_features = d_features.view(-1, self.cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM)
                if self.cfg.MODEL.RELATION_NMS.DO > 0:
                    d_features = F.dropout(d_features, self.cfg.MODEL.RELATION_NMS.DO, self.training)
                d_features = self.classifier_d(d_features)
                # 90，2，16
                d_features = d_features.view(-1, self.fg_class, 8)
                # pdb.set_trace()
                # sorted_features = sorted_features.view(-1, self.fg_class, len(self.target_thresh))
            # if self.reg_iou:
            #     nms_attention_2 = self.relation_module_reg_iou(sorted_features,  nms_position_matrix, iou)
            #     sorted_features_reg_iou = sorted_features + nms_attention_2
            #     sorted_features_reg_iou = F.relu(sorted_features_reg_iou)
            #     sorted_features_reg_iou = sorted_features_reg_iou.view(-1, self.cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM)
            #     sorted_features_reg_iou = self.classifier_2(sorted_features_reg_iou)
            #     sorted_features_reg_iou = sorted_features_reg_iou.view(-1, self.fg_class, len(self.target_thresh))

            sorted_features = sorted_features + nms_attention_1
            sorted_features = self.relu1(sorted_features)
            # [first_n * num_fg_classes, 128]
            sorted_features = sorted_features.view(-1, self.cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM)
            if self.cfg.MODEL.RELATION_NMS.DO >0:
                sorted_features  = F.dropout(sorted_features, self.cfg.MODEL.RELATION_NMS.DO,self.training)
            sorted_features = self.classifier(sorted_features)
            # logit_reshape, [first_n, num_fg_classes, num_thread]
            sorted_features = sorted_features.view(-1, self.fg_class, len(self.target_thresh))
            if not self.reg_iou:
                sorted_features = torch.sigmoid(sorted_features)
            # labels = labels.view(-1,1)
            # labels = torch.clamp(labels, max=1.0)
            # class_weight = self.class_weight[labels.view(-1)].view(-1,1)
            # pdb.set_trace()
            scores = torch.cat([scores[:,:,None]]*len(self.target_thresh), dim = -1)
            # [first_n, num_fg_classes, num_thread]
            # scores = sorted_features * scores
            # scores  = sorted_features
            # pdb.set_trace()
        else:
            nms_rank_embedding = nms_rank_embedding[:, None, :]
            sorted_features = sorted_features + nms_rank_embedding

            nms_position_matrix = extract_multi_position_matrix(bboxes, self.cfg.MODEL.RELATION_NMS.GEO_FEAT_DIM, 1000,
                                                                clswise=True)
            N = nms_position_matrix.shape[1]
            # pdb.set_trace()
            intra_position = torch.cat([nms_position_matrix[:, :int(N/2), :int(N/2)],
                                        nms_position_matrix[:, int(N/2): ,  int(N/2): ]])
            # inter_position = torch.cat([nms_position_matrix[:, :N/2, N/2:], nms_position_matrix[:, N/2: ,  :N/2 ]])
            nms_attention_1 = self.relation_module(sorted_features, intra_position)
            if self.cfg.MODEL.RELATION_NMS.APPEARANCE_INTER:
                inter_scores = self.inter_relation(sorted_features, nms_position_matrix)
            else:
                inter_scores = self.inter_relation(nms_rank_embedding.squeeze(), nms_position_matrix)
            # inter_scores = self.relu2(inter_scores)
            # pdb.set_trace()
            sorted_features = sorted_features + nms_attention_1
            if not self.cfg.MODEL.RELATION_NMS.CONCAT:
                sorted_features = sorted_features + inter_scores
                sorted_features = sorted_features.view(-1, self.cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM)
            else:
                # pdb.set_trace()
                sorted_features = torch.cat([sorted_features, inter_scores], dim = 2)
                sorted_features = sorted_features.view(-1, int(2 *self.cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM))
            sorted_features = self.relu1(sorted_features)
            # [first_n * num_fg_classes, 128]

            # inter_scores = inter_scores.view(-1, self.cfg.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM)
            sorted_features = self.classifier(sorted_features)
            # inter_scores = self.inter_classifier(inter_scores)
            # pdb.set_trace()
            # logit_reshape, [first_n, num_fg_classes, num_thread]
            sorted_features = sorted_features.view(-1, self.fg_class, len(self.target_thresh))
            # inter_scores = inter_scores.view(-1, self.fg_class, len(self.target_thresh))
            sorted_features = torch.sigmoid(sorted_features)
            # inter_scores = torch.sigmoid(inter_scores)
            

            # labels = labels.view(-1,1)
            # labels = torch.clamp(labels, max=1.0)
            # class_weight = self.class_weight[labels.view(-1)].view(-1,1)
            # pdb.set_trace()
            scores = torch.cat([scores[:, :, None]] * len(self.target_thresh), dim=-1)
            # inter_scores = torch.cat([inter_scores[:,:]] * len(self.target_thresh), dim = -1)
            # [first_n, num_fg_classes, num_thread]
            # pdb.set_trace()
            # todo:
            # scores = sorted_features * scores
        loss_dict = {}
        if self.training and not is_student:


            if self.reg_iou:

                # when use regression donot do sorted_features = scores * sorted_features
                reg_label = torch.cat([f.get_field('labels_iou_reg') for f in sorted_boxlists])
                reg_label = reg_label.to(scores.device)
                reg_label = reg_label.type(torch.cuda.FloatTensor)
                if self.reg_iou_mask:
                    positive_mask = reg_label.gt(0.)
                    reg_label = torch.masked_select(reg_label, positive_mask)
                    sorted_features =torch.masked_select(sorted_features,positive_mask)

                sorted_features = sorted_features.to(scores.device)
                sorted_features = sorted_features.type(torch.cuda.FloatTensor)
                if self.discriminative:
                    # 90 2 1
                    d_label = torch.cat([f.get_field('labels_iou_d') for f in sorted_boxlists])
                    # pdb.set_trace()
                    d_loss = self.d_loss(d_features,d_label)
                    loss_dict['d_loss'] = d_loss
                if reg_label.shape is not None:
                    reg_iou_loss = F.mse_loss(reg_label,sorted_features)
                    # pdb.set_trace()
                else:
                    reg_iou_loss = torch.tensor(0.).to(scores.device)
                loss_dict['nms_loss'] = reg_iou_loss

            # pdb.set_trace()
            else:
                sorted_features = scores * sorted_features
                labels = torch.cat([f.get_field('labels') for f in sorted_boxlists])

                labels = labels.to(scores.device)
                labels = labels.type(torch.cuda.FloatTensor)

                # WEIGHTED NMS
                # pdb.set_trace()
                if self.student_loss == 'bce':
                    nms_loss = self.adaptive_bce(scores,sorted_features,labels)
                else:
                    nms_loss = 0.5 *self.adaptive_bce_with_cls_balance(
                        scores,sorted_features,
                                             labels,  fg = self.nms_fg_weight)
                loss_dict['nms_loss']=nms_loss
                # self.nms_fg_weight = self.nms_fg_weight.to(device)
                # normalize_factor = labels.sum()/labels.numel()*4
                # weight = self.nms_fg_weight[labels.data.view(-1).long(
                #
                # )].view_as(labels)/normalize_factor
                # nms_loss = F.binary_cross_entropy(sorted_features,
                #                                   labels,
                #                                   weight)
                # nms_loss = F.binary_cross_entropy(sorted_features, labels,)

                # labels = labels.split(boxes_per_image, dim  = 0)
                # sorted_boxlists = sorted_boxlists.split(boxes_per_image, dim =0)

            return None, loss_dict
        elif is_student and stratergy2:
            # nms_score = f.sig(sorted feature), nms_result_t =
            # nms_score* score
            # pdb.set_trace()
            teacher_sorted_feauture = torch.cat([f.get_field(
                'nms_score') for f in sorted_boxlists])
            teacher_sorted_result = torch.cat([f.get_field(
                'nms_result_t') for f in sorted_boxlists])
            # teacher_score =torch.cat([f.get_field('nms_result_t')
            #                           for f in sorted_boxlists])
            if self.student_loss == 'bce':
                # pdb.set_trace()
                teacher_sorted_result[teacher_sorted_result>self.fg_thread] = 1
                teacher_sorted_result[teacher_sorted_result<=self.fg_thread]=0
                nms_loss = self.adaptive_bce(scores,
                                             sorted_features,
                                             teacher_sorted_feauture, teacher_sorted_result)
                #
                #
                # gap =  - torch.abs(scores -
                #                    sorted_features).view_as(teacher_sorted_feauture)
                # normalize_factor = teacher_sorted_result.sum() / teacher_sorted_result.numel() * 4
                # cls_balance_weight = self.nms_fg_weight[
                #                          teacher_sorted_result.data.view(-1).long(
                # )].view_as(teacher_sorted_result)/normalize_factor
                # weight = (1 - self.alpha*torch.exp(gap)) * cls_balance_weight
                # nms_loss = F.binary_cross_entropy(scores.view_as(
                #     teacher_sorted_feauture),teacher_sorted_feauture, weight)
            elif self.student_loss == 'wbce':
                teacher_sorted_result[
                    teacher_sorted_result > self.fg_thread] = 1
                teacher_sorted_result[
                    teacher_sorted_result <= self.fg_thread] = 0
                nms_loss = self.adaptive_bce_with_cls_balance(scores,
                                             sorted_features,
                                             teacher_sorted_feauture,
                                             teacher_sorted_result,
                                                              fg =
                                                              self.mt_fg_weight)

            elif self.student_loss == 'mse':
                nms_loss = F.mse_loss(scores[:,:,0],
                                      teacher_sorted_feauture)
            else:
                print('do not support %s'%self.student_loss)

            return None, dict(mt_nms_loss=nms_loss)


        elif self.training and is_student:
            scores = scores * sorted_features
            # pdb.set_trace()
            if self.reg_iou:
                nms_score = torch.cat([f.get_field('mt_nms') for f in  sorted_boxlists])

                nms_loss = F.mse_loss(scores, nms_score)
            else:
                labels = torch.cat([f.get_field('labels') for f in sorted_boxlists])
                # pdb.set_trace()
                nms_score = torch.cat([f.get_field('mt_nms') for f in  sorted_boxlists])
                # convert weight to 1 for cls 0
                if self.student_loss == 'wbce':
                    # pdb.set_trace()
                    nms_loss = combined_BCE(scores, labels, nms_score)
                elif self.student_loss =='bce':
                    nms_loss = F.binary_cross_entropy(scores,labels)
                elif self.student_loss == 'mse':
                    # pdb.set_trace()
                    nms_loss = F.mse_loss(scores, nms_score)
                elif self.student_loss == 'kl':
                    nms_loss = F.kl_div(torch.log(scores),nms_score)
                elif self.student_loss == 'smooth':
                    nms_loss  = smooth_l1(scores,nms_score)
                else:
                    print('error, nms loss do not support %s'%self.student_loss)
            # nms_loss = classification_loss(self.student_loss,
            #                                scores,labels,nms_score)

            # if self.student_loss =='kl':
            #     nms_loss = KLloss(scores,nms_score)
            # elif self.student_loss =='bce':
            #     nms_loss = BCE(scores,nms_score)
            # else:
            #     nms_loss = weighted_BCE(scores,labels,nms_score)
            return None, dict(mt_nms_loss=nms_loss)
        elif self.mode == 'train':
            # todo:check condition
            scores = sorted_features * scores
            return idxs, sorted_features, scores

        else:
            input_scores = scores

            if self.reg_iou:
                # pdb.set_trace()
                scores = sorted_features* (scores>self.fg_thread).float()
                scores = scores.clamp_(0.,1.)
                # pdb.set_trace()

            else:
                scores = sorted_features * scores
            scores = self.merge_multi_thread_score_test(scores)
            scores = scores.split(sorted_boxes_per_image, dim = 0)
            bboxes = bboxes.split(sorted_boxes_per_image, dim = 0)
            input_scores = input_scores.split(sorted_boxes_per_image, dim = 0)
            objectness = objectness.split(sorted_boxes_per_image, dim = 0)
            all_scores = all_scores.split(sorted_boxes_per_image,
                                          dim = 0)
            if self.discriminative:
                input_scores = d_features.split(sorted_boxes_per_image, dim = 0)

            # scores = [f.cpu().numpy() for f in scores]
            # bboxes = [f.cpu().numpy() for f in bboxes]
            result = []
            # pdb.set_trace()
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
                # pdb.set_trace()

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
                                                device))
                result_per_image.append(boxlist_for_class)

                # for cyto:
                if self.cfg.MODEL.ROI_HEADS.NMS_TYPE=='cyto':
                    nuclei_bbox = boxlist_for_class.bbox
                    inds = index
                    cyto_bbox = bbox[inds, 0]
                    cyto_score = score[inds, 0]
                    # calculate ioa
                    # inter/(nuclei_area)
                    # todo: check correctness
                    lt = torch.max(nuclei_bbox[:, None, :2],
                                   cyto_bbox[:, :2])  # [N,
                    # M,2]
                    rb = torch.min(nuclei_bbox[:, None, 2:],
                                   cyto_bbox[:, 2:])  # [N,
                    # M,2]
                    TO_REMOVE = 1
                    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
                    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
                    area = (nuclei_bbox[:, 2] - nuclei_bbox[:,
                                                0] + TO_REMOVE) * (
                                   nuclei_bbox[:, 3] - nuclei_bbox[:,
                                                       1] + TO_REMOVE)
                    iou = inter / area[:, None]
                    iou[iou < 0.5] = 0
                    _iou, _idx = torch.sort(iou, dim=0,
                                            descending=True)
                    nuclei_h = ((nuclei_bbox[:, 0] + nuclei_bbox[:,
                                                     2]) / 2)[:, None]
                    nuclei_w = ((nuclei_bbox[:, 1] + nuclei_bbox[:,
                                                     3]) / 2)[:, None]
                    cyto_h = \
                    ((cyto_bbox[:, 0] + cyto_bbox[:, 2]) / 2)[
                        None, ...]
                    cyto_w = \
                    ((cyto_bbox[:, 1] + cyto_bbox[:, 3]) / 2)[
                        None, ...]
                    # pdb.set_trace()
                    distance = (nuclei_h - cyto_h) ** 2 + (
                                nuclei_w - cyto_w) ** 2

                    distance = distance.transpose(1, 0)
                    _idx = _idx.transpose(1, 0)
                    _iou_t = _iou.transpose(1, 0)
                    _iou_t[_iou_t > 0] = 1
                    _iou_t = _iou_t.long().sum(1)
                    list_of_nuclei_id_list = []
                    for j, number in enumerate(_iou_t):
                        nuclei_id_list = _idx[j, :number]
                        distance_list = distance[j, nuclei_id_list]
                        sort_distance, sort_distance_idx = torch.sort(
                            distance_list, descending=False)
                        nuclei_id_list = nuclei_id_list[
                            sort_distance_idx]
                        nuclei_id_list = nuclei_id_list.tolist()
                        list_of_nuclei_id_list.append(nuclei_id_list)
                    # pdb.set_trace()
                    boxlist_for_cyto = BoxList(cyto_bbox,
                                               image_size,
                                               mode="xyxy")
                    boxlist_for_cyto.add_field("scores", cyto_score)
                    cls_scores_all = prob_boxhead[index, 0]
                    boxlist_for_cyto.add_field('all_scores',
                                                cls_scores_all)
                    boxlist_for_cyto.add_field("objectness",
                                               cyto_score)
                    boxlist_for_cyto.add_field("nuclei_id_list",
                                               list_of_nuclei_id_list)

                    boxlist_for_cyto = cyto_nms(boxlist_for_cyto,
                                                self.nms,
                                                score_field="scores")
                    boxlist_for_cyto = boxlist_nms(
                        boxlist_for_cyto, self.nms,
                        score_field="scores"
                    )
                    num_labels = len(boxlist_for_cyto)
                    boxlist_for_cyto.add_field(
                        "labels", torch.full((num_labels,), 1,
                                             dtype=torch.int64,
                                             device=device)
                    )
                    # pdb.set_trace()
                    # boxlist_for_cyto.remove_field['nuclei_id_list']
                    result_per_image.append(boxlist_for_cyto)

                else:



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
                                                    dtype=torch.int64).to(device))
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
            if not self.is_teacher:
                return result, {}
            else:

                return result, {}

    def adaptive_bce(self, classifier, nms_score, label, label_t =
    None):

        if label_t is None:
            label_t = label
        gap = - torch.abs(classifier - nms_score).view_as(label)

        self.nms_fg_weight =self.nms_fg_weight.to(nms_score.device)
        cls_balance_weight = self.nms_fg_weight[label_t.data.view(
            -1).long()].view_as(label_t)
        normalize_factor = (cls_balance_weight.sum() ) \
                           /label_t.numel()

        cls_balance_weight = cls_balance_weight/normalize_factor

        with torch.no_grad():
            weight = torch.pow((1 - self.alpha * torch.exp(gap)) ,
                               self.gamma
                               )* \
                     cls_balance_weight
        # nms_loss = F.binary_cross_entropy(nms_score.view_as(label),
        #                                   label, weight)
        #
        nms_loss = F.binary_cross_entropy(nms_score.view_as(label),
                                          label, weight)
        return nms_loss

    def adaptive_bce_with_cls_balance(self, classifier, nms_score, label, label_t =
    None, fg = None):
        # pdb.set_trace()
        if label_t is None:
            label_t = label
        gap = - torch.abs(classifier - nms_score).view_as(label)
        normalize_factor = label_t.sum() / label_t.numel() * 4
        # try:
        # pdb.set_trace()
        fg =fg.to(label_t.device)
        # except:
        #     pdb.set_trace()
        cls_balance_weight = fg[label_t.data.view(
            -1).long()].view_as(label_t) / normalize_factor
        with torch.no_grad():
            weight = torch.pow((1 - self.alpha * torch.exp(gap)) ,
                               self.gamma
                               )* \
                     cls_balance_weight
        # class balance:
        cls_norm  =weight.sum()/weight.sum(dim=0)
        # pdb.set_trace()
        weight = weight * cls_norm
        # pdb.set_trace()
        nms_loss = F.binary_cross_entropy(nms_score.view_as(label),
                                          label, weight)

        return nms_loss

    def prepare_bbox_iou(self, boxlist, label_index, clswize = True):
        iou = boxlist_iou(boxlist, boxlist)
        if clswize:
            split_point = label_index[0]
            iou1 = iou[:split_point, : split_point]
            iou2 = iou[split_point:, split_point:]
            iou = [iou1, iou2]
        return iou

    def prepare_discriminative_label(self, sorted_boxes, sorted_score, targets):
        TO_REMOVE = 1
        labels = targets.get_field('labels')

        # output = np.zeros((sorted_boxes.shape[0].numpy(),))
        # pdb.set_trace()
        # output_list = []
        output_reg_list = []
        output_d_list = []
        for i in range(self.fg_class):
            cls_label_indice = torch.nonzero(labels == (i + 1))
            cls_target_bbox = targets.bbox[cls_label_indice[:, 0]]

            # todo: avoid None gt situation
            num_valid_gt = len(cls_label_indice)

            if num_valid_gt == 0:

                output = np.zeros(([*sorted_boxes.shape][0], len(self.target_thresh)))
                output_d = - np.ones(([*sorted_boxes.shape][0], len(self.target_thresh)))
                # output_reg = output.copy()
                # output_list.append(output)
                output_reg_list.append(output)
                output_d_list.append(output_d)
            else:
                output_list_per_class = []
                output_reg_list_per_class = []
                eye_matrix = np.eye(num_valid_gt)
                score_per_class = sorted_score[:, i: i + 1].cpu().numpy()
                boxes = sorted_boxes[:, i, :]
                boxes = boxes.view(-1, 4)
                area1 = (boxes[:, 2] - boxes[:, 0] + TO_REMOVE) * (boxes[:, 3] - boxes[:, 1] + TO_REMOVE)
                area2 = (cls_target_bbox[:, 2] - cls_target_bbox[:, 0] + TO_REMOVE) * (
                            cls_target_bbox[:, 3] - cls_target_bbox[:, 1] + TO_REMOVE)
                lt = torch.max(boxes[:, None, :2], cls_target_bbox[:, :2])  # [N,M,2]
                rb = torch.min(boxes[:, None, 2:], cls_target_bbox[:, 2:])  # [N,M,2]
                wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
                inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
                # [num_gt, first_n]
                iou = inter / (area1[:, None] + area2 - inter)
                iou = iou.cpu().numpy()
                for thresh in self.target_thresh:
                    # pdb.set_trace()
                    output_reg = np.max(iou, 1)
                    output_argmax = np.argmax(iou,1)
                    output_argmax[np.where(output_reg == 0)[0]] = -1
                    output_list_per_class.append(output_argmax)
                    # todo: temp comment
                    # overlap_mask = (iou > thresh)
                    # overlap_iou = iou * overlap_mask
                    # valid_bbox_indices = np.where(overlap_mask)[0]
                    # overlap_score = np.tile(score_per_class, (1, num_valid_gt))
                    # overlap_score *= overlap_mask
                    # max_overlap_indices = np.argmax(iou, axis=1)
                    # max_overlap_mask = eye_matrix[max_overlap_indices]
                    # overlap_score *= max_overlap_mask
                    # overlap_iou =overlap_iou * max_overlap_mask
                    # max_score_indices = np.argmax(overlap_score, axis=0)
                    # max_overlap_iou = overlap_iou[max_score_indices, np.arange(overlap_score.shape[1])]
                    # # output = np.zeros(([*sorted_boxes.shape][0],))
                    # output_reg = np.zeros(([*sorted_boxes.shape][0],))
                    # output_idx, inter_1, inter_2  = np.intersect1d(max_score_indices, valid_bbox_indices,return_indices=True)
                    # # output[output_idx] = 1
                    # output_reg[output_idx] = max_overlap_iou[inter_1]
                    # # output_list_per_class.append(output)
                    output_reg_list_per_class.append(output_reg)
                output_per_class = np.stack(output_list_per_class, axis=-1)
                output_reg_per_class = np.stack(output_reg_list_per_class, axis=-1)
                # pdb.set_trace()
                # output_list.append(output_per_class.view())
                output_reg_list.append(output_reg_per_class)
                output_d_list.append(output_per_class)
        # output =  np.stack(output_list, axis=1).astype(np.float32, copy=False)
        output_reg = np.stack(output_reg_list, axis=1).astype(np.float32, copy=False)
        output_d = np.stack(output_d_list, axis=1).astype(np.float32, copy=False)
        # pdb.set_trace()
        return output_reg, output_d
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


    def prepare_label_student(self, sorted_boxes, sorted_score,
                            targets):
        '''

        :param sorted_boxes: [ first n, fg_cls_num, 4]
        :param indice: [first n, fg_cls_num]
        :param sorted_score: [first n, fg_cls_num]
        :param targets: Boxlist obj
        :return: label [first n, num_thread * fg_cls_num]
        '''
        TO_REMOVE = 1
        labels = targets.get_field('labels')
        t_score = targets.get_field('scores')
        # output = np.zeros((sorted_boxes.shape[0].numpy(),))

        output_list = []
        output_score_list = []
        for i in range(self.fg_class):
            cls_label_indice = torch.nonzero(labels == (i + 1))
            cls_target_bbox = targets.bbox[cls_label_indice[:, 0]]

            cls_target_score = t_score[cls_label_indice[:, 0]].cpu().numpy()
            # todo: avoid None gt situation
            num_valid_gt = len(cls_label_indice)
            # pdb.set_trace()
            if num_valid_gt == 0:

                output = np.zeros(([*sorted_boxes.shape][0],
                                   len(self.target_thresh)))
                output_score = np.ones(([*sorted_boxes.shape][0],
                                   len(self.target_thresh)))
                output_score_list.append(output_score)
                output_list.append(output)
            else:
                output_list_score_per_class = []
                output_list_per_class = []
                eye_matrix = np.eye(num_valid_gt)
                score_per_class = sorted_score[:,
                                  i: i + 1].cpu().numpy()
                boxes = sorted_boxes[:, i, :]
                boxes = boxes.view(-1, 4)
                area1 = (boxes[:, 2] - boxes[:, 0] + TO_REMOVE) * (
                            boxes[:, 3] - boxes[:, 1] + TO_REMOVE)
                area2 = (cls_target_bbox[:, 2] - cls_target_bbox[:,
                                                 0] + TO_REMOVE) * (
                                    cls_target_bbox[:,
                                    3] - cls_target_bbox[:,
                                         1] + TO_REMOVE)
                lt = torch.max(boxes[:, None, :2],
                               cls_target_bbox[:, :2])  # [N,M,2]
                rb = torch.min(boxes[:, None, 2:],
                               cls_target_bbox[:, 2:])  # [N,M,2]
                wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
                inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
                # [num_gt, first_n]
                iou = inter / (area1[:, None] + area2 - inter)
                iou = iou.cpu().numpy()
                for thresh in self.target_thresh:
                    overlap_mask = (iou > thresh)
                    valid_bbox_indices = np.where(overlap_mask)[0]
                    overlap_score = np.tile(score_per_class,
                                            (1, num_valid_gt))
                    overlap_nms_target = np.tile(
                        cls_target_score[..., None],
                        (1, score_per_class.shape[0])).transpose(1,0)
                    # overlap_score_nms = np.tile(cls_target_score,
                    #                         (1, num_valid_gt))
                    overlap_score *= overlap_mask
                    overlap_nms_target *= overlap_mask
                    # pdb.set_trace()
                    # overlap_score_nms *=overlap_mask
                    max_overlap_indices = np.argmax(iou, axis=1)
                    max_overlap_mask = eye_matrix[max_overlap_indices]
                    overlap_score *= max_overlap_mask
                    overlap_nms_target *= max_overlap_mask

                    #np.amax(
                    # overlap_nms_target,1)
                    overlap_nms_target = np.amax(overlap_nms_target,1)
                    max_score_indices = np.argmax(overlap_score,
                                                  axis=0)


                    output = np.zeros(([*sorted_boxes.shape][0],))
                    output_nms = np.zeros(([*sorted_boxes.shape][0],))
                    output[np.intersect1d(max_score_indices,
                                          valid_bbox_indices)] = 1
                    # pdb.set_trace()
                    output_nms[np.intersect1d(max_score_indices,valid_bbox_indices)] = \
                        overlap_nms_target[np.intersect1d(max_score_indices,
                                          valid_bbox_indices)]

                    output_list_per_class.append(output)
                    output_list_score_per_class.append(output_nms)
                output_per_class = np.stack(output_list_per_class,
                                            axis=-1)
                output_score_per_class = np.stack(
                    output_list_score_per_class, axis = -1)
                output_list.append(output_per_class)
                output_score_list.append(output_score_per_class)
        output = np.stack(output_list, axis=1).astype(np.float32,
                                                      copy=False)
        output_score= np.stack(output_score_list, axis=1).astype(
            np.float32,
                                                      copy=False)
        return output, output_score

    def prepare_ranking_student(self,cls_score, box_regression,
                             proposals,  nms_idx, nms_result_t, nms_score,
                                mt_learning = None ):

        boxes_per_image = [len(box) for box in proposals]

        concat_boxes = torch.cat([a.bbox for a in proposals], dim=0)
        image_shapes = [box.size for box in proposals]
        # if self.training:
        #     labels = torch.cat([box.get_field('labels') for box in proposals])
        #     labels = labels.split(boxes_per_image, dim=0)
        # pdb.set_trace()
        objectness = [f.get_field('objectness') for f in proposals]
        proposals = self.boxcoder.decode(
            box_regression.view(sum(boxes_per_image), -1),
            concat_boxes
        )
        # todo: use iou to decide the new label
        # pdb.set_trace()

        proposals = proposals.split(boxes_per_image, dim=0)
        cls_score = cls_score.split(boxes_per_image, dim=0)

        results = []
        num_classes = self.fg_class + 1
        # pdb.set_trace()
        for prob, boxes_per_img, image_shape, idx, result_t,score_t,\
            obj in zip(cls_score, proposals, image_shapes, nms_idx,
                       [nms_result_t], [nms_score], objectness):

            boxes = boxes_per_img.reshape(-1, 4 * num_classes)
            scores = prob.reshape(-1, num_classes)
            # pdb.set_trace()

            cat_boxes = []
            for j in range(1, num_classes):
                # skip class 0, because it is the background class
                cls_boxes = boxes[:, j * 4: (j + 1) * 4]
                cat_boxes.append(cls_boxes)
            boxes = torch.cat(
                [bbox[:, :, None] for bbox in cat_boxes], dim=2)
            # scores =  torch.cat([s for s in cat_score])

            scores = scores[:, 1:]
            ori_scores = scores
            num_roi = boxes.shape[0]
            if num_roi <= self.first_n:
                first_n = num_roi
                # pdb.set_trace()
            else:
                first_n = self.first_n
            sorted_scores = torch.gather(scores, dim = 0, index = idx)
            ori_scores = ori_scores[idx]
            sorted_obj = obj[idx]
            sorted_boxes = boxes[idx]

            if self.class_agnostic:
                # [first_n, num_fg_class, 4]
                sorted_boxes = torch.squeeze(sorted_boxes, dim=-1)
            else:
                mask = torch.arange(0, num_classes - 1).to(
                    device=self.device)
                mask = mask.view(1, -1, 1, 1).expand(first_n,
                                                     num_classes - 1,
                                                     4, 1)
                sorted_boxes = torch.gather(sorted_boxes, dim=3,
                                            index=mask).squeeze(dim=3)
            sorted_boxes = sorted_boxes.view(
                first_n * (num_classes - 1), -1)
            # pdb.set_trace()
            # idx = idx.view(-1)
            sorted_obj = sorted_obj.view(first_n * (num_classes - 1))
            boxlist = BoxList(sorted_boxes, image_shape,
                              mode="xyxy", )
            boxlist.add_field('sorted_idx', idx)
            boxlist.add_field('objectness', sorted_obj)
            boxlist.extra_fields['scores'] = sorted_scores
            boxlist.extra_fields["all_scores"] = ori_scores
            boxlist.add_field('nms_result_t', result_t[:,:,0])
            boxlist.add_field('nms_score', score_t[:,:,0])
            boxlist = boxlist.clip_to_image(remove_empty=False)
            results.append(boxlist)
        return  results



    def prepare_ranking(self,  cls_score, box_regression,
                        proposals, targets,mt_learning = None , reg_iou = False):
        '''
        :param score:[num_per_img*batchsize, class]
        :param proposal: list of boxlist
        :return:
        '''
        # if is not train, targets is None which should be set into a none list

        # pdb.set_trace()
        boxes_per_image = [len(box) for box in proposals]

        concat_boxes = torch.cat([a.bbox for a in proposals], dim=0)
        image_shapes = [box.size for box in proposals]
        # if self.training:
        #     labels = torch.cat([box.get_field('labels') for box in proposals])
        #     labels = labels.split(boxes_per_image, dim=0)
        # pdb.set_trace()
        objectness = [f.get_field('objectness') for f in proposals]
        proposals = self.boxcoder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        # todo: use iou to decide the new label
        # pdb.set_trace()


        proposals = proposals.split(boxes_per_image, dim=0)
        cls_score = cls_score.split(boxes_per_image, dim=0)

        results = []
        if self.training:
        # if idx_t is None:
            for prob, boxes_per_img, image_shape,  target,  obj in zip(
                    cls_score, proposals, image_shapes, targets, objectness           ):
                # try:
                #     cls_score_c = cls_score[0].detach().cpu().numpy()
                #     boxes_per_img_c = boxes_per_img.detach().cpu().numpy()
                #     obj_c = obj.detach().cpu().numpy()

                boxlist = self.filter_results(boxes_per_img,
                                      target, prob,
                                      image_shape,
                                      self.fg_class+1, obj,
                                      mt_learning, reg_iou)
                # except:
                #     pdb.set_trace()
                results.append(boxlist)
            # else:
            #     for prob, boxes_per_img, image_shape,  target,  \
            #         obj, i_t in zip(
            #             cls_score, proposals, image_shapes,
            #             targets, objectness  ,idx_t         ):
            #         boxlist = self.filter_results(boxes_per_img,
            #                                       target, prob,
            #                                       image_shape,
            #                                       self.fg_class+1,
            #                                       obj, i_t)
            #         results.append(boxlist)

        else:
            # test do not have target
            for prob, boxes_per_img, image_shape, obj in zip(
                    cls_score, proposals, image_shapes, objectness
            ):
                boxlist = self.filter_results(boxes_per_img, None, prob, image_shape,  self.fg_class+1, obj,reg_iou=reg_iou)
                results.append(boxlist)

        return results

    def prepare_boxlist(self, boxes, scores, image_shape, label, num_classes):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4 * num_classes)
        scores = scores.reshape(-1, num_classes)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        boxlist.add_field('labels', label)
        return boxlist

    def filter_results(self, boxes,  targets, scores, image_shape ,
                       num_classes , obj,mt_learning = False, reg_iou = False):
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


        # if i_t is None:

        sorted_scores, indices = torch.topk(scores, first_n, dim= 0, largest = True, sorted = True)

        if obj.shape[0]<first_n:
            indices = indices[:obj.shape[0]]
            sorted_scores = sorted_scores[:obj.shape[0]]
        if indices.shape[1] !=2:
            pdb.set_trace()
        # else:
        #     pdb.set_trace()
        #     sorted_scores = scores[i_t]
        #     indices = i_t
        # class_idx = torch.range(0, num_classes-1).to(
        #             self.device).long().reshape(-1,1)
        # reshaped_scores = sorted_score
        ori_scores = ori_scores[indices]
        # if not self.training:
        #     max_scores_per_class = torch.max(sorted_scores, dim = 0)
        #     max_scores_per_class = max_scores_per_class.cpu().numpy()
        #     valid_class_thresh = np.minimum(self.target_thresh, max_scores_per_class.max())
        #     valid_class_indices = np.where(max_scores_per_class >= valid_class_thresh)[0]
        #     invalid_class_indices = np.where(max_scores_per_class < valid_class_thresh)[0]
        #     num_valid_classes = len(valid_class_indices)
        #     valid_class_indices_nd = torch.from_numpy(valid_class_indices).gpu()

        # [first_n, num_fg_class, 4, num_fg_cls]
        sorted_obj = obj[indices]
        # try :
        sorted_boxes = boxes[indices]
        # except:
        #     pdb.set_trace()
        # pdb.set_trace()
        if self.class_agnostic:
            # [first_n, num_fg_class, 4]
            sorted_boxes = torch.squeeze(sorted_boxes, dim = -1)
        else:
            # pdb.set_trace()

            mask = torch.arange(0, num_classes - 1).to(
            device=self.device)

            mask = mask.view(1,-1,1,1).expand(first_n, num_classes-1, 4 , 1)
            # [first_n, num_fg_class, 4]
            sorted_boxes = torch.gather(sorted_boxes, dim = 3, index = mask).squeeze(dim = 3)
       # package them into boxlist
       #  pdb.set_trace()
        if self.training:
            if mt_learning:
                labels, nms_target = self.prepare_label_student(
                    sorted_boxes, sorted_scores, targets)
                nms_target = torch.from_numpy(nms_target).to(
                    sorted_scores.device)
            else:
                labels = self.prepare_label(sorted_boxes, sorted_scores,
                                     targets)
                labels_cls = torch.from_numpy(labels).to(sorted_scores.device)
            if reg_iou:
                if self.discriminative:
                    labels_reg, labels_d = self.prepare_discriminative_label(sorted_boxes, sorted_scores,targets)
                    labels_reg = torch.from_numpy(labels_reg).to(sorted_scores.device)
                    labels_d  = torch.from_numpy(labels_d).to(sorted_scores.device)
                else:
                    labels_reg = self.prepare_reg_label(sorted_boxes, sorted_scores,
                                         targets)
                    # labels_cls= torch.from_numpy(labels_cls).to(sorted_scores.device)
                    labels_reg = torch.from_numpy(labels_reg).to(sorted_scores.device)


        sorted_boxes = sorted_boxes.view(first_n * (num_classes-1), -1)
        # pdb.set_trace()
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
                if self.discriminative:
                    boxlist.extra_fields['labels_iou_d'] = labels_d
            else:
                boxlist.extra_fields['labels'] = labels_cls
            if mt_learning:
                boxlist.add_field('mt_nms', nms_target)
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


def cluster(prediction, bandwidth):
    ms = MeanShift(bandwidth, bin_seeding=True)
    # print ('Mean shift clustering, might take some time ...')
    # tic = time.time()
    ms.fit(prediction)
    # print ('time for clustering', time.time() - tic)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    num_clusters = cluster_centers.shape[0]

    return num_clusters, labels, cluster_centers
def remove_idx_in_same_cluster(cluster_ids,scores,unique_id):
    index_list = []
    scores = scores.cpu().numpy()
    for i in unique_id.tolist():
         index = np.nonzero(cluster_ids == i)
         score = scores[index[0]]
         max_score = np.argmax(score)
         index_list.append(index[0][max_score])
    return index_list

class Discriminative_Loss(nn.Module):
    def __init__(self,delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0, beta=1.0, gamma=0.001, usegpu=True, size_average=True):

        super(Discriminative_Loss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target, n_clusters=None):
        target = target.long() + 1
        num, nb_cls, _ = target.shape
        _,_,embed_dim = input.shape
        _loss = []
        # pdb.set_trace()
        # input = torch.norm(input,dim=1)
        for cls in range(nb_cls):
            fg_idx = torch.nonzero(target[:,cls,:])[:, 0]
            if fg_idx.shape[0]<=2:
                continue
            loss = self._d_loss_per_cls(input[:,cls,:], target[:,cls,:], embed_dim)
            _loss.append(loss)
        # pdb.set_trace()
        d_loss = torch.mean(torch.stack(_loss))
        return d_loss
    def _d_loss_per_cls(self, input, target, embed_dim):
        # do not cal cluster-0
        # pdb.set_trace()
        fg_idx = torch.nonzero(target)[:,0]
        target = target[fg_idx]
        input = input[fg_idx]
        unique_labels, unique_number = torch.unique(target,return_counts=True)
        scatter_mask = target.repeat(1, embed_dim)
        mean = ts.scatter_mean(input,scatter_mask , dim = 0)
        # calculate var

        sub = input - mean[target[:,0]]
        sub = torch.clamp(torch.abs(sub).sum(1) - self.delta_var, min=0)** 2
        sum_sub = ts.scatter_add(sub, target[:,0], dim = 0)
        var = sum_sub[unique_labels]/unique_number.float()
        var = var.sum(0)/unique_labels.shape[0]
        # calculate dist
        # remove zero cluster
        mean = torch.index_select(mean, dim = 0, index = unique_labels)
        num_cluster = mean.shape[0]
        print('nbcluster',num_cluster)
        mean_a = mean[:,None,:]
        mean_b = mean[None,:,:]
        diff = mean_a  - mean_b
        margin = 2 * self.delta_dist * (1.0 - torch.eye(num_cluster))
        margin = margin.to(mean.device)
        c_dist = torch.sum(torch.clamp(margin - torch.abs(diff).sum(-1), min=0) ** 2)
        dist =  (c_dist+ 1e-10) / (2 * num_cluster * (num_cluster - 1 + 1e-10))
        # dist = dist.clamp(max=100)

        # calculate reg
        reg =  torch.abs(mean).sum()/num_cluster
        loss =  self.alpha * var + self.beta * dist + self.gamma * reg
        print('var',var.data, 'dist',dist, 'reg', 0.001 * reg.data)
        return loss

        # F.cross_entropy
            #####################################
            # util function for relation module #
            #####################################


def make_refined_matcher(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    return matcher

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

def extract_multi_position_matrix_polar(boxes, iou, clswise = False, ):
    # pdb.set_trace()
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

    delta_y = cy - cy.permute(0,2,1)

    delta_w = w / w.permute(0,2,1)
    delta_h = h / h.permute(0,2,1)

    delta_r = torch.sqrt(delta_y**2 + delta_x**2)
    delta_r = torch.clamp(delta_r, min = 1e-3)
    delta_r = torch.log(delta_r)
    theta = torch.atan2(delta_y, delta_x)

    delta_w = torch.log(delta_w)
    delta_h = torch.log(delta_h)

    size = delta_h.size()
    delta_w = delta_w.view(size[0], size[1],size[2], 1)
    delta_h = delta_h.view(size[0], size[1],size[2], 1)
    delta_r = delta_r.view(size[0], size[1],size[2], 1)
    theta = theta.view(size[0], size[1],size[2], 1)
    if iou is not None:
        iou_shape = iou.shape
        position_mat = torch.cat((delta_r, theta, delta_w, delta_h, iou.view(1, iou_shape[0], iou_shape[1], 1)), -1)
    else:
        position_mat = torch.cat((delta_r, theta, delta_w, delta_h),-1)


    return position_mat


