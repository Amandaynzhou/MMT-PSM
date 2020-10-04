import torch
from torch import nn
from maskrcnn_benchmark.structures.image_list import to_image_list
from torch.nn import functional as F
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import box_roi_heads,mask_roi_heads
from ..relation.relation_module import DuplicationRemovalNetwork
from maskrcnn_benchmark.modeling.roi_heads.box_head\
    .roi_box_feature_extractors import MaskRCNNFPNAdaptor
from maskrcnn_benchmark.utils.miscellaneous import _hflip,batch_hfilp
import pdb




class GeneralizedRCNN(nn.Module):
    def __init__(self, cfg, is_teacher = False, is_student = False,):
        super(GeneralizedRCNN, self).__init__()
        self.mt_learning = False
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, is_teacher)
        self.box_heads = box_roi_heads(cfg)
        self.mask_heads = mask_roi_heads(cfg, is_student)

        if cfg.MODEL.RELATION_NMS.USE_RELATION_NMS:
            self.relation_nms = DuplicationRemovalNetwork(cfg, is_teacher)
        else:
            self.relation_nms = None
        self.mt_fg_hint = cfg.MT.FG_HINT
        self.mt_cls = cfg.MT.CLS_LOSS
        self.hint_adaptor = MaskRCNNFPNAdaptor(cfg)

    def set_module_mode(self, mode):
        self.rpn.set_teacher_mode(mode)
        self.box_heads.box.set_teacher_mode(mode)
        if self.relation_nms:
            self.relation_nms.set_teacher_mode(mode)
        self.mask_heads.mask.set_teacher_mode(mode)

    def forward(self, images, targets=None, tta = None):
        """
        Supervised learning
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        x, result, losses, class_logits, box_regression = self.box_heads(features, proposals, targets)
        if self.relation_nms is not None:# if use relation module. This part is orthogonal with MT framework
            #appearance_feature, proposals, cls_score, box_reg
            class_logits = torch.nn.functional.softmax(class_logits)
            batch_size = [f.bbox.shape[0] for f in result]
            x_list = torch.split(x,batch_size)
            class_logits_list = torch.split(class_logits, batch_size)
            box_regression_list = torch.split(box_regression,batch_size)
            nms_result_list= []
            nms_loss_list = []
            if targets == None:
                targets= [None] * len(x_list)
            for xitem,re,clsl,boxr, target in zip(x_list,
                                          result,
                                          class_logits_list,
                                          box_regression_list,
                                          targets):
                nms_result, nms_loss = self.relation_nms((xitem,
                                                          [re],
                                                          clsl,
                                                          boxr,
                                                          [target]))
                nms_result_list.append(nms_result)
                nms_loss_list.append(nms_loss)
            if not self.training:
                if len(nms_result_list) ==1:
                    nms_result = nms_result_list[0]
                else:
                    nms_result = [f[0] for f in nms_result_list]
                result = nms_result
            else:
                nms_loss = [f['nms_loss'] for f in nms_loss_list]
                nms_loss ={ 'nms_loss':torch.mean(torch.stack(
                    nms_loss))}

        # in train phase, mask branch use proposals from det branch,
        # in test phase, mask branch use proposals from relation-nms branch
        result, detector_losses = self.mask_heads(losses,
                                                  features,
                                                  result,
                                                  targets,
                                                  images)



        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if self.relation_nms is not None:
                losses.update(nms_loss )
            return losses
        else:
            return result

    def forward_teacher(self, images, targets = None):
        result = None
        class_logits = None
        integral_mask_list = []
        if targets is None:
            self.set_module_mode('test')
            # test mode for teacher, this step we generate coarse
            # results and use them as targets for proposal sampling
            # therefore, to speed up, one aug-image(not k-aug version) is used to generate course result.
            images_infer = images[0]
            teacher_infer = self.forward(images_infer)

            if self.mt_fg_hint>0:
                for t in teacher_infer:
                    integral_mask = t.get_field('mask').sum(0)[0]
                    integral_mask_list.append(integral_mask)
        else:
            teacher_infer = targets
            if self.mt_fg_hint>0:
                for t in teacher_infer:
                    integral_mask = t.get_field('masks').decode(800,800)
                    integral_mask = torch.tensor(integral_mask)
                    integral_mask_list.append(integral_mask)

        self.set_module_mode('train')
        images = [to_image_list(image) for image in images]
        aug_features = self.extract_aug_feat(images)
        rpn_feat = aug_features[0]# we do mt on *HEAD* and not on *RPN*, so use the first feature to get ROI

        objectness, rpn_box_regression, sample_idx, label, \
        proposals, _, ffi_boxes\
            = self.rpn.forward_teacher(images[0], rpn_feat,teacher_infer)
        # todo: rewrite  boxhead forward teacher

        if self.cfg.MT.FG_HINT or self.cfg.MT.HINT or self.cfg.MT.ODKD:
            # ffi/fg/naive feat loss
            embeddings = self.get_emb_feature(aug_features)
        else:
            embeddings = None
        if self.cfg.MT.CLS_LOSS or self.cfg.MT.ODKD:
            head_feat, result, losses, class_logits, box_regression, \
                = self.box_heads.forward_teacher(
                    aug_features, proposals, teacher_infer)
        # since we do not have consistency loss in mask head,
        # we skip this part.
        return {'result_t':result,
                'class_logit_t': class_logits,
                'embedding': embeddings,
                'seg_mask': integral_mask_list,
                'ffi_boxes':ffi_boxes
        }


    def forward_student(self, images, result_t):
        loss_dict = {}
        if isinstance(images,list):
            images = [to_image_list(image) for image in images]
        else:
            images = [to_image_list(images)]
        feat_list = self.extract_aug_feat(images,teacher=False)
        if self.cfg.MT.FG_HINT:
            fg_hint_loss = self.get_fg_feature_loss(feat_list,
                result_t['seg_mask'], result_t['embedding'])

            loss_dict.update(mt_fg_loss=fg_hint_loss)
        if self.cfg.MT.FFI: # compared method
            ffi_hint_loss = self.get_ffi_loss(feat_list,
                result_t['ffi_boxes'], result_t['embedding'])
            loss_dict.update(mt_hint_loss=ffi_hint_loss)
        if self.cfg.MT.ODKD:# compared method
            #naive hint loss
            naive_hint_loss = self.get_naive_hint_loss(feat_list, result_t['embedding'])
            loss_dict.update(mt_hint_loss=naive_hint_loss)
        # currently we do not conduct rpn consistent loss,
        # therefore we direct use teacher ROI in student boxhead
        # todo: rewrite boxhead.forward_student
        if self.cfg.MT.CLS_LOSS or self.cfg.MT.ODKD:
            losses = self.box_heads.forward_student(
                    feat_list, result_t['result_t'],
                    result_t['class_logit_t'],)

            loss_dict.update(losses)
        return loss_dict

    def extract_aug_feat(self, imglist, teacher = True):
        feat_list = []
        if teacher:
            for img in imglist:
                feat1 = self.backbone(img.tensors)
                img.hflip()
                feat1_f = self.backbone(img.tensors)
                feat_list.extend([feat1,feat1_f])
        else:
            for i, img in enumerate(imglist):
                if i%2==1:
                    img.hflip()
                feat = self.backbone(img.tensors)
                feat_list.append(feat)
        return feat_list


    def get_emb_feature(self,feature_list):
        embedding_list = []
        for feat in feature_list:
            embedding_list.append(self.hint_adaptor(feat))
        return embedding_list

    def get_fg_feature_loss(self, feature_list, seg_mask,
                            teacher_feat):
        student_feat = self.get_emb_feature(feature_list)

        loss = fg_hint_loss(teacher_feat,student_feat,seg_mask)
        return loss

    def get_ffi_loss(self,feature, int_mask, teacher_feature):
        student_feat = self.get_emb_feature(feature)
        loss = ffi_hint_loss(teacher_feature,student_feat,int_mask)
        return loss

    def get_naive_hint_loss(self,feature, teacher_feature):
        student_feat = self.get_emb_feature(feature)
        loss = naive_hint_loss(teacher_feature,student_feat)
        return loss

########

def fg_hint_loss(teachers, students, masks):
    # proposed hint loss for fg feature
    new_teacher = []
    for i, feat in enumerate(teachers):
        if i%2==1:
            feat = batch_hfilp(feat)
        new_teacher.append(feat)
    hi_size = [f.shape for f  in students[0]]
    mask_list = []
    masks = torch.stack(masks)
    if len(students)>1:
        ori_student_list = students[0::2]
        flip_student_list = students[1::2]
    else:
        ori_student_list = students
        flip_student_list = []
    for size in hi_size:
        mask = masks.to(new_teacher[0][0].device)
        mask = F.adaptive_avg_pool2d(mask[:,None,:,:].float(), size[2:])
        mask[mask>0.5] =1.
        mask[mask<=0.5] = 0.
        mask_list.append(mask)

    dists = []
    for feature in new_teacher:
        for student in ori_student_list:
            for s_f, t_f, msk in zip(student, feature, mask_list):
                dist = (((s_f - t_f)**2) * msk).sum()/(msk.sum() *
                                                       s_f.shape[1]
                                                       +1e-7)
                dists.append(dist)
    if len(students)>1:
        for feature in new_teacher:
            for student in flip_student_list:
                for s_f, t_f, msk in zip(student, feature, mask_list):
                    s_f = _hflip(s_f)
                    dist = (((s_f - t_f) ** 2) * msk).sum() /(msk.sum() * s_f.shape[1]+1e-7)
                    dists.append(dist)
    dist = torch.mean(torch.stack(dists))
    return dist

def ffi_hint_loss(teachers, students, masks):
    dists = []
    masks = torch.stack(masks)
    for s_f, t_f in zip(students, teachers):
        msk =  F.adaptive_avg_pool2d(masks[:,None,:,:].float(), s_f.shape[2:])
        msk[msk > 0.5] = 1.
        msk[msk <= 0.5] = 0.
        dist = (((s_f - t_f)**2) * msk).sum()/(msk.sum() * s_f.shape[1])
        dists.append(dist)
    dist = torch.mean(torch.stack(dists))
    return dist

def naive_hint_loss(teachers, students,):
    dists = []
    for s_f, t_f in zip(students, teachers):
        dist = (((s_f - t_f) ** 2) ).sum() / (s_f.numel())
        dists.append(dist)
    dist = torch.mean(torch.stack(dists))
    return dist


def old_fg_hint( teachers, students, masks):
    ori_teacher_list = teachers[0::2]

    flip_teacher_list = teachers[1::2]

    hi_size = [f.shape for f in students[0]]
    mask_list = []
    f_mask_list = []
    masks = torch.stack(masks)
    for size in hi_size:
        mask = masks.to(ori_teacher_list[0][0].device)
        mask = F.adaptive_avg_pool2d(mask[:, None, :, :].float(),
                                     size[2:])
        mask[mask > 0.5] = 1.
        mask[mask <= 0.5] = 0.
        mask_list.append(mask)
        fmsk = mask.clone()
        fmsk = _hflip(fmsk)
        f_mask_list.append(fmsk)
    dists = []

    for feature in ori_teacher_list:
        for s_f, t_f, msk in zip(students[0], feature, mask_list):
            dist = (((s_f - t_f) ** 2) * msk).sum() / (
                ( msk.sum() * s_f.shape[1])+1e-5)
            dists.append(dist)
    if len(students) > 1:
        for feature in flip_teacher_list:
            for s_f, t_f, msk in zip(students[1], feature,
                                     f_mask_list):
                dist = (((s_f - t_f) ** 2) * msk).sum() / (
                    (msk.sum() * s_f.shape[1])+1e-5)
                dists.append(dist)
    # pdb.set_trace()
    dist = torch.mean(torch.stack(dists))
    print(dist,msk.sum())
    return dist