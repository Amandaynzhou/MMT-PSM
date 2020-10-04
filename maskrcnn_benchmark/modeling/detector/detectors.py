# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}

def build_detection_model(cfg, is_teacher = False, is_student = False):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg, is_teacher, is_student)
