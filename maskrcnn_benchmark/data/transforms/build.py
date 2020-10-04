# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
import pdb

def build_transforms(cfg, is_train=True, domain = 'source',):
    tta = cfg.TEST.TTA
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    if is_train and not tta:
        if domain =='no_label':
            transpart1 = T.Compose([
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_prob)]
            )
            transpart2 = T.Compose([
                T.AdjustBrightness(0.15),
                T.AdjustContrast(0.15),
                T.AdjustHue(0.05),
                # T.AdjustSaturation,
                T.RandomErasing(0.9),
                T.ToTensor(),
                normalize_transform,
            ])
            transform = [transpart1, transpart2]
        elif domain == 'source':
            transform= T.Compose(
                [
                    T.Resize(min_size, max_size),
                    T.RandomHorizontalFlip(flip_prob),
                    # T.AdjustGamma(),
                    T.AdjustBrightness(0.15),
                    T.AdjustContrast(0.15),
                    T.AdjustHue(0.05),
                    # T.AdjustSaturation,
                    T.RandomErasing(0.7),
                    T.ToTensor(),
                    normalize_transform,
                ]
            )
        else:
            print('domain is invalid, no transform is built')


    else:
        if not tta:
            transform = T.Compose(
                [
                    T.Resize(min_size, max_size),
                    T.RandomHorizontalFlip(flip_prob),
                    T.ToTensor(),
                    normalize_transform,
                ]
            )
        else:
            transpart1 = T.Compose([
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_prob)]
            )
            transpart2 = T.Compose([
                T.AdjustBrightness(0.15),
                T.AdjustContrast(0.15),
                T.AdjustHue(0.05),
                # T.AdjustSaturation,
                T.RandomErasing(0.9),
                T.ToTensor(),
                normalize_transform,
            ])
            transform = [transpart1, transpart2]
    return transform
