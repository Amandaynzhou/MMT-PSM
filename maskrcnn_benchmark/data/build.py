# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging

import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file

from . import datasets as D
from . import samplers
import pdb
from .collate_batch import BatchCollator,\
    TTABatchCollator,BatchCollatorWoLabel_Compared,\
    BatchCollatorWoLabelK
from .transforms import build_transforms


def build_dataset(dataset_list, transforms, dataset_catalog,
                  is_train=True, gen_fake = 0, gen_true=0, aug_k =
                  2, syn_mt = False):
    """
    Arguments:
        syn_mt: use mean teacher to train synthesis data
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(
                dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        # pdb.set_trace()
        data = dataset_catalog.get(dataset_name, aug_k, syn_mt)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        if data["factory"] in ["PapNucleiGenTrueDataset",
                               'PapNucleiSynDataset',
                               'PapNucleiSynMTDataset']:
            args["gen_fake"] = gen_fake
            args["gen_true"] = gen_true
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        # pdb.set_trace()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(
            img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_batch,
        num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler



def build_mt_data_loader(dataset, shuffle, is_distributed,aspect_grouping, images_per_gpu,
        num_iters, start_iter,cfg,is_train,collator):
    # import pdb;pdb.set_trace()
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_gpu,
        num_iters, start_iter
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS//4 if is_train else 1

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )
    return data_loader


def make_mt_data_loader(cfg, is_train=True, is_distributed=False,
                     start_iter=0, mode = 'source', img_ratio = 1.):
    num_gpus = get_world_size()
    dataset_list_dict = {'source':'papnuclei_source',
                        'no_label':'papnuclei_no_label',}

    if is_train:
        images_per_batch = int(cfg.SOLVER.IMS_PER_BATCH * img_ratio)
        assert (
                images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [
        1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG,
        True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    if is_train:
        dataset = dataset_list_dict[mode]
    elif cfg.DATASETS.MODE_IN_TEST == 'val':
        dataset = cfg.DATASETS.VAL
    else:
        dataset= cfg.DATASETS.TEST
    transforms = build_transforms(cfg, is_train, domain = mode )
    syn_mt = True if cfg.SYN.MT_LOSS > 0 else False
    dataset = build_dataset([dataset], transforms,
                                   DatasetCatalog, is_train,
                            aug_k=cfg.MT.AUG_K + cfg.MT.AUG_S,
                            syn_mt = syn_mt, gen_true=cfg.DATASETS.GEN_TRUE)
    collators = build_collator(cfg, mode)
    dataloader = build_mt_data_loader(dataset[0],
                                             shuffle,
                                             is_distributed,
                                             aspect_grouping,
                                             images_per_gpu,
                                             num_iters, start_iter,
                                             cfg, is_train, collators)
    return dataloader

def build_collator(cfg, mode='source'):
    size_divisibility = cfg.DATALOADER.SIZE_DIVISIBILITY
    if mode == 'source':
        collator = BatchCollator(size_divisibility)
    elif mode == 'no_label':
        if not cfg.MT.ODKD and not cfg.MT.FFI:
            collator = BatchCollatorWoLabelK(size_divisibility,
                                             cfg.MT.AUG_K+ cfg.MT.AUG_S )
        else:
            collator = BatchCollatorWoLabel_Compared(size_divisibility )
    else:
        raise NotImplementedError
    return collator

def make_data_loader(cfg, is_train=True, is_distributed=False,
                     start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [
        1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG,
        True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    # import pdb;pdb.set_trace()
    if is_train:
        dataset_list = cfg.DATASETS.TRAIN
    elif cfg.DATASETS.MODE_IN_TEST == 'val':
        dataset_list = cfg.DATASETS.VAL
    elif cfg.DATASETS.MODE_IN_TEST == 'gen':
        dataset_list =   cfg.DATASETS.GEN
    else:
        dataset_list = cfg.DATASETS.TEST

    # dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(dataset_list, transforms,
                             DatasetCatalog, is_train,
                             gen_fake=cfg.DATASETS.GEN_FAKE,
                             gen_true=cfg.DATASETS.GEN_TRUE)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu,
            num_iters, start_iter
        )
        if cfg.TEST.TTA:
            collator = TTABatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        else:
            collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS if is_train else 1

        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
