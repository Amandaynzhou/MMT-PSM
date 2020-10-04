# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import json
import torch
from collections import defaultdict
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.data.build import build_dataset
from maskrcnn_benchmark.data.collate_batch import BatchCollator
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from pycocotools import mask as maskUtils
import numpy as np
import pdb
import cv2
import math

def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:
        median = (data[size // 2] + data[size // 2 - 1]) / 2
        data[0] = median
    if size % 2 == 1:
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]
def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]
def annToMask(boxlist):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    h,w = boxlist.size
    masks = boxlist.get_field('masks')
    labels = boxlist.get_field('labels')
    # pdb.set_trace()
    index = list(np.nonzero(labels==1)[:,0])
    polygons = []
    for i, seg in enumerate(masks.polygons):
        if i in index:
            polygons.append(seg)
    RLES = []
    for segm in polygons:
        rles =  maskUtils.frPyObjects([p.numpy() for p in segm.polygons], h, w )
        rle = maskUtils.merge(rles)
        RLES.append(rle)

    return RLES

def split(cfg, num_fold):
    # when calculate the overlap ratio, set the sliding window overlap = 0
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TEST

    transforms = build_transforms(cfg,False)
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, False)
    collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    data_loader = torch.utils.data.DataLoader(
        datasets[0],
        num_workers=1,
        collate_fn=collator,

    )

    name_dic = defaultdict(list)
    for i,  data in enumerate(data_loader,0):
        boxlists = data[1][0]

        masks = annToMask(boxlists)
        isCrowed = len(masks) * [0]
        iou = maskUtils.iou(masks, masks, isCrowed) - np.eye(len(masks))
        maxiou = np.max(iou,0)
        avgiou = np.mean(maxiou)
        name = datasets[0].get_img_info(i)['file_name']
        name_dic[name].append(avgiou)
    names = []
    overlapping = []
    for k, v in name_dic.items():
        names.append(k)
        overlapping.append(sum(v)/len(v))
    median = get_median(overlapping)
    # pdb.set_trace()
    hard = []
    easy = []
    for name, miou in zip(names, overlapping):
        if miou>median:
            hard.append(name)
        else:
            easy.append(name)

    # pdb.set_trace()
    easy_n  = chunks(easy, num_fold)
    hard_n = chunks(hard, num_fold)

    i = 1
    splitfile = {}
    for e,h in zip(easy_n,hard_n):
        splitfile[i] = {'easy': e, 'hard':h}
        i+=1

    with open('split.json', 'w') as f:
        json.dump(splitfile, f)

def main():

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    cv2.setNumThreads(0)
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    split(cfg, 3)



if __name__ == "__main__":
    main()
