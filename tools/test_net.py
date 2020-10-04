# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import \
    setup_environment  # noqa F401 isort:skip

import argparse
import os
import json
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader,make_mt_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import pdb
import scipy.io as sio
import matplotlib.pyplot as plt

def _get_models_in_dir(path):
    if os.path.isdir(path):
        models = []
        for file in os.listdir(path):
            # skip 000000.pth
            if file.endswith('.pth') and '_00000' not in file:
                models.append(os.path.join(path, file))
    else:
        models = [path]
    return models


def _find_yaml_in_dir(directory, isdir=True):
    if not isdir:
        directory = '/'.join(directory.split('/')[:-1])
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml'):
                return os.path.join(directory, file)


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Inference")
    parser.add_argument(
        '--test_path',
        default=None,
        help='test model path or a single model weight file, if it is a path, will test all models inside the path'
    )
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--gpuid", default='0', help='set gpu id')
    args = parser.parse_args()

    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    
    #pdb.set_trace()
    if args.test_path is not None:
        _cfg = _find_yaml_in_dir(args.test_path, os.path.isdir(args.test_path))
        model_list = _get_models_in_dir(args.test_path)
        save_dir = args.test_path if os.path.isdir(
            args.test_path) else '/'.join((args.test_path).split('/')[:-1])
    else:
        _cfg = args.config_file
        model_list = [cfg.MODEL.WEIGHT]
        save_dir = ""
    #pdb.set_trace()
    cfg.merge_from_file(_cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank(), test=True)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    collect_evaluate_results = {}

    for model_path in model_list:
        results = test_once(cfg, save_dir, model_path, distributed)
        # pdb.set_trace()
        if cfg.DATASETS.MODE_IN_TEST =='test' and \
                'isbi' in args.config_file:
            sio.savemat(os.path.join(save_dir,
                                     'isbi2015_%s.mat'%model_path.split(
                                         '/')[-1].strip('.pth') ),
                        results[0])
            continue


        else:
            collect_evaluate_results[model_path] = results[0].results['segm']
    if cfg.DATASETS.MODE_IN_TEST == 'test' and 'isbi' in args.config_file:
        return 0

        # pdb.set_trace()
    # pdb.set_trace()
    # todo: prepare print
    x_list = []
    AJI = []
    MAP = []
    AP50 = []
    AP75 = []
    AP85 = []
    best_student ={}
    best_teacher = {}
    for k,v in collect_evaluate_results.items():
        if 't_model' not in k:
            name = k.split('/')[-1].strip('.pth').strip('model_')
            if 'fina' in name:
                continue
            name = int(name)
            x_list.append(name)
            aji = (v['AJI']['nuclei'] + v['AJI']['cytoplasm'])/2
            map = v['mAP']['all']
            ap50 = v['AP50']['all']
            ap75 = v['AP75']['all']
            ap85 = v['AP85']['all']
            AJI.append(aji)
            MAP.append(map)
            AP50.append(ap50)
            AP75.append(ap75)
            AP85.append(ap85)
    plt.plot(x_list, AJI,color='green',label='AJI')
    plt.plot(x_list,MAP,color='red',label='mAP')
    plt.plot(x_list,AP50,color = 'skyblue',label='AP50')
    plt.plot(x_list,AP75,color='blue',label='AP75')
    plt.plot(x_list, AP85, color='black', label='AP85')
    # show number
    anno_xs = []
    anno_map = []
    anno_aji = []
    # pdb.set_trace()
    idxmap = MAP.index(max(MAP))
    idxaji = AJI.index(max(AJI))
    anno_xs.append(x_list[idxmap])
    anno_map.append(MAP[idxmap])
    anno_aji.append(AJI[idxmap])
    anno_xs.append(x_list[idxaji])
    anno_map.append(MAP[idxaji])
    anno_aji.append(AJI[idxaji])
    best_student['map_best'] = {x_list[idxmap]: {'map':MAP[idxmap], 'aji':AJI[idxmap]}}
    best_student['aji_best']  = {x_list[idxaji ]: {'map':MAP[idxaji ], 'aji':AJI[idxaji ]}}

    for a, b in zip(anno_xs, anno_aji):
        plt.annotate('(%.0f,%.4f)'%(a,b), (a,b))
    for a, b in zip(anno_xs, anno_map):
        plt.annotate('(%.0f,%.4f)'%(a,b), (a,b))
  # for teacher
    x_list = []
    AJI = []
    MAP = []
    AP50 = []
    AP75 = []
    AP85 = []
    for k, v in collect_evaluate_results.items():
        if 't_model' in k:
            name = k.split('/')[-1].strip('.pth').strip('model_')
            if 'fina' in name:
                continue
            name = int(name)
            x_list.append(name)
            aji = (v['AJI']['nuclei'] + v['AJI']['cytoplasm']) / 2
            map = v['mAP']['all']
            ap50 = v['AP50']['all']
            ap75 = v['AP75']['all']
            ap85 = v['AP85']['all']
            AJI.append(aji)
            MAP.append(map)
            AP50.append(ap50)
            AP75.append(ap75)
            AP85.append(ap85)
    if len(AJI)>0:
        plt.plot(x_list, AJI, '--', color='green', label='AJI')
        plt.plot(x_list, MAP,'--', color='red', label='mAP')
        plt.plot(x_list, AP50,'--', color='skyblue', label='AP50')
        plt.plot(x_list, AP75,'--', color='blue', label='AP75')
        plt.plot(x_list, AP85, '--',color='black', label='AP85')
        # show number
        anno_xs = []
        anno_map = []
        anno_aji = []
        # pdb.set_trace()
        idxmap = MAP.index(max(MAP))
        idxaji = AJI.index(max(AJI))
        anno_xs.append(x_list[idxmap])
        anno_map.append(MAP[idxmap])
        anno_aji.append(AJI[idxmap])
        anno_xs.append(x_list[idxaji])
        anno_map.append(MAP[idxaji])
        anno_aji.append(AJI[idxaji])
        best_teacher['map_best'] = {x_list[idxmap]: {'map': MAP[idxmap], 'aji': AJI[idxmap]}}
        best_teacher['aji_best'] = {x_list[idxaji]: {'map': MAP[idxaji], 'aji': AJI[idxaji]}}
        for a, b in zip(anno_xs, anno_aji):
            plt.annotate('(%.0f,%.4f)' % (a, b), (a, b))
        for a, b in zip(anno_xs, anno_map):
            plt.annotate('(%.0f,%.4f)' % (a, b), (a, b))

    plt.legend()
    plt.savefig(os.path.join(save_dir,"result.jpg"))

    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(collect_evaluate_results, f)
    # write best result
    with open(os.path.join(save_dir, 'best_result.json'), 'w') as f:
        json.dump({'student':best_student, 'teacher':best_teacher},f)



def test_once(cfg, save_dir, weight_name, distributed):
    torch.cuda.empty_cache()

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = save_dir
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(weight_name, test=True)

    iou_types = ()  # ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST if not cfg.TEST.GEN else cfg.DATASETS.GEN
    if output_dir:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(output_dir, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False,
                                        is_distributed=distributed)
    results = []
    if not cfg.TEST.GEN:
        for output_folder, dataset_name, data_loader_val in zip(output_folders,
                                                                dataset_names,
                                                                data_loaders_val):
            result, _ = inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                generate_data=cfg.TEST.GEN,
                visual_num= cfg.TEST.VISUAL_NUM
            )
            # pdb.set_trace()
            results.append(result)

            synchronize()

        return results
    else:
        for output_folder, dataset_name, data_loader_val in zip(output_folders,
                                                                dataset_names,
                                                                data_loaders_val):
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                generate_data=cfg.TEST.GEN
            )
            # pdb.set_trace()


if __name__ == "__main__":
    main()
