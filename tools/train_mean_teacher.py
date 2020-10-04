# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

import argparse
import os
import shutil
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader,\
    make_mt_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import cv2
from maskrcnn_benchmark.engine.MTtrainer import MTtrainer

def train(cfg, local_rank, distributed, save_path='.', writer = None):
    # cfg.SOLVER.IMS_PER_BATCH =3# force it to 3
    model_s = build_detection_model(cfg, is_student = True)
    model_t = build_detection_model(cfg, is_teacher = True)
    device_t = torch.device('cuda:0')
    device_s = torch.device('cuda:0')
    model_s.to(device_s)
    model_t.to(device_t)
    optimizer = make_optimizer(cfg, model_s)
    scheduler = make_lr_scheduler(cfg, optimizer)
    output_dir = save_path
    save_to_disk = get_rank() == 0
    checkpointer_s = DetectronCheckpointer(
        cfg, model_s, optimizer, scheduler, output_dir, save_to_disk
    )
    checkpointer_t = DetectronCheckpointer(cfg, model_t,optimizer=None,scheduler= scheduler,save_dir= output_dir,save_to_disk= save_to_disk)
    _init_weight = 'e2e_mask_rcnn_R_50_FPN_1x.pth'
    _ = checkpointer_s.load(_init_weight,True)
    _ = checkpointer_t.load(_init_weight, True)
    sourceDataLoader = make_mt_data_loader(cfg,
                                            is_train=True,
                                            is_distributed=distributed,
                                            start_iter=0,
                                            mode='source',
                                            img_ratio=1/2)
    data_loader_dict = {'source': sourceDataLoader, }
    if cfg.DATASETS.NO_LABEL:
        noLabelDataLoader = make_mt_data_loader(cfg,
                                            is_train=True,
                                            is_distributed=distributed,
                                            start_iter=0,
                                            mode='no_label',
                                            img_ratio=1/2)

        data_loader_dict.update({'no_label': noLabelDataLoader })
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    trainer = MTtrainer(model_s,model_t,data_loader_dict,optimizer,
                        scheduler, checkpointer_s,checkpointer_t,\
              checkpoint_period, cfg)

    trainer.train()
    return model_s


def test(cfg, model, distributed, output_dir):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("segm",)
    # if cfg.MODEL.MASK_ON:
    # iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(output_dir, "inference",
                                         dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False,
                                        is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(
            output_folders,
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
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Training")
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
    parser.add_argument("--gpuid", default='0', help='set gpu id')
    parser.add_argument('--debug', action='store_true')
    cv2.setNumThreads(0)
    args = parser.parse_args()
    # pdb.set_trace()
    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    # else:
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if args.debug:
        subname = 'debug'
    else:
        subname = 'Sr1.0' \
              '_U{unlabel}' \
              '_mtCls{cls_loss}'\
              '_mtHint{hint_loss}' \
              'mt{lambdas}_pl{pl}'.format(
        unlabel = str(cfg.DATASETS.NO_LABEL),
        cls_loss = str(cfg.MT.CLS_LOSS),
        hint_loss = str(cfg.MT.FG_HINT),
        lambdas = str(cfg.MT.LAMBDA),
        pl = '1' if cfg.MT.PLTRAIN else '0'
    )

    if cfg.MT.ODKD:
        subname = 'odkd_cls{cls}_hint{hint}_lambda{la}'.format(
            cls = str(cfg.MT.CLS_LOSS),
            hint = str(cfg.MT.HINT),
            la  = str(cfg.MT.LAMBDA)
        )
    if cfg.MT.FFI:
        subname = 'FFI_hint{hint}_lambda{la}'.format(

            hint = str(cfg.MT.HINT),
            la  = str(cfg.MT.LAMBDA)
        )
    save_path = subname
    output_dir = 'result/pap/' + save_path
    if output_dir:
        mkdir(output_dir)
        shutil.copy2(args.config_file, output_dir)
    # copy a cfg file to save path
    # Writter = SummaryWriter(os.path.join('tensorboard',output_dir))
    Writter = None
    logger = setup_logger("maskrcnn_benchmark", output_dir,
                          get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info(
        "Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed, output_dir, writer = Writter)

    if not args.skip_test:
        test(cfg, model, args.distributed, output_dir)


if __name__ == "__main__":
    main()
