# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from maskrcnn_benchmark.utils.model_serialization import load_state_dict, strip_prefix_if_present, align_and_update_state_dicts
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url
import pdb

class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,

    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        # pdb.set_trace()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))

        # import time
        # time.sleep(10)
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load_extra_data(self, f = None):
        checkpoint = {}
        if not f:
            self.logger.info("No checkpoint found. Initializing model from scratch")
        else:
            if 'e2e_mask_rcnn_R_50_FPN_1x.pth' in f:
                self.transfer_learning = True
                checkpoint['iteration'] = -1
        return checkpoint


    def load(self, f=None, test = False):
        if test:
            self.logger.info("Loading checkpoint from {}".format(f))

            checkpoint = self._load_file(f)

            load_state_dict(self.model, checkpoint.pop("model"))
        else:
            if 'e2e_mask_rcnn_R_50_FPN_1x.pth' in f:
                self.transfer_learning = True
            else:
                self.transfer_learning = False

            if  not self.transfer_learning and self.has_checkpoint():

                # override argument with existing checkpoint
                self.transfer_learning = False
                f = self.get_checkpoint_file()
                # f  = False
            if not f:
                # no checkpoint could be found
                self.logger.info("No checkpoint found. Initializing model from scratch")
                return {}
            self.logger.info("Loading checkpoint from {}".format(f))
            checkpoint = self._load_file(f)
            # delete this two because we add new module which transfer learning model does not have
            del checkpoint['scheduler'], checkpoint['optimizer']
            self._load_model(checkpoint)
            if self.transfer_learning:
                # default last epoch of loaded weight is 89999
                checkpoint['iteration'] = -1

            if not self.transfer_learning:
            # if use transfer learning , do not load pretrain model scheduler and optimizer
                if "optimizer" in checkpoint and self.optimizer:
                    self.logger.info("Loading optimizer from {}".format(f))

                    # pdb.set_trace()
                    # pdb.set_trace()
                    self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
                if "scheduler" in checkpoint and self.scheduler:
                    self.logger.info("Loading scheduler from {}".format(f))
                    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))




        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")

        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        # pdb.set_trace()
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        # pdb.set_trace()
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
            # pdb.set_trace()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""

        return last_saved.strip('\n')

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        # pdb.set_trace()
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        if not self.transfer_learning:
            load_state_dict(self.model, checkpoint.pop("model"))
        else:
            # pdb.set_trace()
            # delete roi_head.box/mask.predictor.cls_score/bbox_pred/mask_fcn_logits in state_dict
            pretrained_weights = checkpoint.pop("model")
            model_state_dict = self.model.state_dict()
            loaded_state_dict = strip_prefix_if_present(pretrained_weights , prefix="module.")
            align_and_update_state_dicts(model_state_dict, loaded_state_dict)
            model_state_dict = {k:v for k,v in model_state_dict.items() if 'cls_score' not in k and 'bbox_pred' not in k
                                and 'mask_fcn_logits' not in k}
            self.model.load_state_dict(model_state_dict, strict= False)

class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # pdb.set_trace()
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded

    def load_optimizer(self,checkpoint):
        self.logger.info("Loading optimizer from ckpt")
        self.optimizer.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")).pop("optimizer"))
