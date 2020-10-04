import datetime
import logging
import time
from maskrcnn_benchmark.solver.build import make_lr_scheduler
import torch
import torch.distributed as dist
import pdb
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import sigmoid_rampdown,sigmoid_rampup
import numpy as np
from collections import Counter
from functools import partial


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        reduced_losses = {}
        with torch.no_grad():
            for k, v in loss_dict.items():
                reduced_losses[k] = v
        return reduced_losses
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def _gather_data_from_loaders(source, nolabel):
    # (images_s1,targets_s, _) = source
    (images_s1,targets_s, _) = source
    (images_nl_list,_) = nolabel
    return images_nl_list,images_s1, targets_s

def init_teacher_weight(model_s, model_t):

    for ema_param, param in zip(model_t.parameters(),
                                model_s.parameters()):

        ema_param.data.mul_(0.).add_(1, param.data)

def accumulate_loss_dict(losses):
    c = Counter()
    for loss in losses:
        c.update(loss)
    c = dict(c)
    for k in c.keys():
        c[k] = c[k]/len(losses)
    return c


def weight_sum_losses(loss_dict, step, rampup_length,
                      rampdown_length, total_length,
                      l = 1,
                      balanced  =None, start_mt = 1000):
    '''
    :param loss_dict: losses from model
    :param step: current iter
    :param rampup_length: steps for consistency loss start from
    0 to 1
    :param rampdown_length: steps for consistency loss from 1 to 0
    at the end of training
    :param total_length: training iters
    :param l: lambda_value to balance supervised and consistency loss
    :param balanced: the weight for each loss term
    :param start_mt: start the mean teacher framework when the
    student model is relative stable.
    :return:  weighted loss

    loss = sup_loss[i] * balanced[i] + weight(Time) * (consist_loss[
    j]*balanced[j] )

    '''
    if (step - start_mt) < rampup_length and (step -start_mt)>0:
        weight = l * sigmoid_rampup(step-start_mt, rampup_length)
    elif (total_length - step)< rampdown_length:
        weight = l * sigmoid_rampdown(total_length - step, rampup_length)
    else:
        weight = l
    # pdb.set_trace()
    loss = {}
    # pdb.set_trace()
    for k,v in loss_dict.items():
        if 'mt' in k:
            # mean teacher loss
            loss.update({k: weight * v})
        else:
            loss.update({k: v})
    for k,v in loss.items():
            try:
                loss.update({k:v * balanced[k]})
            except:
                continue
    return loss

class MTtrainer(object):
    def __init__(self,model_s, model_t, data_loader, optimizer,
            scheduler,ckpt_s, ckpt_t,checkpoint_period, cfg):
        super(MTtrainer, self).__init__()
        self.cfg = cfg
        self.logger =  logging.getLogger("maskrcnn_benchmark.trainer")
        self.scheduler =  scheduler
        self.meters = MetricLogger(delimiter="  ")
        self.max_iter = len(data_loader['source'])
        self.start_iter = 0

        self.student = model_s
        self.student_bs = cfg.MT.AUG_S
        self.teacher = model_t
        self.teacher_bs = cfg.MT.AUG_K
        # pdb.set_trace()
        self.device_s = torch.device('cuda:0')
        self.device_t = torch.device('cuda:0')
        self.checkpoint_period = checkpoint_period
        self.ckpt_s = ckpt_s
        self.ckpt_t = ckpt_t
        self.optimizer = optimizer
        # mt hyperparameter
        self.lambda_value = cfg.MT.LAMBDA
        self.alpha = cfg.MT.ALPHA
        self.alpha_rampup = cfg.MT.ALPHA_RAMPUP
        self.rampup_step = cfg.MT.RAMPUP_STEP
        self.rampdown_step = cfg.MT.RAMPDOWN_STEP
        self.start_mt = cfg.MT.START_MT
        #loss weight
        self.balanced_weight =  {
                 'mt_classifier': (cfg.MT.CLS_LOSS),
                 'nms_loss': cfg.MODEL.RELATION_NMS.LOSS,
                 'mt_fg_loss':cfg.MT.FG_HINT,
                 }

        self.dataloader_s = data_loader['source']
        if self.cfg.DATASETS.NO_LABEL:
            self.dataloader_u = data_loader['no_label']
        if cfg.DATASETS.SYN:
            # todo: add in training
            self.dataloader_syn = data_loader['synthesis']
        self.n_step_unlabel = cfg.MT.N_STEP_UNLABEL
        self.weight_sum_loss = partial(weight_sum_losses,
                                       rampup_length
                                       =self.rampup_step,
                                       rampdown_length =
                                       self.rampdown_step,
                                       total_length = self.max_iter,
                                       l = self.lambda_value,
                                       balanced =
                                       self.balanced_weight,
                                       start_mt = self.start_mt)

    def train(self):
        self.student.train()
        self.teacher.eval()
        start_training_time = time.time()
        # pdb.set_trace()
        end = time.time()

        for iteration, data_source in enumerate(self.dataloader_s,self.start_iter):
            # if iteration>self.start_mt:
            #     self.checkpoint_period = 50
            data_s, target_s, _ = data_source
            loss_dict = self.forward_source(data_s,target_s)
            if iteration> self.start_mt and self.lambda_value>0 and self.cfg.DATASETS.NO_LABEL:
                unlabel_loss_dict = self.forward_unlabel()
                loss_dict.update(unlabel_loss_dict)

            # calculate losses
            self.scheduler.step()
            losses_dict = self.weight_sum_loss(loss_dict, iteration)
            losses = sum(loss for k, loss in losses_dict.items())
            # reduce losses over all GPUs for logging purposes
            # todo add loss weight to different loss
            loss_dict_reduced = reduce_loss_dict(losses_dict)
            losses_reduced = sum(
                loss for loss in loss_dict_reduced.values())
            self.meters.update(loss=losses_reduced, **loss_dict_reduced)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            # update teacher
            if self.lambda_value > 0 and iteration>(self.start_mt-10):
                self.update_teacher(iteration-(self.start_mt-10))

            batch_time = time.time() - end
            end = time.time()
            self.meters.update(time=batch_time)

            eta_seconds = self.meters.time.global_avg * (self.max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if iteration % 20 == 0 or iteration == self.max_iter:
                self.logger.info(
                    self.meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(self.meters),
                        lr=self.optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if iteration % self.checkpoint_period == 0:
                self.save_model(iteration)
            if iteration == self.max_iter:
                self.save_model(final = True)
        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info("Total training time: {} ({:.4f} s / it)"
            .format(total_time_str, total_training_time / (self.max_iter)))

    def save_model(self, iteration=0, final = False):
        if iteration > 0 and not final:
            self.ckpt_s.save("model_{:07d}".format(iteration))
            if iteration > self.start_mt:
                self.ckpt_t.save("t_model_{:07d}".format(iteration))
        if final:
            self.ckpt_s.save("model_final".format(iteration))
            if iteration > self.start_mt:
                self.ckpt_t.save("t_model_final".format(iteration))

    def forward_source(self, image, target):
        image = image.to(self.device_s)
        target = [t.to(self.device_s) for t in target]
        losses_dict = self.student(image, target)
        return losses_dict

    def forward_unlabel(self):
        # image list has N augmented images
        unlabel_loss_dict = []
        loss_dict = {}
        for _ in range(self.n_step_unlabel):
            data_unlabel = next(iter(self.dataloader_u))
            data_u_list, _ = data_unlabel
            teacher_list = data_u_list[:self.teacher_bs]
            student = data_u_list[-self.student_bs:]
            teacher_list = [f.to(self.device_t) for f in teacher_list]
            student =[ s.to(self.device_s) for s in student]
            with torch.no_grad():
                try:
                    teacher_results = self.teacher.forward_teacher(
                        teacher_list)
                except:
                    self.logger.info(
                        'teacher bug, may because no bbox, skip this pair')
                    continue
            loss_dict = self.student.forward_student(student,
                                                     teacher_results)
            unlabel_loss_dict.append(loss_dict)
        # average the unlabel_loss
        if unlabel_loss_dict == []:
            # skip this iter
            self.optimizer.zero_grad()
            return loss_dict
        loss_dict = accumulate_loss_dict(unlabel_loss_dict)
        return loss_dict

    def update_teacher(self, iter):
        alpha = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.teacher.parameters(),
                                    self.student.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



