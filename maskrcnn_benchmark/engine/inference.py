# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm
import pdb
from maskrcnn_benchmark.data.datasets.evaluation import evaluate, generate
from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize


def compute_on_dataset(model, data_loader, device, tta= False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        try:
            if not tta:
                images, targets, image_ids = batch
                images = images.to(device)
            else:
                img, targets, image_ids = batch
                img1,img2 = img
                img1 = img1.to(device)
                img2 = img2.to(device)
                images = (img1,img2)
            with torch.no_grad():
                # pdb.set_trace()
                output = model(images, tta = tta)
                # pdb.set_trace()
                output = [o.to(cpu_device) for o in output]
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
        except:
            continue
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    # pdb.set_trace()
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    try:
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("maskrcnn_benchmark.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the evaluation"
            )
    except:
        pdb.set_trace()
    # convert to a list

    # predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        generate_data = False,
        visual_num = 0
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()

    predictions = compute_on_dataset(model, data_loader, device, tta = False)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    # pdb.set_trace()
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        visual_num = visual_num
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
