# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str

import base64


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for batch in tqdm(data_loader):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            _, output = model.extract_object_representation(images)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        for img_id, t_result, result in zip(image_ids, targets, output):
            print(t_result)
            print(result.get_field("labels").size(), result.get_field("attrs").size(), result.bbox.numpy().shape, result.get_field("attrs").size())
            d = {
                "image_id": img_id,
                "num_boxes": int(t_result.num_boxes),
                "image_h": int(t_result.image_height),
                "image_w": int(t_result.image_width)
                "labels": base64.b64encode(result.get_field("labels").numpy()),
                "attrs": base64.b64encode(result.get_field("attrs").numpy()),
                "bbox": base64.b64encode(result.bbox.numpy()),
                "feature": base64.b64encode(result.get_field("attrs").numpy())
            }
            results_dict.update(d)
    return results_dict
                

def extract(
        model,
        data_loader,
        dataset_name,
        output_folder,
        device="cuda"
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return


