# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import sys
import csv, codecs

import torch
from tqdm import tqdm

from ..utils.comm import is_main_process, get_world_size, get_rank
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str

import base64

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_h', 'image_w', 'num_boxes', 'labels', 'attrs', 'bbox', 'feature']


def compute_on_dataset(output_folder, model, data_loader, device, timer=None):
    model.eval()
    cpu_device = torch.device("cpu")
    with codecs.open(os.path.join(output_folder, "result_%d.tsv" % get_rank()), 'w', encoding = 'utf8') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)
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
                writer.writerow({
                    "image_id": img_id,
                    "num_boxes": result.bbox.numpy().shape[0],
                    "image_h": int(t_result.size[1]),
                    "image_w": int(t_result.size[0]),
                    "labels": base64.b64encode(result.get_field("labels")).decode('utf-8'),
                    "attrs": base64.b64encode(result.get_field("attrs")).decode('utf-8'),
                    "bbox": base64.b64encode(result.bbox).decode('utf-8'),
                    "feature": base64.b64encode(result.get_field("features")).decode('utf-8')
                })   

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
    compute_on_dataset(output_folder, model, data_loader, device, inference_timer)
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

    if not is_main_process():
        return

    with codecs.open(os.path.join(output_folder, "result.tsv"), 'w', encoding = 'utf8') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)  
        for i in range(num_devices):
            csv_path = os.path.join(output_folder, "result_%d.tsv" % i)
            with open(csv_path) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    writer.writerow(item)
            os.remove(csv_path)



