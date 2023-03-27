import numpy as np
import os
import copy
import functools
import datetime
import torch as th
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from utils import dist_util, logger
from utils.video_datasets import load_data
from utils.resample import create_named_schedule_sampler
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from utils.resample import LossAwareSampler, UniformSampler
from utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)

from model import Model
import config

INITIAL_LOG_LOSS_SCALE = 20.0
dist_util.setup_dist()
if __name__ == "__main__":
    # Read the YAML file
    file_path = 'model_config.yml'
    logger.log("creating data loader...")
    load_data_config, model_config, Diffusion_config = config.read_yaml(file_path)
    #print(load_data_config.data_dir)
    data = load_data(data_dir = load_data_config.data_dir,
                     batch_size = load_data_config.batch_size,
                     image_size = load_data_config.image_size,
                     num_workers= load_data_config.num_workers,
                     rgb = load_data_config.rgb,
                     seq_len = load_data_config.seq_len)
    logger.log("creating model and diffusion...")


    #RAMVID(file_path, data).run()
    lr = model_config.lr
    current_lr = lr
    step = 0
    while (
            current_lr
    ):
        ####### One
        batch = next(data)
        # print(batch.shape)
        Model(file_path).run_step(batch)
        if step % model_config.log_interval == 0:
            logger.dumpkvs()
        if step % model_config.save_interval == 0:
            Model(file_path).save()
            # Run for a finite amount of time in integration tests. Does access an environment variable
            if os.environ.get("DIFFUSION_TRAINING_TEST", "") and step > 0:
                exit()
        step += 1
        # Save the last checkpoint if it wasn't already saved.
    if (step - 1) % model_config.save_interval != 0:
        Model(file_path).save()
