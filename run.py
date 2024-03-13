import os
import yaml
import torch 
import random
import numpy as np
import multiprocessing as mp
from box import Box
from preprocessing.preprocess_dataset import preprocess

with open("config/config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

def set_random_seed(SEED):
    # set up seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def count_parameters(model):
    """
    counting total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    """
    epoch timing
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_model(rank=None):
    pass
    
    

def ddp_main(world_size,):    
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    mp.spawn(train_model, args=(), nprocs=world_size, join=True)

if __name__ == "__main__":
    if cfg.general.ddp:
        gpus = cfg.general.gpus
        world_size = cfg.general.world_size
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29800"
        ddp_main(world_size)

    else:
        train_model()