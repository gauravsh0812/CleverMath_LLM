import os
import yaml
import random
import numpy as np
import multiprocessing as mp
from box import Box
import torch 
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from preprocessing.create_dataloaders import data_loaders
from models.unet import UNet


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

def define_model(vocab, device):

    # Image Auto-Encoder 
    UNET = UNet(
        Cin_UNet=cfg.training.encoder.input_channels, 
        Cout_UNet=len(vocab)
    )

    # Text Encoder



    pass

def train_model(rank=None):
    # set_random_seed
    set_random_seed(cfg.general.seed)
    
    # to save trained model and logs
    FOLDER = ["trained_models", "logs"]
    for f in FOLDER:
        if not os.path.exists(f):
            os.mkdir(f)

    # to log losses
    loss_file = open("logs/loss_file.txt", "w")
    
    # defining model using DataParallel
    if torch.cuda.is_available() and cfg.general.device == "cuda":
        if not cfg.general.ddp:
            print(f"using single gpu:{cfg.general.gpus}...")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.general.gpus)
            device = torch.device(f"cuda:{cfg.general.gpus}")
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                vocab,
            ) = data_loaders()
            model = define_model(vocab, device).to(device)

        elif cfg.general.ddp:
            # create default process group
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            # add rank to config
            cfg.general.rank = rank
            device = f"cuda:{rank}"
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                vocab,
            ) = data_loaders()
            model = define_model(vocab, rank)
            model = DDP(
                model.to(f"cuda:{rank}"),
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
            )

    else:
        import warnings

        warnings.warn("No GPU input has provided. Falling back to CPU. ")
        device = torch.device("cpu")
        (
            train_dataloader,
            test_dataloader,
            val_dataloader,
            vocab,
        ) = preprocess()
        model = define_model(vocab, device).to(device)

    print("MODEL: ")
    print(f"The model has {count_parameters(model)} trainable parameters")

    # intializing loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])


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