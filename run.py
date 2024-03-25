import os
import yaml
import random
import time
import math
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
from models.cnn import CNN
from models.roberta import RobertaEncoder
from models.model import ClevrMath_model
from src.training import train
from src.testing import evaluate


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

def define_model(max_len):

    # Image Auto-Encoder 
    image_length = (cfg.dataset.image_width * cfg.dataset.image_height)
    dropout = cfg.training.general.dropout
    features = cfg.training.unet_encoder.features
    dim = features[0]
    unet = UNet(
        Cin_UNet=cfg.training.unet_encoder.input_channels, 
        features=features,
        dropout=dropout,
        image_length=image_length,
    )

    # CNN encoder 
    dim = 512
    cnn = CNN(input_channels=cfg.training.cnn_encoder.input_channels, 
              dec_hid_dim=cfg.training.cnn_encoder.hid_dim,
              dropout=dropout,
              image_length=image_length)

    # Text Encoder
    roberta = RobertaEncoder()

    encoder = cfg.training.model_type.encoder
    decoder = cfg.training.model_type.decoder

    model = ClevrMath_model(encoder, 
                            decoder,
                            dim,
                            image_length,
                            max_len,
                            num_classes=11)

    return model

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
                max_len,
            ) = data_loaders()
            model = define_model(max_len).to(device)

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
                max_len,
            ) = data_loaders()
            model = define_model(max_len)
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
            max_len,
        ) = data_loaders()
        model = define_model(max_len).to(device)

    print("MODEL: ")
    print(f"The model has {count_parameters(model)} trainable parameters")

    # intializing loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    # optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.training.general.learning_rate,
        weight_decay=cfg.training.general.weight_decay,
        betas=cfg.training.general.betas,
    )

    best_valid_loss = float("inf")
    
    if not cfg.general.load_trained_model_for_testing:
        count_es = 0
        for epoch in range(cfg.training.general.epochs):
            if count_es <= cfg.training.general.early_stopping:
                start_time = time.time()

                # training and validation
                train_loss = train(
                    model,
                    cfg.dataset.path_to_data, 
                    train_dataloader,
                    optimizer,
                    criterion,
                    cfg.training.general.clip,
                    device,
                    ddp=cfg.general.ddp,
                    rank=rank,
                )

                val_loss,_ = evaluate(
                    model,
                    cfg.dataset.path_to_data,
                    val_dataloader,
                    criterion,
                    device,
                )

                end_time = time.time()
                # total time spent on training an epoch
                
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                
                # saving the current model for transfer learning
                if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0):
                    torch.save(
                        model.state_dict(),
                        f"trained_models/latest_model.pt",
                    )

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    count_es = 0
                    if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0):
                        torch.save(
                            model.state_dict(),
                            f"trained_models/best_model.pt",
                        )
                else:
                    count_es += 1

                # logging
                if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0):
                    print(
                        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s"
                    )
                    print(
                        f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
                    )
                    print(
                        f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}"
                    )

                    loss_file.write(
                        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n"
                    )
                    loss_file.write(
                        f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n"
                    )
                    loss_file.write(
                        f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}\n"
                    )

            else:
                print(
                    f"Terminating the training process as the validation \
                    loss hasn't been reduced from last {cfg.training.general.early_stopping} epochs."
                )
                break

        print(
            "best model saved as:  ",
            f"trained_models/best_model.pt",
        )

    if cfg.general.ddp:
        dist.destroy_process_group()

    time.sleep(3)

    print(
        "loading best saved model: ",
        f"trained_models/best_model.pt",
    )
    # loading pre_tained_model
    model.load_state_dict(
        torch.load(
            f"trained_models/best_model.pt"
        )
    )

    test_loss, accuracy = evaluate(
        model,
        cfg.dataset.path_to_data,
        test_dataloader,
        criterion,
        device,
        is_test=True,
    )

    if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0):
        print(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test Accuracy: {accuracy: .3f}"
        )
        loss_file.write(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test Accuracy: {accuracy: .3f}"
        )

    # stopping time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


def ddp_main(world_size,):    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
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