import os
import yaml
import random
import time
import math
import wandb
import numpy as np
import multiprocessing as mp
from box import Box
import torch 
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from preprocessing.create_dataloaders import data_loaders
from models.clip import ClipVisionEncoder
from models.roberta import RobertaEncoder
from models.model import ClevrMath_model
from models.adaptor import Adaptor
from src.training import train
from src.testing import evaluate
import optuna
from optuna.trial import TrialState


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
    
    ENC = ClipVisionEncoder(finetune=cfg.training.clip.finetune,
                            config=cfg.training.clip.configuration)
    DEC = RobertaEncoder()    
    ADA = Adaptor(cfg.training.clip.configuration.hidden_size, 
                  cfg.training.roberta.in_dim,
                  cfg.training.adaptor.features,
                  max_len,
                  num_classes=cfg.training.general.num_classes,)

    # freezing the pre-trained models
    # only training the adaptor layer
    for param in ENC.parameters():
        param.requires_grad = cfg.training.clip.finetune

    for param in DEC.parameters():
        param.requires_grad = cfg.training.roberta.finetune

    for param in ADA.parameters():
        param.requires_grad = cfg.training.adaptor.finetune   

    model = ClevrMath_model(ENC, 
                            DEC,
                            ADA,)

    return model

def objective(trial):
    
    tcfg = cfg.training
    ccfg = tcfg.clip.configuration

    # parameters
    tcfg.general.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    tcfg.general.weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    tcfg.general.batch_size = trial.suggest_int("batch_size", low=8, high=32, step=4)
    tcfg.general.dropout = trial.suggest_float("dropout", low=0.1, high=0.5, step=0.1)
    tcfg.general.beta_1 = trial.suggest_float("beta1", low=0.5, high=0.9, step=0.1)
    tcfg.general.beta_2 = trial.suggest_float("beta2", low=0.5, high=0.999, step=0.1)
    
    ccfg.hidden_size = trial.suggest_int("hidden_size", low=128, high=512, step=128)
    ccfg.intermediate_size = trial.suggest_int("intermediate_size", low=128, high=512, step=128)
    ccfg.projection_dim = trial.suggest_int("projection_dim", low=64, high=512, step=64)
    ccfg.num_hidden_layers = trial.suggest_int("num_hidden_layers", low=3, high=12, step=2)
    ccfg.num_attention_heads = trial.suggest_categorical("num_attention_heads", [2,4,8,16])

    # set_random_seed
    set_random_seed(cfg.general.seed)
    
    # defining model using DataParallel
    if torch.cuda.is_available() and cfg.general.device == "cuda":
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

    # intializing loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.training.general.learning_rate,
        weight_decay=cfg.training.general.weight_decay,
        betas=cfg.training.general.betas,
    )

    print("trial: ", trial.params.items())

    for epoch in range(cfg.training.general.epochs):
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
            rank=0,
        )

    val_loss, accuracy = evaluate(
    model,
    cfg.dataset.path_to_data,
    val_dataloader,
    criterion,
    device,
    )

    trial.report(val_loss, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
        
    return val_loss

def tune():

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE]
    )

    # if config["DDP"] and rank==0:
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    tune()