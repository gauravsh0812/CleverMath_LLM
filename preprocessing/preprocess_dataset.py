import os
import yaml
from box import Box
from datasets import load_dataset, DownloadConfig

with open("conifg/config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

def download_dataset():

    dl_config = DownloadConfig(
        resume_download=True,
        num_proc=cfg.general.ncpus,
        force_download=True
    )
    
    # Load 'general' instance of dataset
    dataset = load_dataset(
        'dali-does/clevr-math', 
        download_config=dl_config
    )

    # Load version with only multihop in test data
    dataset_multihop = load_dataset(
        'dali-does/clevr-math', 
        'multihop',
        download_config=dl_config
    )    

def preprocess():
    if cfg.dataset.download_for_first_time:
        print("Downloading the Clevr-Math dataset for the first time.")
        download_dataset()
        print("The dataset has downloaded at ~/.cache/huggingface/datasets. \
              Moving the dataset to the config defined data_path.")
        cmd = f"mv ~/.cache/huggingface/datasets/* {cfg.dataset.data_path}"
        os.system(cmd)
    
    
    

