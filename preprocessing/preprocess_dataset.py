import os
import yaml
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import SequentialSampler
from box import Box
from datasets import load_dataset, DownloadConfig, DatasetDict

# reading config file
with open("config/config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

# opening files in "w" mode
questions = open("data/questions.lst","w")
labels = open("data/labels.lst", "w")
templates = open("data/templates.lst", "w")


def download_dataset(name):
    """
    downloading dataset from hugginface
    """

    """
    an example of dataset sample:
    {
        'template': 'adversarial', 
        'id': 'CLEVR_val_000000.png', 
        'question': 'Subtract all gray cubes. How many red cylinders are left?', 
        'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=480x320 at 0x7FB1A10167C0>, 
        'label': 0
    }
    """

    print("downloading the dataset if doesn't exists...")

    raw_data_path = os.path.join(
        cfg.dataset.data_path,
        "raw_data"
    )

    dl_config = DownloadConfig(
        resume_download=True, 
        num_proc=cfg.general.ncpus,
        force_download=True
    )
    
    dataset_train = load_dataset(
        'dali-does/clevr-math',
        name=name,
        download_config=dl_config,
        split='train[:{}]'.format(1000),
        cache_dir=raw_data_path
    )
    
    dataset_val = load_dataset(
        'dali-does/clevr-math',
        name=name,
        download_config=dl_config,
        split='validation[:{}]'.format(200),
        cache_dir=raw_data_path
    )
    
    dataset_test = load_dataset(
        'dali-does/clevr-math',
        name=name,
        download_config=dl_config,
        split='test[:{}]'.format(200),
        cache_dir=raw_data_path
    )

    dataset = DatasetDict({
      'train':dataset_train,
      'validation':dataset_val,
      'test':dataset_test
    })

    dataset['train'] = dataset['train'].select(range(1000))
    dataset['validation'] = dataset['validation'].select(range(200))
    dataset['test'] = dataset['test'].select(range(200))

    return dataset

def rearrange_dataset(dataset):
    """
    re-arranging the raw dataset 
    to make it easier to access.
    * images
    * questions.lst
    * labels.lst
    * templates.lst
    All data will be in sequentially arranged
    i.e. Q1 - Label1 - Template 1 - image 0.png (indexes at 0)
    """

    print("re-arranging the dataset...")
    count = 0
    for t in ["train", "test", "validation"]:
        for t_data in dataset[t]: 
            # copying the images
            t_data["image"].save(f"data/images/{count}.png")
            count+=1

            # writing the corresponding questions and labels
            questions.write(f"{t_data['question']} \n")
            labels.write(f"{t_data['label']} \n")
            templates.write(f"{t_data['template']} \n")

class Img2MML_dataset(Dataset):
    def __init__(self, dataframe, vocab, tokenizer):
        self.dataframe = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        eqn = self.dataframe.iloc[index, 1]
        indexed_eqn = []
        for token in eqn.split():
            if self.vocab.stoi[token] is not None:
                indexed_eqn.append(self.vocab.stoi[token])
            else:
                indexed_eqn.append(self.vocab.stoi["<unk>"])

        return self.dataframe.iloc[index, 0], torch.Tensor(indexed_eqn)

class My_pad_collate(object):
    """
    padding mml to max_len, and stacking images
    return: mml_tensors of shape [batch, max_len]
            stacked image_tensors [batch]
    """

    def __init__(self, device, vocab, max_len):
        self.device = device
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab.stoi["<pad>"]

    def __call__(self, batch):
        _img, _mml = zip(*batch)

        # padding mml
        # padding to a fix max_len equations with more tokens than
        # max_len will be chopped down to max_length.

        batch_size = len(_mml)
        padded_mml_tensors = (
            torch.ones([batch_size, self.max_len], dtype=torch.long)
            * self.pad_idx
        )
        for b in range(batch_size):
            if len(_mml[b]) <= self.max_len:
                padded_mml_tensors[b][: len(_mml[b])] = _mml[b]
            else:
                padded_mml_tensors[b][: self.max_len] = _mml[b][: self.max_len]

        # images tensors
        _img = torch.Tensor(_img)

        return (
            _img.to(self.device),
            padded_mml_tensors.to(self.device),
        )

def create_dataloaders():

    print("creating dataloaders...")
    q = open("data/questions.lst").readlines()
    l = open("data/labels.lst").readlines()
    t = open("data/templates.lst").readlines()

    assert len(q) == len(l) == len(t)

    image_num = range(0, len(q))

    

if __name__ == "__main__":
    paths = [
        "data/raw_data",
        "data/images"
    ]
    for _path in paths:
        if not os.path.exists(_path):
            os.mkdir(_path)
    
    dataset = download_dataset(name="general")
    rearrange_dataset(dataset)
    questions.close()
    labels.close()

    # dataset_multihop = dataset.filter(
    #     lambda e:
    #     e['template'].startswith('subtraction-multihop'), 
    #     num_proc=4
    # )

    # dataset_adversarial = dataset.filter(
    #     lambda e:
    #     e['template'].startswith('adversarial'), 
    #     num_proc=4
    # )
    # dataset_subtraction = dataset.filter(
    #     lambda e:
    #     e['template'].startswith('subtraction'), 
    #     num_proc=4
    # )
    # dataset_addition = dataset.filter(
    #     lambda e:
    #     e['template'].startswith('addition'), 
    #     num_proc=4
    # )

    create_dataloaders()