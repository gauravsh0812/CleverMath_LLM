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

if cfg.general.preprocess:
    # if running for the first time.
    paths = [
            "data",
            "data/raw_data",
            "data/images"
        ]
    for _path in paths:
        if not os.path.exists(_path):
            os.mkdir(_path)

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

    raw_data_path = "data/raw_data"

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
            # the labels ranges from 0-10
            # the templates are: {addition, adverserail, subtraction, subtraction-multihop}
            questions.write(f"{t_data['question']} \n")
            labels.write(f"{t_data['label']} \n")
            templates.write(f"{t_data['template']} \n")

    


class Img2MML_dataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.dataframe = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        qtn = self.dataframe.iloc[index, 1]
        indexed_qtn = []
        for token in qtn.split():
            if self.vocab.stoi[token] is not None:
                indexed_qtn.append(self.vocab.stoi[token])
            else:
                indexed_qtn.append(self.vocab.stoi["<unk>"])

        return self.dataframe.iloc[index, 0], torch.Tensor(indexed_qtn)

# class My_pad_collate(object):
#     """
#     padding mml to max_len, and stacking images
#     return: mml_tensors of shape [batch, max_len]
#             stacked image_tensors [batch]
#     """

#     def __init__(self, device, vocab, max_len):
#         self.device = device
#         self.vocab = vocab
#         self.max_len = max_len
#         self.pad_idx = vocab.stoi["<pad>"]

#     def __call__(self, batch):
#         _img, _q, _l = zip(*batch)
#         batch_size = len(_q)
#         padded_questions = (
#             torch.ones([batch_size, self.max_len], dtype=torch.long)
#             * self.pad_idx
#         )
#         for b in range(batch_size):
#             assert len(_q[b]) <= self.max_len
#             padded_questions[b][: len(_q[b])] = _q[b]
    
#         # images tensors
#         _img = torch.Tensor(_img)
#         _l = torch.Tensor(_l, dtype=torch.long)

#         return (
#             _img.to(self.device),
#             padded_questions.to(self.device),

#         )

def tokenizer(x):
    return x.split()
    
def create_dataloaders():

    print("creating dataloaders...")
    q = open("data/questions.lst").readlines()
    l = open("data/labels.lst").readlines()
    t = open("data/templates.lst").readlines()

    assert len(q) == len(l) == len(t)

    max_len = max([len(tokenizer(i)) for i in q])

    image_num = range(0, len(q))

    # split the image_num into train, test, validate
    train_val_images, test_images = train_test_split(
        image_num, test_size=0.1, random_state=42
    )
    train_images, val_images = train_test_split(
        train_val_images, test_size=0.1, random_state=42
    )

    for t_idx, t_images in enumerate([train_images, test_images, val_images]):
        qi_data = {
            "IMG": [num for num in t_images],
            "QUESTION": [
                ("<sos> " + q[num] + " <eos>") for num in t_images
            ],
            "LABEL": [l[num] for num in t_images],
        }
    
        if t_idx == 0:
            train = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "LABEL"])
        elif t_idx == 1:
            test = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "LABEL"])
        else:
            val = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "LABEL"])
    
    
    print("saving dataset files to data/ folder...")
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    
    # build vocab
    print("building vocab...")

    counter = Counter()
    for line in train["QUESTION"]:
        counter.update(line.split())

    # <unk>, <pad> will be prepended in the vocab file
    vocab = Vocab(
        counter,
        min_freq=cfg.dataset.vocab_freq,
        specials=["<pad>", "<unk>", "<sos>", "<eos>"],
    )

    # writing vocab file...
    vfile = open("data/vocab.txt", "w")
    for vidx, vstr in vocab.stoi.items():
        vfile.write(f"{vidx} \t {vstr} \n")
    
    # initializing pad collate class
    # mypadcollate = My_pad_collate(cfg.general.device, 
    #                               vocab, 
    #                               max_len)

    print("building dataloaders...")

    # initailizing class Img2MML_dataset: train dataloader
    imml_train = Img2MML_dataset(train, vocab)
    # creating dataloader
    if cfg.general.ddp:
        train_sampler = DistributedSampler(
            dataset=imml_train,
            num_replicas=cfg.general.world_size,
            rank=cfg.general.rank,
            shuffle=cfg.dataset.shuffle,
        )
        sampler = train_sampler
        shuffle = False

    else:
        sampler = None
        shuffle = cfg.dataset.shuffle
        
    train_dataloader = DataLoader(
        imml_train,
        batch_size=cfg.training.general.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        # collate_fn=mypadcollate,
        pin_memory=cfg.dataset.pin_memory,
    )

    # initailizing class Img2MML_dataset: val dataloader
    imml_val = Img2MML_dataset(val, vocab)

    if cfg.general.ddp:
        val_sampler = SequentialSampler(imml_val)
        sampler = val_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = cfg.dataset.shuffle

    val_dataloader = DataLoader(
        imml_val,
        batch_size=cfg.training.general.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        # collate_fn=mypadcollate,
        pin_memory=cfg.dataset.pin_memory,
    )

    # initailizing class Img2MML_dataset: test dataloader
    imml_test = Img2MML_dataset(test, vocab)
    if cfg.general.ddp:
        test_sampler = SequentialSampler(imml_test)
        sampler = test_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = cfg.dataset.shuffle

    test_dataloader = DataLoader(
        imml_test,
        batch_size=cfg.training.general.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        sampler=None,
        # collate_fn=mypadcollate,
        pin_memory=cfg.dataset.pin_memory,
    )

    return train_dataloader, test_dataloader, val_dataloader, vocab


def data_loaders():

    if cfg.general.preprocess:
        # only if preprocess is True
        # will re-download, and re-arrange the dataset.
        # need to be done only first time.
        dataset = download_dataset(name="general")
        rearrange_dataset(dataset)
        questions.close()
        labels.close()

        # generate the image_tensors to fasten the process
        get_image_tensors()

    return create_dataloaders()

if __name__ == "__main__":    
    data_loaders()