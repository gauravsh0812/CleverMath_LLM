import os
import yaml
import torch
from box import Box
from datasets import load_dataset, DownloadConfig, DatasetDict
from PIL import Image
from torchvision import transforms
from multiprocessing import Pool

# reading config file
with open("config/config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

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

def preprocess_images(img):

    print("creating image tensors...")
    
    IMAGE = Image.open(f"data/images/{img}")
    
    # checking the size of the image
    w, h = IMAGE.size
    assert w == 480 and h == 320

    # convert to tensor
    convert = transforms.ToTensor()
    IMAGE = convert(IMAGE)

    # saving the image 
    torch.save(f"data/image_tensors/{img.split('.')[0]}.pt")


def getting_image_tensors():
    """
    we don't need to crop and pad the image 
    as they all are of same size i.e. 480,320.
    And it has a lot of wide spaces near the edges.
    """
    images = os.listdir("data/images")

    # create an image_tensors folder
    if not os.path.exists("data/image_tensors"):
        os.mkdir("data/image_tensors")

    with Pool(cfg.general.num_cpus) as pool:
        result = pool.map(preprocess_images, images) 

    blank_images = [i for i in result if i is not None]

    with open("logs/blank_images.lst", "w") as out:
        out.write("\n".join(str(item) for item in blank_images))

    
def preprocess():

    # only if preprocess is True
    # will re-download, and re-arrange the dataset.
    # need to be done only first time.
    dataset = download_dataset(name="general")
    rearrange_dataset(dataset)
    questions.close()
    labels.close()

    # converting images to tensors
    getting_image_tensors()

if __name__ == "__main__":    
    preprocess()