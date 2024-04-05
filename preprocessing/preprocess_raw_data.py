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
        f"{cfg.dataset.path_to_data}",
        f"{cfg.dataset.path_to_data}/raw_data",
        f"{cfg.dataset.path_to_data}/images"
    ]
for _path in paths:
    if not os.path.exists(_path):
        os.mkdir(_path)


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

    raw_data_path = f"{cfg.dataset.path_to_data}/raw_data"

    dl_config = DownloadConfig(
        resume_download=True, 
        num_proc=cfg.general.ncpus,
        force_download=True
    )
    
    dataset_train = load_dataset(
        'dali-does/clevr-math',
        name=name,
        download_config=dl_config,
        split='train[:{}]'.format(-1),
        cache_dir=raw_data_path
    )
    
    dataset_val = load_dataset(
        'dali-does/clevr-math',
        name=name,
        download_config=dl_config,
        split='validation[:{}]'.format(-1),
        cache_dir=raw_data_path
    )
    
    dataset_test = load_dataset(
        'dali-does/clevr-math',
        name=name,
        download_config=dl_config,
        split='test[:{}]'.format(-1),
        cache_dir=raw_data_path
    )

    dataset = DatasetDict({
      'train':dataset_train,
      'validation':dataset_val,
      'test':dataset_test
    })

    dataset['train'] = dataset['train'].select(range(10000))
    dataset['validation'] = dataset['validation'].select(range(2000))
    dataset['test'] = dataset['test'].select(range(3000))

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
    qtns, lbls, tmps = list(),list(),list()

    # opening files in "w" mode
    questions = open(f"{cfg.dataset.path_to_data}/questions.lst","w")
    labels = open(f"{cfg.dataset.path_to_data}/labels.lst", "w")
    templates = open(f"{cfg.dataset.path_to_data}/templates.lst", "w")
    for t in ["train", "test", "validation"]:
        for t_data in dataset[t]: 
            # copying the images
            t_data["image"].save(f"{cfg.dataset.path_to_data}/images/{count}.png")
            count+=1

            # writing the corresponding questions and labels
            # the labels ranges from 0-10
            # the templates are: {addition, adverserail, subtraction, subtraction-multihop}

            questions.write(t_data['question'] + "\n")
            labels.write(str(t_data['label']) + "\n")
            templates.write(t_data['template'] + "\n")

    questions.close()
    labels.close()
    templates.close()

def preprocess_images(img):
    
    IMAGE = Image.open(f"{cfg.dataset.path_to_data}/images/{img}")
    
    # checking the size of the image
    w, h = IMAGE.size

    # Desired image size
    desired_width = 480
    desired_height = 320

    if w != 480 or h != 320:

        # Original image size
        original_width, original_height = IMAGE.size
        
        # Calculate aspect ratio of the original image
        aspect_ratio = original_width / original_height

        # Calculate the new dimensions while maintaining the aspect ratio
        new_width = desired_width
        new_height = int(new_width / aspect_ratio)

        # Calculate the padding needed to achieve the desired dimensions
        padding_height = desired_height - new_height
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top

        # Create a new image with the desired dimensions
        # the images are 4D - RGB and Alpha (could be transparency)
        new_image = Image.new('RGBA', (desired_width, desired_height), (255, 255, 255, 255)) # White background

        # Paste the original image onto the new image, centered and padded as needed
        new_image.paste(IMAGE, (0, padding_top))

        # Save the new image
        new_image.save(f'{cfg.dataset.path_to_data}/padded_images/{img}')

        IMAGE = new_image

        # Create an attention mask image
        mask_image = Image.new('L', (desired_width, desired_height), 0) # 'L' mode for grayscale

        # Fill the original content area with 1
        mask_image.paste(1, (0, padding_top, desired_width, new_height + padding_top))

        # Save the mask image
        mask_image.save(f'{cfg.dataset.path_to_data}/attention_masks/{img}')


    # convert to tensor
    convert = transforms.ToTensor()
    IMAGE = convert(IMAGE)

    # saving the image 
    torch.save(IMAGE, f"{cfg.dataset.path_to_data}/image_tensors/{img.split('.')[0]}.pt")


def getting_image_tensors():
    """
    we don't need to crop the image but we will 
    pad them to be of the same size i.e. w=480, h=320.
    And it has a lot of wide spaces near the edges.
    """

    print("creating image tensors...")

    images = os.listdir(f"{cfg.dataset.path_to_data}/images")

    # create an image_tensors folder
    if not os.path.exists(f"{cfg.dataset.path_to_data}/image_tensors"):
        os.mkdir(f"{cfg.dataset.path_to_data}/image_tensors")

    # to store padded images for reference
    # and to store the corresponding attention mask
    if not os.path.exists(f"{cfg.dataset.path_to_data}/padded_images"):
        os.mkdir(f"{cfg.dataset.path_to_data}/padded_images")
    if not os.path.exists(f"{cfg.dataset.path_to_data}/attention_masks"):
        os.mkdir(f"{cfg.dataset.path_to_data}/attention_masks")
    

    with Pool(cfg.general.ncpus) as pool:
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
    # questions.close()
    # labels.close()

    # converting images to tensors
    if cfg.dataset.get_image_tensors:
        getting_image_tensors()

if __name__ == "__main__":    
    preprocess()