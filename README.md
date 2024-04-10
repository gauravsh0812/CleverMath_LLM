# ClevrMath_LLM

Keep preprocess in the config "True" if running it for the first time. 
Once you have run it, there is no need to re-process the dataset. 
The "data/" folder should have the "raw_data" along with the preprocessed and re-arranged dataset i.e.
"images/", "questions.lst", "labels.lst", and "templates.lst". If you have these files (updated) then no need to 
re-do these steps. This can avoided by setting the preprocess param to False in the config file.

### Requirements 
```
conda create -n clevrmath python=3.10 -y
source activate clevrmath

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Dataset
If running it for the first time, then we need to donwload and preprocess the dataset.
```
python preprocessing/preprocess_raw_data.py
``` 
To get image tensors, make sure to set "get_image_tensors" True in config file.


### Training
First we need to get the masks and scores.
```
python models/create_masks.py
```
```
python run.py
```