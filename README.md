# CleverMath_LLM

Keep preprocess in the config "True" if running it for the first time. 
Once you have run it, there is no need to re-process the dataset. 
The "data/" folder should have the "raw_data" along with the preprocessed and re-arranged dataset i.e.
"images/", "questions.lst", "labels.lst", and "templates.lst". If you have these files (updated) then no need to 
re-do these steps. This can avoided by setting the preprocess param to False in the config file.

```
python run.py
```