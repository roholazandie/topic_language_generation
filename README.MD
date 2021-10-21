# Topical Language Modeling

### Install
run the command:
```
pip install -r requirements.txt
```

### Dataset
For training the topic models you need to download the alexa reading set. can be downloaded from [here](https://drive.google.com/file/d/1YeeLRt0xedS774yLCF6kgRY9y9OhRA1q/view?usp=sharing)
Then set the `dataset_dir` in `alexa_lsi_config.json` to the directory of reading set.
You also need to set `cached_dir` to a directory to save the trained topic models.

### Running
to run use the following:

```
python simple_generation.py
```