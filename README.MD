# Topical Language Modeling

### Install
run the command:
```
pip install -r requirements.txt
```

### Dataset
For training the topic models you need to download the alexa reading set. can be downloaded from [here](https://github.com/alexa/Topical-Chat/tree/master/reading_sets/pre-build)
Then set the `dataset_dir` in `alexa_lsi_config.json` to the directory of reading set.
You also need to set `cached_dir` to a directory to save the trained topic models.

### Running
to run use the following:

```
python simple_generation.py
```