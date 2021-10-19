from topical_tokenizers import TransformerGPT2Tokenizer
from datasets.dataset import Dataset
import os
import logging
logging.basicConfig(level=logging.DEBUG)

class NYTimesDataset(Dataset):

    def __init__(self, dirname, tokenizer, do_tokenize=True):
        super().__init__(dirname, tokenizer, do_tokenize)
        logging.debug("using nytimes dataset")
        self.dirname = dirname
        self.tokenizer = tokenizer
        self.do_tokenize = do_tokenize

    def _process_text(self, text):
        token_ids = self.tokenizer.tokenize(text)
        return token_ids

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for file in files:
                with open(root + "/" + file) as file_reader:
                    article = []
                    for line in file_reader:
                        if line.startswith("URL:"):
                            if article:
                                tokens = self._process_text(" ".join(article))
                                yield tokens
                                article = []
                        else:
                            if line.rstrip():
                                article.append(line.rstrip())


if __name__ == "__main__":
    filename = "/media/rohola/data/newyork_articles/"
    cached_dir = "/home/rohola/cached_models"
    tokenizer = TransformerGPT2Tokenizer(cached_dir)
    nytimes_dataset = NYTimesDataset(filename, tokenizer)

    for i, article in enumerate(nytimes_dataset):
        print(i, article)