from transformers import LogitsProcessor
import torch
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


class TopicLogitsProcessor(LogitsProcessor):
    def __init__(self, topic_word_vector, gamma, logit_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topic_word_vector = topic_word_vector
        self.gamma = gamma
        self.logit_threshold = logit_threshold

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        gamma = self.gamma
        LOGIT_THRESHOLD = self.logit_threshold
        #gamma = 1 #higher values of gamma corresponds to more on topic
        #LOGIT_THRESHOLD = -90 # smaller values of Threshold is more on topic
        logits = logits.squeeze(0)

        logscores = torch.log(self.topic_word_vector)
        indices = logits < LOGIT_THRESHOLD#todo cut logits relatively not absolutely


        #logscores[logscores == -float("Inf")] = 0
        logscores[indices] = logits[indices].double()

        total_logit = logits + gamma * logscores

        #total_logit[total_logit == -float("Inf")] = -1e10
        #total_probs = sparsemax(total_logit, dim=-1)
        ###

        ##entmax
        #total_probs = entmax15(total_logit, dim=-1)

        return total_logit.unsqueeze(0)
