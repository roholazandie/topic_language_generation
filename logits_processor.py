import torch

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from transformers import LogitsProcessorList, HammingDiversityLogitsProcessor, RepetitionPenaltyLogitsProcessor, \
    NoRepeatNGramLogitsProcessor, NoBadWordsLogitsProcessor, MinLengthLogitsProcessor, PrefixConstrainedLogitsProcessor, \
    ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor, InfNanRemoveLogitsProcessor
from transformers.generation_logits_process import EncoderNoRepeatNGramLogitsProcessor

from topic_logit_processor import TopicLogitsProcessor


def _get_logits_processor(
        model,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        encoder_no_repeat_ngram_size: int,
        encoder_input_ids: torch.LongTensor,
        bad_words_ids: List[List[int]],
        topic_word_vector: torch.FloatTensor,
        gamma: float,
        logit_threshold: float,
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        remove_invalid_values: bool,
) -> LogitsProcessorList:
    """
    This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
    :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
    """

    # init warp parameters
    repetition_penalty = repetition_penalty if repetition_penalty is not None else model.config.repetition_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else model.config.no_repeat_ngram_size
    )
    encoder_no_repeat_ngram_size = (
        encoder_no_repeat_ngram_size
        if encoder_no_repeat_ngram_size is not None
        else model.config.encoder_no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else model.config.bad_words_ids
    min_length = min_length if min_length is not None else model.config.min_length
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    diversity_penalty = diversity_penalty if diversity_penalty is not None else model.config.diversity_penalty
    forced_bos_token_id = (
        forced_bos_token_id if forced_bos_token_id is not None else model.config.forced_bos_token_id
    )
    forced_eos_token_id = (
        forced_eos_token_id if forced_eos_token_id is not None else model.config.forced_eos_token_id
    )
    remove_invalid_values = (
        remove_invalid_values if remove_invalid_values is not None else model.config.remove_invalid_values
    )
    # instantiate processors list
    processors = LogitsProcessorList()

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
        if model.config.is_encoder_decoder:
            processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))
        else:
            raise ValueError(
                "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
            )
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    if topic_word_vector is not None:
        processors.append(TopicLogitsProcessor(topic_word_vector, gamma, logit_threshold))
    if min_length is not None and eos_token_id is not None and min_length > -1:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if prefix_allowed_tokens_fn is not None:
        processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
    if forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
    if forced_eos_token_id is not None:
        processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
    if remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    return processors