from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import random
import warnings
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

import math

def _collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class DataCollatorForEMLM(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling that masks entire words.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    .. note::
        This collator relies on details of the implementation of subword tokenization by
        :class:`~transformers.BertTokenizer`, specifically that subword tokens are prefixed with `##`. For tokenizers
        that do not adhere to this scheme, this collator will produce an output that is roughly equivalent to
        :class:`.DataCollatorForLanguageModeling`.
    """

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]


        batch_input = _collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in list(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            mask_labels.append(self._whole_word_mask(ref_tokens))
        
        batch_mask = _collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers."
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        cand_tokens = []

        num_emowords = 0

        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
                cand_tokens[-1] = cand_tokens[-1] + token[2:]
            else:
                cand_indexes.append([i])
                cand_tokens.append(token)

        zipped = list(zip(cand_indexes, cand_tokens))
        
        random.shuffle(zipped)

        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))

        covered_indexes = set()

        num_emo_words = 0
        zipped_emoword_locations = set()

        for i, elem in enumerate(zipped):
          if elem[1] in self.emotion_set:
            zipped_emoword_locations.add(i)
            num_emo_words += 1

        random_numbers = [random.random() for _ in range(len(zipped))]

        if len(zipped) != num_emo_words:
          non_emoword_probability = max(0, len(zipped) * 0.15 - num_emo_words * self.k) / (len(zipped) - num_emo_words)

        num_covered = 0
        for i in range(len(random_numbers)):
          if num_covered > num_to_predict:
            break

          p = self.k
          if i not in zipped_emoword_locations:
            p = non_emoword_probability

          if random_numbers[i] < p:
            for idx in zipped[i][0]:
              num_covered += 1
              covered_indexes.add(idx)

        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]

        return mask_labels

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
