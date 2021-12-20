# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Load SQuAD dataset. """

from __future__ import absolute_import, division, print_function
from typing import List, Dict
import logging
import math
import collections
import re
from torch import nn
import torch
import numpy as np
from nltk import tokenize
from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
from transformers import BertModel

logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 sup_ids=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.sup_ids = sup_ids
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.sup_ids:
            s += ", sup_ids: %d" % (self.sup_ids)
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 sup_ids=None,
                 start_position=None,
                 
                 
                 
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.sup_ids = sup_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

class NERSample:
    # def __init__(self, sample_id: str, sentence: List, sentence_labels: List,  labels: List, ner_labels: List):
    def __init__(self, sample_id: str, sentence: List, sentence_labels: List, labels: List, ner_labels: List, mode: str):
        self.sample_id = sample_id
        self.sentence = sentence
        self.sentence_labels = sentence_labels
        self.labels = labels
        self.ner_labels = ner_labels
        self.mode = mode

class NERInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 labels,
                 words,
                 tokens,
                 annotated_tokens,
                 ner_labels,
                 label_map,
                 inv_label_map):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.words = words
        self.tokens = tokens
        self.annotated_tokens = annotated_tokens
        self.ner_labels = ner_labels
        self.label_map = label_map
        self.inv_label_map = inv_label_map

class SCDataset(object):
    def __init__(self, texts, tags, ner_labels, preprocessor, preprocess, tokenizer):

        self.texts = texts
        self.tags = tags
        self.ner_labels = ner_labels
        self.label_map = {label: i for i, label in enumerate(self.ner_labels)}
        self.inv_label_map = {i: label for i, label in enumerate(self.ner_labels)}
        self.preprocessor = preprocessor
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        self.preprocess = preprocess
        self.TOKENIZER = tokenizer
        self.MAX_SEQ_LEN = 256
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

    def __getitem__(self, item):
        textlist = self.texts[item]
        tags = self.tags[item]
        tokens = []
        annotated_tokens = []
        label_ids = []
        for indx, (word, label) in enumerate(zip(textlist, tags)):
            if self.preprocess:
                if word == '[MASK]':
                    clean_word = word
                else:
                    clean_word = self.preprocessor.preprocess(word)
                word_tokens = self.TOKENIZER.tokenize(clean_word)
                annotated_word_tokens = [f'{token}_{indx}_{label}' if i == 0 else f'{token}_{indx}_-100' for i, token in enumerate(word_tokens)]
            else:
                word_tokens = self.TOKENIZER.tokenize(word)
                annotated_word_tokens = [f'{token}_{indx}_{label}' if i == 0 else f'{token}_{indx}_-100' for i, token in enumerate(word_tokens)]
                # ignore words that are preprocessed because the preprocessor return '' and the tokeniser replace that with empty list which gets ignored here
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                annotated_tokens.extend(annotated_word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([self.label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = self.TOKENIZER.num_special_tokens_to_add()
        if len(tokens) > self.MAX_SEQ_LEN - special_tokens_count:
            tokens = tokens[: (self.MAX_SEQ_LEN - special_tokens_count)]
            annotated_tokens = annotated_tokens[: (self.MAX_SEQ_LEN - special_tokens_count)]
            label_ids = label_ids[: (self.MAX_SEQ_LEN - special_tokens_count)]

        # Add the [SEP] token
        tokens += [self.TOKENIZER.sep_token]
        annotated_tokens += [self.TOKENIZER.sep_token]
        label_ids += [self.pad_token_label_id]
        token_type_ids = [0] * len(tokens)

        # Add the [CLS] TOKEN
        tokens = [self.TOKENIZER.cls_token] + tokens
        annotated_tokens = [self.TOKENIZER.cls_token] + annotated_tokens
        label_ids = [self.pad_token_label_id] + label_ids
        token_type_ids = [0] + token_type_ids

        input_ids = self.TOKENIZER.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.MAX_SEQ_LEN - len(input_ids)

        input_ids += [self.TOKENIZER.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [self.pad_token_label_id] * padding_length

        assert len(input_ids) == self.MAX_SEQ_LEN
        assert len(attention_mask) == self.MAX_SEQ_LEN
        assert len(token_type_ids) == self.MAX_SEQ_LEN
        assert len(label_ids) == self.MAX_SEQ_LEN


        return NERInputFeatures(torch.tensor(input_ids, dtype=torch.long)[None, :],
                         torch.tensor(attention_mask, dtype=torch.long)[None, :],
                         torch.tensor(token_type_ids, dtype=torch.long)[None, :],
                         torch.tensor(label_ids, dtype=torch.long)[None, :],
                         np.array(textlist),
                         np.array(tokens),
                         np.array(annotated_tokens),
                         self.ner_labels,
                         self.label_map,
                         self.inv_label_map)


class SecondSCDataset:
    def __init__(self, texts, tags, ner_labels,  preprocessor, preprocess, tokenizer):

        self.texts = texts
        self.tags = tags
        self.ner_labels = ner_labels
        self.label_map = {label: i for i, label in enumerate(self.ner_labels)}
        self.inv_label_map = {i: label for i, label in enumerate(self.ner_labels)}
        self.preprocessor = preprocessor
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        self.preprocess = preprocess
        self.TOKENIZER = tokenizer
        self.MAX_SEQ_LEN = 256

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        textlist = self.texts[item]
        tags = self.tags[item]

        tokens = []
        annotated_tokens = []
        label_ids = []
        for indx, (word, label) in enumerate(zip(textlist, tags)):
            if self.preprocess:
                if word == '[MASK]':
                    clean_word = word
                else:
                    clean_word = self.preprocessor.preprocess(word)
                    word_tokens = self.TOKENIZER.tokenize(clean_word)
                    if len(word_tokens) > 1:
                        annotated_word_tokens = [f'{token}_{indx}_{label}' if i == 1 else f'{token}_{indx}_-100' for
                                             i, token in enumerate(word_tokens)]
                    else:
                        annotated_word_tokens = [f'{token}_{indx}_{label}' if i == 0 else f'{token}_{indx}_-100' for
                                                 i, token in enumerate(word_tokens)]

            else:
                word_tokens = self.TOKENIZER.tokenize(word)
                annotated_word_tokens = [f'{token}_{indx}_{label}' if i == 1 else f'{token}_{indx}_-100' for i, token in
                                         enumerate(word_tokens)]
                # ignore words that are preprocessed because the preprocessor return '' and the tokeniser replace that with empty list which gets ignored here
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                annotated_tokens.extend(annotated_word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                if len(word_tokens) > 1:
                    label_ids.extend([self.pad_token_label_id] + [self.label_map[label]] + [self.pad_token_label_id] * (
                                len(word_tokens) - 2))
                else:
                    label_ids.extend([self.label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = self.TOKENIZER.num_special_tokens_to_add()
        if len(tokens) > self.MAX_SEQ_LEN - special_tokens_count:
            tokens = tokens[: (self.MAX_SEQ_LEN - special_tokens_count)]
            annotated_tokens = annotated_tokens[: (self.MAX_SEQ_LEN - special_tokens_count)]
            label_ids = label_ids[: (self.MAX_SEQ_LEN - special_tokens_count)]

        # Add the [SEP] token
        tokens += [self.TOKENIZER.sep_token]
        annotated_tokens += [self.TOKENIZER.sep_token]
        label_ids += [self.pad_token_label_id]
        token_type_ids = [0] * len(tokens)

        # Add the [CLS] TOKEN
        tokens = [self.TOKENIZER.cls_token] + tokens
        annotated_tokens = [self.TOKENIZER.cls_token] + annotated_tokens
        label_ids = [self.pad_token_label_id] + label_ids
        token_type_ids = [0] + token_type_ids

        input_ids = self.TOKENIZER.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.MAX_SEQ_LEN - len(input_ids)

        input_ids += [self.TOKENIZER.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [self.pad_token_label_id] * padding_length

        assert len(input_ids) == self.MAX_SEQ_LEN
        assert len(attention_mask) == self.MAX_SEQ_LEN
        assert len(token_type_ids) == self.MAX_SEQ_LEN
        assert len(label_ids) == self.MAX_SEQ_LEN

        return NERInputFeatures(torch.tensor(input_ids, dtype=torch.long)[None, :],
                                torch.tensor(attention_mask, dtype=torch.long)[None, :],
                                torch.tensor(token_type_ids, dtype=torch.long)[None, :],
                                torch.tensor(label_ids, dtype=torch.long)[None, :],
                                np.array(textlist),
                                np.array(tokens),
                                np.array(annotated_tokens),
                                self.ner_labels,
                                self.label_map,
                                self.inv_label_map)

class SCModel(nn.Module):
    def __init__(self, num_tag, path):
        super(SCModel, self).__init__()
        self.num_tag = num_tag
        self.bert = BertModel.from_pretrained(path, output_attentions=True, output_hidden_states=True)
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(self.bert.config.hidden_size, self.num_tag)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           output_attentions=True, output_hidden_states=True)
        bo_tag = self.bert_drop(output['last_hidden_state'])
        logits = self.out_tag(bo_tag)
        return logits

def align_predictions(predictions, label_ids, inv_label_map):
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(inv_label_map[label_ids[i][j]])
                # preds_list[i].append(f"{inv_label_map[preds[i][j]]}: {max(predictions[i, j]):.2f}")
                # preds_list[i].append((inv_label_map[preds[i][j]], str(max(predictions[i, j]))))
                preds_list[i].append((inv_label_map[preds[i][j]], max(predictions[i, j].tolist())))
    return out_label_list, preds_list
