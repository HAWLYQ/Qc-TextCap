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
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from .modeling_utils import (WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig, PreTrainedModel,
                             prune_linear_layer, add_start_docstrings)

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-config.json",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                     "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(PretrainedConfig):
    r"""
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        # print('pytorch_transformers_hu\modeling_bert.py BertConfig', kwargs)
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                                                               and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            if 'init_use_bbox_att' in kwargs:
                self.init_use_bbox_att = kwargs['init_use_bbox_att']
            if 'mid_use_bbox_att' in kwargs:
                self.mid_use_bbox_att = kwargs['mid_use_bbox_att']
            if 'init_use_bbox_att_purevision' in kwargs:
                self.init_use_bbox_att_purevision = kwargs['init_use_bbox_att_purevision']
            if 'objocr_pre_encoding_layers' in kwargs:
                self.objocr_pre_encoding_layers = kwargs['objocr_pre_encoding_layers']
            if 'drop_auto_question' in kwargs:
                # print('add drop_auto_question')
                self.drop_auto_question = kwargs['drop_auto_question']
            if 'drop_simple_cap' in kwargs:
                # print('add drop_auto_question')
                self.drop_simple_cap = kwargs['drop_simple_cap']
            if 'use_vqa_obj' in kwargs:
                self.use_vqa_obj = kwargs['use_vqa_obj']
            if 'use_vqa_ocr' in kwargs:
                self.use_vqa_ocr = kwargs['use_vqa_ocr']
            if 'use_raw_que' in kwargs:
                self.use_raw_que = kwargs['use_raw_que']
            if 'use_vision_que' in kwargs:
                self.use_vision_que = kwargs['use_vision_que']
            if 'only_attend_vision' in kwargs:
                self.only_attend_vision = kwargs['only_attend_vision']
            if 'use_discri_att' in kwargs:
                self.use_discri_att = kwargs['use_discri_att']
            if 'use_cross_encoder' in kwargs:
                self.use_cross_encoder = kwargs['use_cross_encoder']
            if 'que_vision_txt_fuse_type' in kwargs:
                self.que_vision_txt_fuse_type = kwargs['que_vision_txt_fuse_type']
            if 'sep_question_emb' in kwargs:
                self.sep_question_emb = kwargs['sep_question_emb']
            if 'avoid_repeat' in kwargs:
                self.avoid_repeat = kwargs['avoid_repeat']
            if 'drop_obj' in kwargs:
                self.drop_obj = kwargs['drop_obj']
            if 'drop_ocr' in kwargs:
                self.drop_ocr = kwargs['drop_ocr']

        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


# anwen hu 2021/3/14 for cross attention
class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_hidden_states, key_value_hidden_states, key_value_attention_mask, head_mask=None):
        # key_value_attention_mask : # batch * head * N_q * N_k
        mixed_query_layer = self.query(query_hidden_states) # batch * N_q * D_hid
        mixed_key_layer = self.key(key_value_hidden_states) # batch * N_k * D_hid
        mixed_value_layer = self.value(key_value_hidden_states) # batch * N_k * D_hid

        query_layer = self.transpose_for_scores(mixed_query_layer) # batch * head * N_q * D_hid
        key_layer = self.transpose_for_scores(mixed_key_layer) # batch * head * N_k * D_hid
        value_layer = self.transpose_for_scores(mixed_value_layer) # #  batch * head * N_k * D_hid

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # batch * head * N_q * N_k
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + key_value_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # batch * head * N_q * N_k

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) # # batch * head * N_q * D_hid

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


def displacement_emb_separate(obj_box, ocr_box):
    """
    calculate the displacement vector between objects and ocrs
    :param obj_box: batch * N_obj * 4
    :param ocr_box: batch * N_ocr * 4
    :return: batch * N_obj * N_ocr * 4
    """
    assert obj_box.size(0) == ocr_box.size(0)
    batch_size = obj_box.size(0)
    obj_num = obj_box.size(1)
    ocr_num = ocr_box.size(1)

    obj_x_min, obj_y_min, obj_x_max, obj_y_max = torch.chunk(obj_box, 4, dim=-1)
    ocr_x_min, ocr_y_min, ocr_x_max, ocr_y_max = torch.chunk(ocr_box, 4, dim=-1)

    obj_cx = (obj_x_min + obj_x_max) * 0.5
    obj_cy = (obj_y_min + obj_y_max) * 0.5
    obj_w = (obj_x_max - obj_x_min) + 1.
    obj_h = (obj_y_max - obj_y_min) + 1.

    ocr_cx = (ocr_x_min + ocr_x_max) * 0.5
    ocr_cy = (ocr_y_min + ocr_y_max) * 0.5
    ocr_w = (ocr_x_max - ocr_x_min) + 1.
    ocr_h = (ocr_y_max - ocr_y_min) + 1.

    delta_x = obj_cx - ocr_cx.view(batch_size, 1, -1)  # batch * N_obj * N_ocr
    delta_x = torch.clamp(torch.abs(delta_x / obj_w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = obj_cy - ocr_cy.view(batch_size, 1, -1)  # batch * N_obj * N_ocr
    delta_y = torch.clamp(torch.abs(delta_y / obj_h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(1. / (obj_w / ocr_w.view(batch_size, 1, -1)))  # batch * N_obj * N_ocr
    delta_h = torch.log(1. / (obj_h / ocr_h.view(batch_size, 1, -1)))  # batch * N_obj * N_ocr

    delta_x = delta_x.view(batch_size, obj_num, ocr_num, 1)
    delta_y = delta_y.view(batch_size, obj_num, ocr_num, 1)
    delta_w = delta_w.view(batch_size, obj_num, ocr_num, 1)
    delta_h = delta_h.view(batch_size, obj_num, ocr_num, 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
    return position_mat


def displacement_emb_union(obj_box, ocr_box, drop_obj, drop_ocr):
    """
    treat obj_box and ocr_box equally, calculate the displacement vector
    :param obj_box: batch * N_obj * 4
    :param ocr_box: batch * N_ocr * 4
    :return: batch * (N_obj+N_ocr) * (N_ocr+N_ocr) * 4
    """
    assert obj_box.size(0) == ocr_box.size(0)
    batch_size = obj_box.size(0)
    obj_num = obj_box.size(1)
    ocr_num = ocr_box.size(1)
    if drop_obj:
        union_num = ocr_num
        union_box = ocr_box  # batch * N_ocr * 4
    elif drop_ocr:
        union_num = obj_num
        union_box = obj_box  # batch * N_obj * 4
    else:
        union_num = obj_num + ocr_num
        union_box = torch.cat([obj_box, ocr_box], dim=1)  # batch * (N_obj+N_ocr) * 4

    x_min, y_min, x_max, y_max = torch.chunk(union_box, 4, dim=-1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = cx - cx.view(batch_size, 1, -1)  # batch * N_union * N_union
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)  # batch * N_union * N_union
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(1. / (w / w.view(batch_size, 1, -1)))  # batch * N_union * N_union
    delta_h = torch.log(1. / (h / h.view(batch_size, 1, -1)))  # batch * N_union * N_union

    delta_x = delta_x.view(batch_size, union_num, union_num, 1)
    delta_y = delta_y.view(batch_size, union_num, union_num, 1)
    delta_w = delta_w.view(batch_size, union_num, union_num, 1)
    delta_h = delta_h.view(batch_size, union_num, union_num, 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
    return position_mat


# anwenhu 2020/9/29
class BertSelfBoxAttentionObjOcrPreInds(nn.Module):
    """
    first do self-attetntion on obj, ocr, pre_inds features
    then do bbox influenced self-attention on obj and ocr features
    """
    def __init__(self, config):
        super(BertSelfBoxAttentionObjOcrPreInds, self).__init__()
        self.self = BertSelfAttention(config)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.displacement_linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(4, 1, bias=True)) for _ in range(self.num_attention_heads)])
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # batch, N, N_head, D_head
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, hidden_states, attention_mask, head_mask=None, bbox_coor=None):
        # attention_mask: batch * 1 * N_all * N_all (N_all=N_txt+N_obj+N_ocr+N_seq)
        # normal multi-head attention on all hidden_states (obj, ocr, pre_inds)
        selfatt_ouput = self.self(hidden_states, attention_mask, head_mask)[0]
        # add self attention influenced by box attention among obj and ocr features
        obj_begin = bbox_coor['obj_begin']  # obj_begin = 1
        ocr_end = bbox_coor['ocr_end']
        # print('pytorch_transformers\modeling_bert.py obj_begin', obj_begin)
        # print('pytorch_transformers\modeling_bert.py ocr_end', ocr_end)
        obj_box = bbox_coor['obj_bbox']
        ocr_box = bbox_coor['ocr_bbox']
        # print('pytorch_transformers\modeling_bert.py selfatt_ouput', selfatt_ouput.shape)
        # print('pytorch_transformers\modeling_bert.py attention_mask', attention_mask.shape)

        obj_ocr_hidden_states = selfatt_ouput[:, obj_begin:ocr_end]  # fetch obj and ocr hidden_states
        pre_hidden_states = selfatt_ouput[:,ocr_end:]  # fetch pre_inds hidden_states
        txt_hidden_states = selfatt_ouput[:,0:obj_begin]  # fetch question embedding (meaningless)
        obj_ocr_attention_mask = attention_mask[:, :, obj_begin:ocr_end, obj_begin:ocr_end] # fetch obj and ocr attention mask

        displacement_emb = displacement_emb_union(obj_box, ocr_box)  # batch * N_union * N_union * 4
        union_num = displacement_emb.size(1)
        batch_size = displacement_emb.size(0)
        flatten_displacement_emb = displacement_emb.view(-1, 4)
        dis_weights_per_head = [l(flatten_displacement_emb).view(batch_size, 1, union_num, union_num) for l in
                                self.displacement_linears]  # list(batch * 1 * N_union * N_union)
        dis_weights = torch.cat((dis_weights_per_head), 1)
        dis_weights = F.relu(dis_weights)  # batch * N_head * N_union * N_union

        mixed_query_layer = self.query(obj_ocr_hidden_states)  # batch * N * D_hid (N=N_union)
        mixed_key_layer = self.key(obj_ocr_hidden_states)  # batch * N * D_hid
        mixed_value_layer = self.value(obj_ocr_hidden_states)  # batch * N * D_hid

        query_layer = self.transpose_for_scores(mixed_query_layer)  # batch * N_head * N * D_head
        key_layer = self.transpose_for_scores(mixed_key_layer)  # batch * N_head * N * D_head
        value_layer = self.transpose_for_scores(mixed_value_layer)  # batch * N_head * N * D_head

        # Take the dot product between "query" and "key" to get the raw attention scores.
        visual_attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # batch * N_head * N * N
        visual_attention_scores = visual_attention_scores / math.sqrt(self.attention_head_size)

        # fuse displacement weights to visual attention scores
        attention_scores = torch.log(torch.clamp(dis_weights, min=1e-6)) + visual_attention_scores # batch * N_head * N * N
        # print('pytorch_transformers\modeling_bert.py fused attention scores', attention_scores.shape)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + obj_ocr_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) # batch * N_head * N * D_head

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # batch *  N *  N_head  * D_head
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # batch *  N * N_hid
        # concatenate question and pre_inds hidden states
        context_layer = torch.cat([txt_hidden_states, context_layer, pre_hidden_states], dim=1)
        # print('pytorch_transformers\modeling_bert.py selfboxatttnion final output', context_layer.shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


# anwen hu 2020/10/13
class BertSelfBoxAttentionObjOcr(nn.Module):
    """
    only do bbox influenced self-attention on obj and ocr image features
    """
    def __init__(self, config):
        super(BertSelfBoxAttentionObjOcr, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.displacement_linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(4, 1, bias=True)) for _ in range(self.num_attention_heads)])
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # anwen hu 2021/7/23
        self.drop_obj = config.drop_obj
        self.drop_ocr = config.drop_ocr

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # batch, N, N_head, D_head
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None, bbox_coor=None):
        # hidden_states； batch * N_all* D_h
        # attention_mask: batch * 1 * N_all * N_all (N_all=N_obj+N_ocr)

        # add self attention influenced by box attention among obj and ocr features

        # print('pytorch_transformers\modeling_bert.py obj_begin', obj_begin)
        # print('pytorch_transformers\modeling_bert.py ocr_end', ocr_end)
        obj_box = bbox_coor['obj_bbox']
        ocr_box = bbox_coor['ocr_bbox']

        # calculate bbox attention
        displacement_emb = displacement_emb_union(obj_box, ocr_box, self.drop_obj, self.drop_ocr)  # batch * N_union * N_union * 4
        union_num = displacement_emb.size(1)
        batch_size = displacement_emb.size(0)
        flatten_displacement_emb = displacement_emb.view(-1, 4)
        dis_weights_per_head = [l(flatten_displacement_emb).view(batch_size, 1, union_num, union_num) for l in
                                self.displacement_linears]  # list(batch * 1 * N_union * N_union)
        dis_weights = torch.cat((dis_weights_per_head), 1)
        dis_weights = F.relu(dis_weights)  # batch * N_head * N_union * N_union

        mixed_query_layer = self.query(hidden_states)  # batch * N * D_hid (N=N_union)
        mixed_key_layer = self.key(hidden_states)  # batch * N * D_hid
        mixed_value_layer = self.value(hidden_states)  # batch * N * D_hid

        query_layer = self.transpose_for_scores(mixed_query_layer)  # batch * N_head * N * D_head
        key_layer = self.transpose_for_scores(mixed_key_layer)  # batch * N_head * N * D_head
        value_layer = self.transpose_for_scores(mixed_value_layer)  # batch * N_head * N * D_head

        # Take the dot product between "query" and "key" to get the raw attention scores.
        visual_attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # batch * N_head * N * N
        visual_attention_scores = visual_attention_scores / math.sqrt(self.attention_head_size)

        # fuse displacement weights to visual attention scores
        attention_scores = torch.log(torch.clamp(dis_weights, min=1e-6)) + visual_attention_scores # batch * N_head * N * N
        # print('pytorch_transformers\modeling_bert.py fused attention scores', attention_scores.shape)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) # batch * N_head * N * D_head

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # batch *  N *  N_head  * D_head
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # batch *  N * N_hid
        # print('pytorch_transformers\modeling_bert.py BertSelfBoxAttentionObjOcr final output', context_layer.shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class DiscriBertSelfAttention(nn.Module):
    """
    only do bbox influenced self-attention on obj and ocr image features
    """
    def __init__(self, config):
        super(DiscriBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        """self.displacement_linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(4, 1, bias=True)) for _ in range(self.num_attention_heads)])"""
        # self.overall_query = nn.Linear(config.hidden_size, self.all_head_size)
        # self.overall_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # batch, N, N_head, D_head
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None, modality_attention_mask=None):
        # hidden_states； batch * N_all* D_h
        # attention_mask: batch * 1 * N_all * N_all (N_all=N_obj+N_ocr)

        # add self attention influenced by overall semantic relevance
        # order: que_feat, cap_emb, obj_feat, ocr_feat, dec_emb
        obj_mask = modality_attention_mask['obj_mask']
        ocr_mask = modality_attention_mask['ocr_mask']
        que_mask = modality_attention_mask['que_mask']
        cap_mask = modality_attention_mask['cap_mask']
        # dec_mask = modality_attention_mask['dec_mask']
        dec_max_num = modality_attention_mask['dec_max_num']

        que_begin = 0
        cap_begin = que_begin + que_mask.size(-1)
        obj_begin = cap_begin + cap_mask.size(-1)
        ocr_begin = obj_begin + obj_mask.size(-1)
        dec_begin = ocr_begin + ocr_mask.size(-1)

        # get overall semantic for each modality
        obj_feat = hidden_states[:, obj_begin:obj_begin + obj_mask.size(-1)] * obj_mask.unsqueeze(2)  # batch * N_obj * D_hid
        ocr_feat = hidden_states[:, ocr_begin:ocr_begin + ocr_mask.size(-1)] * ocr_mask.unsqueeze(2)
        que_feat = hidden_states[:, que_begin:que_begin + que_mask.size(-1)] * que_mask.unsqueeze(2)
        cap_feat = hidden_states[:, cap_begin:cap_begin + cap_mask.size(-1)] * cap_mask.unsqueeze(2)
        obj_overall_feat = torch.sum(obj_feat, dim=1)/torch.sum(obj_mask, dim=-1).unsqueeze(1) # batch * D_hid
        ocr_overall_feat = torch.sum(ocr_feat, dim=1) / torch.sum(ocr_mask, dim=-1).unsqueeze(1)
        que_overall_feat = torch.sum(que_feat, dim=1) / torch.sum(que_mask, dim=-1).unsqueeze(1)
        cap_overall_feat = torch.sum(cap_feat, dim=1) / torch.sum(cap_mask, dim=-1).unsqueeze(1)
        # concatenate according the order:  que_feat, cap_emb, obj_feat, ocr_feat, dec_emb
        overall_feats = torch.cat([que_overall_feat.unsqueeze(1).repeat(1, que_mask.size(-1), 1),
                                   cap_overall_feat.unsqueeze(1).repeat(1, cap_mask.size(-1), 1),
                                   obj_overall_feat.unsqueeze(1).repeat(1, obj_mask.size(-1), 1),
                                   ocr_overall_feat.unsqueeze(1).repeat(1, ocr_mask.size(-1), 1)], dim=1)  # batch * (N-N_dec) * D_hid
        # calculate overall semantic attention among obj, ocr, que, cap
        dec_feat = hidden_states[:, dec_begin:] # batch * N_dec * D_hid
        # overall_query_feat = self.overall_query(dec_feat)  # batch * N_dec * D_hid
        # overall_key_feat = self.overall_key(overall_feats)  # batch * (N-N_dec) * D_hid
        overall_query_feat = self.query(dec_feat)  # batch * N_dec * D_hid
        overall_key_feat = self.key(overall_feats)  # batch * (N-N_dec) * D_hid
        overall_query_feat = self.transpose_for_scores(overall_query_feat) # batch * N_head * N_dec * D_hid
        overall_key_feat = self.transpose_for_scores(overall_key_feat) # batch * N_head * (N-N_dec) * D_hid
        overall_attention_scores = torch.matmul(overall_query_feat, overall_key_feat.transpose(-1, -2))  # batch * N_head * N_dec * (N-N_dec)
        # calculate overall semantic attention with dec
        dec_mask = attention_mask[:, :, -dec_max_num:, -dec_max_num:].squeeze(1).unsqueeze(3)  # batch * N_dec * N_dec * 1
        dec_overall_feat = dec_feat.unsqueeze(1).repeat(1, dec_feat.size(1), 1, 1) * dec_mask  # batch * N_dec * N_dec * D_hid
        dec_overall_feat = torch.sum(dec_overall_feat, dim=2)  # batch * N_dec * D_hid
        dec_overall_key_feat = self.transpose_for_scores(self.key(dec_overall_feat)) # batch * N_head * N_dec * D_hid
        # mul + sum = dot product
        dec_overall_attention_scores = torch.sum(torch.mul(overall_query_feat, dec_overall_key_feat), dim=-1) # batch * N_head * N_dec
        dec_overall_attention_scores = dec_overall_attention_scores.unsqueeze(3).repeat(1, 1, 1, dec_max_num) # batch * N_head * N_dec * N_dec
        # the whole overall attention map for dec tokens
        overall_attention_scores = torch.cat([overall_attention_scores, dec_overall_attention_scores], dim=-1) # batch * N_head * N_dec * N
        overall_attention_scores = overall_attention_scores / math.sqrt(self.attention_head_size) # batch * N_head * N_dec * N
        # overall_weights = F.relu(overall_attention_scores)  # batch * N_head * N_dec * 4

        # calculate each token attention
        mixed_query_layer = self.query(hidden_states)  # batch * N * D_hid (N=N_union)
        mixed_key_layer = self.key(hidden_states)  # batch * N * D_hid
        mixed_value_layer = self.value(hidden_states)  # batch * N * D_hid

        query_layer = self.transpose_for_scores(mixed_query_layer)  # batch * N_head * N * D_head
        key_layer = self.transpose_for_scores(mixed_key_layer)  # batch * N_head * N * D_head
        value_layer = self.transpose_for_scores(mixed_value_layer)  # batch * N_head * N * D_head

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # batch * N_head * N * N
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # fuse semantic weights to self attention scores
        # attention_scores = torch.log(torch.clamp(overall_weights, min=1e-6)) + visual_attention_scores # batch * N_head * N * N
        self_attention_scores = attention_scores[:, :, 0:dec_begin,:]
        dec_attention_scores = attention_scores[:, :, -dec_max_num:, :] # batch * N_head * N_dec * N
        fusion_dec_attention_scores =overall_attention_scores+dec_attention_scores
        attention_scores = torch.cat([self_attention_scores, fusion_dec_attention_scores], dim=2) # batch * N_head * N * N

        # print('pytorch_transformers\modeling_bert.py fused attention scores', attention_scores.shape)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) # batch * N_head * N * D_head

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # batch *  N *  N_head  * D_head
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # batch *  N * N_hid
        # print('pytorch_transformers\modeling_bert.py BertSelfBoxAttentionObjOcr final output', context_layer.shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.mid_use_bbox_att = config.mid_use_bbox_att
        # self.self = BertSelfAttention(config)
        # Anwen Hu 2020/10/13 whether to add box influenced self-attention between normal layers
        # print('modelling_bert.py BertAttention self.mid_use_bbox_att', self.mid_use_bbox_att)
        if self.mid_use_bbox_att:
            self.self = BertSelfBoxAttentionObjOcrPreInds(config)
        else:
            self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Anwen Hu 2020/10/6 revise forward input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
    def forward(self, input_tensor, attention_mask, head_mask=None, bbox_coor=None):
        if not self.mid_use_bbox_att:
            self_outputs = self.self(input_tensor, attention_mask, head_mask)
        else:
            # Anwen Hu 2020/10/6
            # revise self-attention input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
            self_outputs = self.self(input_tensor, attention_mask, head_mask, bbox_coor)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# anwenhu use bert cross attention + residual
class CrossBertAttention(nn.Module):
    def __init__(self, config):
        super(CrossBertAttention, self).__init__()
        self.cross = BertCrossAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, query_hidden_states, key_value_hidden_states, key_value_attention_mask, head_mask=None):
        cross_outputs = self.cross(query_hidden_states, key_value_hidden_states, key_value_attention_mask, head_mask)
        attention_output = self.output(cross_outputs[0], query_hidden_states)
        outputs = (attention_output,) + cross_outputs[1:]  # add attentions if we output them
        return outputs


# anwen hu 2020/10/13 designed specially for pre self-attention among obj and ocr features
class BertAttentionObjOcr(nn.Module):
    def __init__(self, config):
        super(BertAttentionObjOcr, self).__init__()
        # self.self = BertSelfAttention(config)
        # Anwen Hu 2020/10/13 revise self-attention
        self.self = BertSelfBoxAttentionObjOcr(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Anwen Hu 2020/10/6 revise forward input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
    def forward(self, input_tensor, attention_mask, head_mask=None, bbox_coor=None):
        # self_outputs = self.self(input_tensor, attention_mask, head_mask)
        # Anwen Hu 2020/10/6 revise self-attention input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
        self_outputs = self.self(input_tensor, attention_mask, head_mask, bbox_coor)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class DiscriBertAttention(nn.Module):
    def __init__(self, config):
        super(DiscriBertAttention, self).__init__()
        # self.self = BertSelfAttention(config)
        # Anwen Hu 2020/10/13 revise self-attention
        self.self = DiscriBertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Anwen Hu 2020/3/14 revise forward input ,add modality attention mask and modality begin
    def forward(self, input_tensor, attention_mask, head_mask=None, modality_attention_mask=None):
        # self_outputs = self.self(input_tensor, attention_mask, head_mask)
        # Anwen Hu 2020/10/6 revise self-attention input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
        self_outputs = self.self(input_tensor, attention_mask, head_mask, modality_attention_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    # Anwen Hu 2020/10/6 revise forward input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
    def forward(self, hidden_states, attention_mask, head_mask=None, bbox_coor=None):
        # Anwen Hu 2020/10/6 revise attention input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, bbox_coor)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class CrossBertLayer(nn.Module):
    def __init__(self, config):
        super(CrossBertLayer, self).__init__()
        self.attention = CrossBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, query_hidden_states, key_value_hidden_states, key_value_attention_mask, head_mask=None):
        # Anwen Hu 2020/10/6 revise attention input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
        attention_outputs = self.attention(query_hidden_states, key_value_hidden_states, key_value_attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class DiscriBertLayer(nn.Module):
    def __init__(self, config):
        super(DiscriBertLayer, self).__init__()
        self.attention = DiscriBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None, modality_attention_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, modality_attention_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


# anwen hu 2020/10/13 designed specially for pre self-attention among obj and ocr features
class BertLayerObjOcr(nn.Module):
    def __init__(self, config):
        super(BertLayerObjOcr, self).__init__()
        self.attention = BertAttentionObjOcr(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    # Anwen Hu 2020/10/6 revise forward input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
    def forward(self, hidden_states, attention_mask, head_mask=None, bbox_coor=None):
        # Anwen Hu 2020/10/6 revise attention input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, bbox_coor)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoderMid(nn.Module):
    def __init__(self, config):
        super(BertEncoderMid, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # Anwen Hu 2020/10/6 revise forward input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
    def forward(self, hidden_states, attention_mask, head_mask=None, bbox_coor=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # Anwen Hu 2020/10/6 revise layer input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
            # print('pytorch_transformers_hu\modeling_bert.py BertEncoder bbox_ocr', bbox_coor)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], bbox_coor)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


# anwen hu 2020/10/13 add encoder for obj and ocr features before multimodal transformer
class BertEncoderPre(nn.Module):
    def __init__(self, config):
        super(BertEncoderPre, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.obj_ocr_layer = nn.ModuleList([BertLayerObjOcr(config) for _ in range(config.num_hidden_layers)])
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # Anwen Hu 2020/10/6 revise forward input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
    def forward(self, hidden_states, attention_mask, head_mask=None, bbox_coor=None):
        all_hidden_states = ()  # store hidden_states of each layer
        all_attentions = ()  # store hidden_states of each layer

        # fetch obj and ocr features
        obj_begin = bbox_coor['obj_begin']  # obj_begin = 1
        ocr_end = bbox_coor['ocr_end']
        # print('modeling_bert.py BertEncoderPre obj_begin', obj_begin)
        # print('modeling_bert.py BertEncoderPre ocr_end', ocr_end)
        # print('modeling_bert.py BertEncoderPre hidden_states', hidden_states.shape)
        # print('modeling_bert.py BertEncoderPre attention_mask', attention_mask.shape)
        obj_ocr_hidden_states = hidden_states[:, obj_begin:ocr_end]  # fetch obj and ocr hidden_states
        pre_hidden_states = hidden_states[:, ocr_end:]  # fetch pre_inds hidden_states
        txt_hidden_states = hidden_states[:, 0:obj_begin]  # fetch question embedding (meaningless)
        obj_ocr_attention_mask = attention_mask[:, :, obj_begin:ocr_end, obj_begin:ocr_end]  # fetch obj and ocr attention mask

        # Anwen Hu 2020/10/13 add pre self-attention layers for obj and ocr features
        for i, layer_module in enumerate(self.obj_ocr_layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (obj_ocr_hidden_states,)
            # Anwen Hu 2020/10/6 revise layer input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
            # print('pytorch_transformers_hu\modeling_bert.py BertEncoderPre bbox_ocr', bbox_coor)
            obj_ocr_layer_outputs = layer_module(obj_ocr_hidden_states, obj_ocr_attention_mask, head_mask[i], bbox_coor)
            obj_ocr_hidden_states = obj_ocr_layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (obj_ocr_layer_outputs[1],)

        # print('modeling_bert.py BertEncoderPre obj_ocr_hidden_states', obj_ocr_hidden_states.shape)
        # concatenate processed obj and ocr hidden states with question and pre_inds hidden states
        hidden_states = torch.cat([txt_hidden_states, obj_ocr_hidden_states, pre_hidden_states], dim=1)

        # normal self-attention
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # Anwen Hu 2020/10/6 revise layer input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
            # print('pytorch_transformers_hu\modeling_bert.py BertEncoder bbox_ocr', bbox_coor)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], bbox_coor)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


# anwen hu 2021/3/13 # use DiscriBertLayers
class DiscriBertEncoder(nn.Module):
    def __init__(self, config):
        super(DiscriBertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([DiscriBertLayer(config) for _ in range(config.num_hidden_layers)])

    # Anwen Hu 2020/10/6 revise forward input ,add modality_attention_mask,
    def forward(self, hidden_states, attention_mask, head_mask=None, modality_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # Anwen Hu 2020/10/6 revise layer input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
            # print('pytorch_transformers_hu\modeling_bert.py BertEncoder bbox_ocr', bbox_coor)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], modality_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


# anwen hu 2021/3/14 # use CrossBertLayers
class CrossBertEncoder(nn.Module):
    def __init__(self, config):
        super(CrossBertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([CrossBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, query_hidden_states, key_value_hidden_states, key_value_attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(query_hidden_states, key_value_hidden_states, key_value_attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.BertConfig`): Model configuration class with all the parameters of the model. 
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``
                
                ``token_type_ids:   0   0   0   0  0     0   0``

            Bert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`pytorch_transformers.BertTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


@add_start_docstrings("The bare Bert Model transformer outputing raw hidden-states without any specific head on top.",
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with two heads on top as done during the pre-training:
    a `masked language modeling` head and a `next sentence prediction (classification)` head. """,
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForPreTraining(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[
                                                                 2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """,
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `next sentence prediction (classification)` head on top. """,
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForNextSentencePrediction(BertPreTrainedModel):
    r"""
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``next_sentence_label`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next sequence prediction (classification) loss.
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        seq_relationship_scores = outputs[0]

    """

    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
                      BERT_START_DOCSTRING)
class BertForMultipleChoice(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``
                
                ``token_type_ids:   0   0   0   0  0     0   0``
    
            Indices can be obtained using :class:`pytorch_transformers.BertTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.bert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForQuestionAnswering(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]

    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,
                end_positions=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
