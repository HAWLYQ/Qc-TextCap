# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import math
import torch
from torch import nn
import torch.nn.functional as F
import copy

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

from pytorch_transformers_hu.modeling_bert import BertEncoderMid
from pytorch_transformers_hu.modeling_bert import BertEncoderPre
from pytorch_transformers_hu.modeling_bert import BertConfig as BertConfigHu
# fix the bug that BertConfigHu is not a instance of PretrainedConfig
from pytorch_transformers_hu.modeling_bert import BertPreTrainedModel as BertPreTrainedModelHu
from pytorch_transformers_hu.modeling_bert import BertLayerObjOcr

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.layers import ClassifierLayer

from pythia.modules.encoders import ImageEncoder


@registry.register_model("m4c")
class M4C(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # self.mmt_config = BertConfig(**self.config.mmt)
        # anwen hu 2020/10/7 add mid_use_bbox_att to mmt_config
        # anwen hu 2020/10/13 add init_use_bbox_att to mmt_config
        # anwen hu 2020/10/20 add init_use_bbox_att_purevision to mmt_config
        self.mmt_config = BertConfigHu(**self.config.mmt)
        # print('m4c.py self.mmt_config.init_use_bbox_att_purevision', self.mmt_config.init_use_bbox_att_purevision)
        # print('m4c.py self.mmt_config.init_use_bbox_att', self.mmt_config.init_use_bbox_att)
        # print('m4c.py self.mmt_config.mid_use_bbox_att', self.mmt_config.mid_use_bbox_att)
        self._datasets = registry.get("config").datasets.split(",")
        # anwen hu
        self.sc_learning = False

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        if self.mmt_config.init_use_bbox_att_purevision:
            self._build_obj_ocr_encoding()
        else:
            self._build_obj_encoding()
            self._build_ocr_encoding()
        self._build_mmt()
        self._build_output()

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.config.text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                'bert-base-uncased', config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append({
                'module': self.text_bert,
                'lr_scale': self.config.lr_scale_text_bert,
            })
        else:
            self.writer.write('NOT initializing text_bert from BERT_BASE')
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            self.writer.write(
                'Projecting text_bert output to {} dim'.format(
                    self.mmt_config.hidden_size
                )
            )
            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):
        # object appearance feature: Faster R-CNN
        self.obj_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append({
            'module': self.obj_faster_rcnn_fc7,
            'lr_scale': self.config.lr_scale_frcn,
        })
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.obj_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)

    def _build_ocr_encoding(self):
        self.remove_ocr_fasttext = getattr(
            self.config.ocr, 'remove_ocr_fasttext', False
        )
        self.remove_ocr_phoc = getattr(
            self.config.ocr, 'remove_ocr_phoc', False
        )
        self.remove_ocr_frcn = getattr(
            self.config.ocr, 'remove_ocr_frcn', False
        )
        self.remove_ocr_semantics = getattr(
            self.config.ocr, 'remove_ocr_semantics', False
        )
        self.remove_ocr_bbox = getattr(
            self.config.ocr, 'remove_ocr_bbox', False
        )

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        self.finetune_modules.append({
            'module': self.ocr_faster_rcnn_fc7,
            'lr_scale': self.config.lr_scale_frcn,
        })

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    # anwen hu 2020/10/20
    def _build_obj_ocr_encoding(self):
        """
        architecture for encoding obj and ocr vision features before MMT
        :return:
        """
        # for obj
        # object appearance feature: Faster R-CNN
        self.obj_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append({
            'module': self.obj_faster_rcnn_fc7,
            'lr_scale': self.config.lr_scale_frcn,
        })
        # convert 2048 dim vector to 768 dim vector
        self.linear_obj_feat_to_bboxen_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.mmt_config.hidden_size, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.obj_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)
        # for ocr
        self.remove_ocr_fasttext = getattr(
            self.config.ocr, 'remove_ocr_fasttext', False
        )
        self.remove_ocr_phoc = getattr(
            self.config.ocr, 'remove_ocr_phoc', False
        )
        self.remove_ocr_frcn = getattr(
            self.config.ocr, 'remove_ocr_frcn', False
        )
        self.remove_ocr_semantics = getattr(
            self.config.ocr, 'remove_ocr_semantics', False
        )
        self.remove_ocr_bbox = getattr(
            self.config.ocr, 'remove_ocr_bbox', False
        )

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        self.finetune_modules.append({
            'module': self.ocr_faster_rcnn_fc7,
            'lr_scale': self.config.lr_scale_frcn,
        })
        # use pure vision, so input dim is self.config.obj.mmt_in_dim(2048) rather than self.config.ocr.mmt_in_dim(3002)
        self.linear_ocr_feat_to_bboxen_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )
        # 1672 = 300 (FastText) + 604 (PHOC) + 768 (ocr hidden)
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            1672, self.mmt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

        # for fuse
        self.objocr_bert = nn.ModuleList([BertLayerObjOcr(self.mmt_config) for _ in range(self.mmt_config.objocr_pre_encoding_layers)])

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append({
            'module': self.mmt,
            'lr_scale': self.config.lr_scale_mmt,
        })

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)

        # fixed answer vocabulary scores
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        num_choices -= self.config.classifier.ocr_max_num
        self.fixed_vocab_size = num_choices
        self.classifier = ClassifierLayer(
            self.config["classifier"]["type"],
            in_dim=self.mmt_config.hidden_size,
            out_dim=num_choices,
            **self.config["classifier"]["params"]
        )

        self.answer_processor = registry.get(
            self._datasets[0] + "_answer_processor"
        )
        # print('m4c.py self.answer_processor', self.answer_processor)  pythia.datasets.processors.Processor

    def forward(self, sample_list, **kwargs):

        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(sample_list, fwd_results)
        if self.mmt_config.init_use_bbox_att_purevision:
            self._forward_objocr_encoding(sample_list, fwd_results)
        else:
            self._forward_obj_encoding(sample_list, fwd_results)
            self._forward_ocr_encoding(sample_list, fwd_results)
        if 'sc' in kwargs and kwargs['sc']:
            self.sc_learning = True
            with torch.no_grad():
                self._forward_mmt_and_output(sample_list, fwd_results, decoding_strategy='greedy')  # first greedy
                self._forward_mmt_and_output(sample_list, fwd_results, decoding_strategy='sample')  # then sample, so prev_inds store the inds of sample decoding

            self._forward_mmt_and_output_sample(sample_list, fwd_results)
            results = {"scores": fwd_results["scores"], "scores_sample": fwd_results["scores_sample"],
                       'sampleseqLogprobs': fwd_results["sampleseqLogprobs"], 'prev_inds': fwd_results["prev_inds"]}
            # print('m4c.py sampleseqLogprobs', fwd_results["sampleseqLogprobs"])
            return results
        else:
            # assert kwargs['beam_size'] > 0
            if 'beam_size' in kwargs and kwargs['beam_size'] > 0:
                self.beam_size = kwargs['beam_size']
                self._forward_mmt_and_output(sample_list, fwd_results, decoding_strategy='beam')
            else:
                self._forward_mmt_and_output(sample_list, fwd_results)
            # only keep scores in the forward pass results
            results = {"scores": fwd_results["scores"], 'prev_inds': fwd_results["prev_inds"]}
            if 'beam_size' in kwargs and kwargs['beam_size']>0:
                results['final_beams'] = fwd_results['final_beams']
            return results

    def _forward_txt_encoding(self, sample_list, fwd_results):
        fwd_results['txt_inds'] = sample_list.text

        # binary mask of valid text (question words) vs padding
        fwd_results['txt_mask'] = _get_mask(
            sample_list.text_len, sample_list.text.size(1)
        )

    # anwen hu 2020/10/20 encoding obj and ocr vision features with bbox
    def _forward_objocr_encoding(self, sample_list, fwd_results):
        # processing obj vision features
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)
        obj_feat = self.linear_obj_feat_to_bboxen_in(obj_fc7)  # 2048 > 768
        # print('m4c.py _forward_objocr_encoding obj_feat', obj_feat.shape)
        # obj_feat = self.obj_drop(self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(obj_fc7)))
        # max_features store the original obj num in the numpy feature file: 100
        obj_nums = sample_list.image_info_0.max_features  # batch_size
        # print('m4c.py _forward_objocr_encoding obj_nums', obj_nums)
        obj_mask = _get_mask(obj_nums, obj_feat.size(1))  # batch * features_max_len
        # print('m4c.py _forward_objocr_encoding obj_mask', obj_mask.shape)

        # processing ocr vision features
        ocr_fasttext = sample_list.context_feature_0
        # print('m4c.py _forward_objocr_encoding sample_list.image_feature_1', sample_list.image_feature_1.shape)
        ocr_fc6 = sample_list.image_feature_1[:, :ocr_fasttext.size(1), :]
        # print('m4c.py _forward_objocr_encoding ocr_fc6', ocr_fc6.shape)
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)
        ocr_feat = self.linear_ocr_feat_to_bboxen_in(ocr_fc7)  # 2048 > 768
        # print('m4c.py _forward_objocr_encoding ocr_feat', ocr_feat.shape)
        # cr_feat = self.ocr_drop(self.ocr_feat_layer_norm(self.linear_ocr_feat_to_mmt_in(ocr_fc7)))
        ocr_nums = sample_list.context_info_0.max_features
        # print('m4c.py _forward_objocr_encoding ocr_nums', ocr_nums)
        ocr_mask = _get_mask(ocr_nums, ocr_feat.size(1))

        # prepare attention mask
        attention_mask = torch.cat([obj_mask, ocr_mask], dim=1)
        objocr_num = attention_mask.size(1)  # N
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # batch * 1 * 1 * N
        extended_attention_mask = extended_attention_mask.repeat(1, 1, objocr_num, 1)  # batch * 1 * N * N
        head_mask = [None] * self.mmt_config.objocr_pre_encoding_layers

        obj_ocr_hidden_states = torch.cat([obj_feat, ocr_feat], dim=1)
        obj_bbox = sample_list.obj_bbox_coordinates[:, :obj_fc6.shape[1], :]  # batch * features_max_len * 4
        ocr_bbox = sample_list.ocr_bbox_coordinates
        bbox_coor = {'ocr_bbox': ocr_bbox, 'obj_bbox': obj_bbox}
        obj_end = obj_mask.size(-1)
        # print('m4c.py obj_end', obj_end)
        # feed to obj and ocr encoding layers
        for i, layer_module in enumerate(self.objocr_bert):
            # Anwen Hu 2020/10/6 revise layer input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
            # print('pytorch_transformers_hu\modeling_bert.py BertEncoderPre bbox_ocr', bbox_coor)
            obj_ocr_layer_outputs = layer_module(obj_ocr_hidden_states, extended_attention_mask, head_mask[i], bbox_coor)
            obj_ocr_hidden_states = obj_ocr_layer_outputs[0]
        obj_hidden_states = obj_ocr_hidden_states[:, :obj_end]
        ocr_hidden_states = obj_ocr_hidden_states[:, obj_end:]
        # print('m4c.py _forward_objocr_encoding obj_hidden_states', obj_hidden_states.shape)
        # print('m4c.py _forward_objocr_encoding ocr_hidden_states', ocr_hidden_states.shape)

        # add spatial features for obj hidden states
        obj_mmt_in = (
            self.obj_feat_layer_norm(
                self.linear_obj_feat_to_mmt_in(obj_hidden_states)
            ) + self.obj_bbox_layer_norm(
            self.linear_obj_bbox_to_mmt_in(obj_bbox)
        )
        )
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results['obj_mmt_in'] = obj_mmt_in
        fwd_results['obj_mask'] = obj_mask

        # add features for ocr hidden states
        # OCR FastText feature (300-dim)
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        if self.remove_ocr_fasttext:
            ocr_fasttext = torch.zeros_like(ocr_fasttext)
        if self.remove_ocr_phoc:
            ocr_phoc = torch.zeros_like(ocr_phoc)

        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_hidden_states],
            dim=-1
        )

        if self.remove_ocr_semantics:
            ocr_feat = torch.zeros_like(ocr_feat)
        if self.remove_ocr_bbox:
            ocr_bbox = torch.zeros_like(ocr_bbox)
        ocr_mmt_in = (
            self.ocr_feat_layer_norm(
                self.linear_ocr_feat_to_mmt_in(ocr_feat)
            ) + self.ocr_bbox_layer_norm(
            self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
        )
        )
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results['ocr_mmt_in'] = ocr_mmt_in
        fwd_results['ocr_mask'] = ocr_mask

    def _forward_obj_encoding(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = obj_fc7
        # print('m4c.py _forward_obj_encoding sample_list.obj_bbox_coordinates', sample_list.obj_bbox_coordinates.shape)
        obj_bbox = sample_list.obj_bbox_coordinates[:, :obj_fc6.shape[1], :]
        # print('m4c.py _forward_obj_encoding obj_bbox', obj_bbox.shape)
        obj_mmt_in = (
            self.obj_feat_layer_norm(
                self.linear_obj_feat_to_mmt_in(obj_feat)
            ) + self.obj_bbox_layer_norm(
            self.linear_obj_bbox_to_mmt_in(obj_bbox)
        )
        )
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results['obj_mmt_in'] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features
        fwd_results['obj_mask'] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_encoding(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:, :ocr_fasttext.size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_order_vectors = torch.zeros_like(sample_list.order_vectors)

        if self.remove_ocr_fasttext:
            ocr_fasttext = torch.zeros_like(ocr_fasttext)
        if self.remove_ocr_phoc:
            ocr_phoc = torch.zeros_like(ocr_phoc)
        if self.remove_ocr_frcn:
            ocr_fc7 = torch.zeros_like(ocr_fc7)
        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc7, ocr_order_vectors],
            dim=-1
        )
        ocr_bbox = sample_list.ocr_bbox_coordinates
        # print('pythia/models/m4c.py ocr_feat:', ocr_feat.size())
        # print('pythia/models/m4c.py ocr_bbox:', ocr_bbox.size())
        if self.remove_ocr_semantics:
            ocr_feat = torch.zeros_like(ocr_feat)
        if self.remove_ocr_bbox:
            ocr_bbox = torch.zeros_like(ocr_bbox)
        ocr_mmt_in = (
            self.ocr_feat_layer_norm(
                self.linear_ocr_feat_to_mmt_in(ocr_feat)
            ) + self.ocr_bbox_layer_norm(
            self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
        )
        )
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results['ocr_mmt_in'] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results['ocr_mask'] = _get_mask(ocr_nums, ocr_mmt_in.size(1))

    def _forward_mmt(self, sample_list, fwd_results):
        # first forward the text BERT layers
        text_bert_out = self.text_bert(
            txt_inds=fwd_results['txt_inds'],
            txt_mask=fwd_results['txt_mask']
        )
        fwd_results['txt_emb'] = self.text_bert_out_linear(text_bert_out) # question text, none for textcaps


        # anwen hu 2020/10/7 add bbox_info containing obj_box, ocr_box
        ocr_bbox = sample_list.ocr_bbox_coordinates
        obj_bbox = sample_list.obj_bbox_coordinates
        # print('m4c.py forward mmt ocr_bbox shape', ocr_bbox.shape)
        # print('m4c.py forward mmt obj_bbox shape', obj_bbox.shape)
        mmt_results = self.mmt(
            txt_emb=fwd_results['txt_emb'],
            txt_mask=fwd_results['txt_mask'],
            obj_emb=fwd_results['obj_mmt_in'],
            obj_mask=fwd_results['obj_mask'],
            ocr_emb=fwd_results['ocr_mmt_in'],
            ocr_mask=fwd_results['ocr_mask'],
            fixed_ans_emb=self.classifier.module.weight,
            prev_inds=fwd_results['prev_inds'],
            bbox_coor={'ocr_bbox':ocr_bbox,'obj_bbox':obj_bbox}
        )
        fwd_results.update(mmt_results)

    def _forward_mmt_txt(self, sample_list, fwd_results):
        text_bert_out = self.text_bert(
            txt_inds=fwd_results['txt_inds'],
            txt_mask=fwd_results['txt_mask']
        )
        fwd_results['txt_emb'] = self.text_bert_out_linear(text_bert_out)

    def _forward_mmt_mmt(self, sample_list, fwd_results):
        # anwen hu 2020/5/20 for the first step of beam search
        if fwd_results['prev_inds'].size(0) != fwd_results['txt_emb'].size(0):
            print('m4c.py line 320 used')
            fwd_results['txt_emb'] = fwd_results['txt_emb'].repeat(fwd_results['prev_inds'].size(0), 1, 1)
            fwd_results['txt_mask'] = fwd_results['txt_mask'].repeat(fwd_results['prev_inds'].size(0), 1)
            fwd_results['obj_mmt_in'] = fwd_results['obj_mmt_in'].repeat(fwd_results['prev_inds'].size(0), 1, 1)
            fwd_results['obj_mask'] = fwd_results['obj_mask'].repeat(fwd_results['prev_inds'].size(0), 1)
            fwd_results['ocr_mmt_in'] = fwd_results['ocr_mmt_in'].repeat(fwd_results['prev_inds'].size(0), 1, 1)
            fwd_results['ocr_mask'] = fwd_results['ocr_mask'].repeat(fwd_results['prev_inds'].size(0), 1)
            """print(fwd_results['txt_emb'].size())
            print(fwd_results['txt_mask'].size())
            print(fwd_results['obj_mmt_in'].size())
            print(fwd_results['obj_mask'].size())
            print(fwd_results['ocr_mmt_in'].size())
            print(fwd_results['ocr_mask'].size())"""

        # anwen hu 2020/10/7 add bbox_info containing obj_box, ocr_box
        ocr_bbox = sample_list.ocr_bbox_coordinates
        obj_bbox = sample_list.obj_bbox_coordinates
        mmt_results = self.mmt(
            txt_emb=fwd_results['txt_emb'],
            txt_mask=fwd_results['txt_mask'],
            obj_emb=fwd_results['obj_mmt_in'],
            obj_mask=fwd_results['obj_mask'],
            ocr_emb=fwd_results['ocr_mmt_in'],
            ocr_mask=fwd_results['ocr_mask'],
            fixed_ans_emb=self.classifier.module.weight,
            prev_inds=fwd_results['prev_inds'],
            bbox_coor={'ocr_bbox': ocr_bbox, 'obj_bbox': obj_bbox}
        )
        fwd_results.update(mmt_results)

    def _forward_output(self, sample_list, fwd_results, sample=False):
        mmt_dec_output = fwd_results['mmt_dec_output']
        mmt_ocr_output = fwd_results['mmt_ocr_output']
        ocr_mask = fwd_results['ocr_mask']
        # print('mmt_dec_output size', mmt_dec_output.size()) # 64 * 30 * 768
        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(
            mmt_dec_output, mmt_ocr_output, ocr_mask
        )
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)  # 64*30*7903
        if sample:
            fwd_results['scores_sample'] = scores
        else:
            fwd_results['scores'] = scores

    def _beam_step(self, scores, beam_size, t, beam_seq, beam_seq_scores, beam_scores_sum):
        # INPUTS:
        # scores: probabilities augmented after diversity t =0 1*vocab_size; t>0 beam_size*vocab_size
        # beam_size: obvious
        # t        : time instant
        # beam_seq : tensor contanining the beams
        # beam_seq_scores: tensor contanining the beam scores
        # beam_scores_sum: tensor contanining joint scores
        # OUPUTS:
        # beam_seq : tensor containing the word indices of the decoded captions
        # beam_seq_scores : log-probability of each decision made, same size as beam_seq
        # beam_scores_sum : joint log-probability of each beam

        # scores = torch.log(torch.sigmoid(scores)) # scores:batch * vocabsize
        sigmoid_scores = torch.sigmoid(scores)
        # print('max:', torch.max(sigmoid_scores, dim=1))
        for c in range(sigmoid_scores.size(1)):
            if c >= 7853:  # 7853 is the vocab size of the old vizwiz vocab
                # print('raw:', sigmoid_scores[:, c])
                enhanced_sigmoid_scores = sigmoid_scores[:, c] * 1.5  # size=5
                sigmoid_scores[:, c], _ = torch.min(torch.cat((enhanced_sigmoid_scores.unsqueeze(1), torch.Tensor([[1.0]]*beam_size).cuda()), dim=1), dim=1)
                # print('enhance', sigmoid_scores[:, c])
        scores = torch.log(sigmoid_scores)
        ys, ix = torch.sort(scores, 1, True)  # each row, from big to small
        # print('ys:', ys.size())
        candidates = []
        cols = min(beam_size, ys.size(1))
        rows = beam_size
        if t == 0:
            rows = 1
        for c in range(cols):  # for each column (word, essentially)
            for q in range(rows):  # for each beam expansion
                # compute logprob of expanding beam q with word in (sorted) position c
                local_score = ys[q, c].cpu()
                candidate_scores = beam_scores_sum[q] + local_score
                candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_scores, 'r': local_score})

        candidates = sorted(candidates, key=lambda x: -x['p'])
        # print('candidates', candidates)

        # new_state = [_.clone() for _ in state]
        # beam_seq_prev, beam_seq_scores_prev
        # if t >= 1:
        # we''ll need these as reference when we fork beams around
        beam_seq_prev = beam_seq[:, :t+1].clone()
        beam_seq_scores_prev = beam_seq_scores[:, :t+1].clone()

        for vix in range(beam_size):
            v = candidates[vix]
            # fork beam index q into index vix
            # if t >= 1:
            beam_seq[vix, :t+1] = beam_seq_prev[v['q'], :]
            beam_seq_scores[vix, :t+1] = beam_seq_scores_prev[v['q'], :]
            # rearrange recurrent states
            """for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                new_state[state_ix][:, vix] = state[state_ix][:, v['q']]  # dimension one is time step"""
            # append new end terminal at the end of this beam
            beam_seq[vix, t+1] = v['c']  # c'th word is the continuation
            beam_seq_scores[vix, t+1] = v['r']  # the raw logprob here
            beam_scores_sum[vix] = v['p']  # the new (sum) logprob along this beam
            # state = new_state
        # print('beam_scores_sum', beam_scores_sum)
        # print('beam_sqe', beam_seq)
        # print('beam_seq_scores', beam_seq_scores)
        # print('bea,_scores_sum', beam_scores_sum)
        return beam_seq, beam_seq_scores, beam_scores_sum

    def _forward_mmt_and_output(self, sample_list, fwd_results, decoding_strategy='greedy'):
        # if self.training and not self.sc_learning:
        if self.training and not self.sc_learning:
            # print('original training')
            fwd_results['prev_inds'] = sample_list.train_prev_inds.clone()
            self._forward_mmt(sample_list, fwd_results)
            self._forward_output(sample_list, fwd_results)
        else:
            # decoding_strategy = 'beam'
            # print('inferences strategy', decoding_strategy)
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results['prev_inds'] = torch.zeros_like(
                sample_list.train_prev_inds
            )
            # anwen hu 2021/3/26: used for avoiding repeating same ocr tokens
            fwd_results['prev_ocrs_distri'] = torch.zeros(
                [sample_list.train_prev_inds.size(0), self.config.classifier.ocr_max_num]).cuda()
            # print('self.answer_processor.BOS_IDX', self.answer_processor.BOS_IDX)
            fwd_results['prev_inds'][:, 0] = self.answer_processor.BOS_IDX
            if self.sc_learning:
                if decoding_strategy == 'sample':
                    fwd_results['sampleseqLogprobs'] = torch.zeros_like(
                        sample_list.train_prev_inds, dtype=torch.double
                    )
            # greedy decoding at test time
            # TODO: add beam search
            # anwen hu 2020/5/16 move text bert out
            self._forward_mmt_txt(sample_list, fwd_results)
            if decoding_strategy == 'beam':
                beam_size = self.beam_size
                beam_seq = torch.LongTensor(beam_size, dec_step_num).zero_()
                beam_seq[:, 0] = self.answer_processor.BOS_IDX
                fwd_results['prev_inds'] = beam_seq.cuda()
                beam_seq_scores = torch.FloatTensor(beam_size, dec_step_num).zero_()
                beam_scores_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
                done_beams = []
            for t in range(dec_step_num):
                # print('decoding step', t)
                # self._forward_mmt(sample_list, fwd_results)
                self._forward_mmt_mmt(sample_list, fwd_results)
                if decoding_strategy == 'sample':
                    self._forward_output(sample_list, fwd_results, sample=True)
                else:
                    self._forward_output(sample_list, fwd_results, sample=False)
                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                if decoding_strategy == 'greedy':
                    # print('step', t, ' m4c.py old prev_inds[7]', fwd_results['prev_inds'][7, 1:])
                    # print('step', t, ' m4c.py argmax_inds[7]', argmax_inds[7, :-1])
                    """argmax_inds = fwd_results["scores"].argmax(dim=-1)  # 64*30*dim > 64 * 30
                    fwd_results['prev_inds'][:, 1:] = argmax_inds[:, :-1] # prev_inds may be changed by following step"""
                    if self.mmt_config.avoid_repeat:
                        t_scores = torch.sigmoid(fwd_results["scores"][:, t, :])  # 64*30*dim > 64 * dim
                        t_input_index = fwd_results['prev_inds'][:, t].unsqueeze(1)  # 64 * 1
                        t_input_onehot = torch.zeros(t_scores.size(0), t_scores.size(1), dtype=torch.float32).cuda()
                        t_input_onehot = t_input_onehot.scatter_(1, t_input_index, 1)  # 64*dim
                        # pre ocr inds won't be focused again
                        t_input_onehot = t_input_onehot + torch.nn.functional.pad(fwd_results['prev_ocrs_distri'],
                                                                                  [self.fixed_vocab_size, 0, 0, 0],
                                                                                  'constant')
                        t_input_onehot = torch.clamp(t_input_onehot, max=1.0)
                        t_scores = torch.mul(t_scores, (1.0 - t_input_onehot))  # mask previous id
                        argmax_inds = t_scores.argmax(dim=-1)  # 64
                        # save ocr distribution
                        argmax_inds_onehot = torch.zeros(t_scores.size(0), t_scores.size(1),
                                                         dtype=torch.float32).cuda().scatter_(1, argmax_inds.unsqueeze(1),
                                                                                              1)
                        argmax_ocr_onehot = argmax_inds_onehot[:, -self.config.classifier.ocr_max_num:]
                        fwd_results['prev_ocrs_distri'] = fwd_results['prev_ocrs_distri'] + argmax_ocr_onehot
                        if t != dec_step_num - 1:
                            # fwd_results['prev_inds'][:, t+1] = argmax_inds[:, t]
                            fwd_results['prev_inds'][:, t + 1] = argmax_inds
                    else:
                        argmax_inds = fwd_results["scores"].argmax(dim=-1)
                        if t != dec_step_num - 1:
                            fwd_results['prev_inds'][:, t + 1] = argmax_inds[:, t]

                    # anwen hu prev_inds won't be changed by following step, slightly worse than method above
                    """if t != dec_step_num - 1:
                        fwd_results['prev_inds'][:, t+1] = argmax_inds[:, t]"""
                    # print('step', t, 'greedy prev_inds[0]:', fwd_results['prev_inds'][0])
                elif decoding_strategy == 'sample':
                    # TODO: prev_inds may be changed by following step
                    if t != dec_step_num - 1:
                        logprobs = F.log_softmax(fwd_results["scores_sample"][:, t], dim=-1)  # 64*30*dim > 64*dim
                        # sorted_logprobs,_  = torch.sort(logprobs, dim=1, descending=True)
                        # print('top sample logprobs', sorted_logprobs[:, :5])
                        # print(logprobs.size())
                        argmax_inds = torch.distributions.Categorical(logits=logprobs.detach()).sample().unsqueeze(
                            1)  # 64 * 1
                        sampleLogprobs = logprobs.gather(1, argmax_inds)  # 64 * 1
                        # print(argmax_inds.size())
                        fwd_results['prev_inds'][:, t + 1] = argmax_inds[:, 0]
                        fwd_results['sampleseqLogprobs'][:, t + 1] = sampleLogprobs.view(-1)

                elif decoding_strategy == 'beam':
                    if t != dec_step_num - 1:
                        beam_seq, beam_seq_scores, beam_scores_sum = \
                            self._beam_step(fwd_results['scores'][:, t],
                                            beam_size,
                                            t,
                                            beam_seq,
                                            beam_seq_scores,
                                            beam_scores_sum)
                        # see if there is sentence finished
                        for vix in range(beam_size):
                            # if time's up... or if end token is reached then copy beams
                            if beam_seq[vix, t+1] == 2 or t+1 == dec_step_num-1:
                                final_beam = {
                                    'seq': beam_seq[vix, 1:].clone(),
                                    'seq_scores': beam_seq_scores[vix, 1:].clone(),
                                    'score': beam_scores_sum[vix].clone()
                                }
                                # print('final beam', final_beam)
                                done_beams.append(final_beam)
                                # don't continue beams from finished sequences
                                beam_scores_sum[vix] = -1000
                        fwd_results['prev_inds'] = beam_seq.cuda()

                else:
                    print('unknown decoding strategy')
                    exit(0)
            if decoding_strategy == 'beam':
                # exit(0)
                # sorted_beams = sorted(done_beams, key=lambda x: x['p'], reverse=True)
                fwd_results['final_beams'] = sorted(done_beams, key=lambda x: -x['score'])[:beam_size]
                fwd_results['scores'] = fwd_results['scores'][0].unsqueeze(0)
                # print('best scores size', fwd_results['scores'].size())

    def _forward_mmt_and_output_sample(self, sample_list, fwd_results):
        self._forward_mmt(sample_list, fwd_results)
        self._forward_output(sample_list, fwd_results, sample=True)

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer_attributes.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * m['lr_scale']
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output


class MMT(BertPreTrainedModelHu):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        #  anwen hu 2020/10/07 whether to use box att before multimodal transformer
        self.init_use_bbox_att = config.init_use_bbox_att
        #  anwen hu 2020/10/13 whether to use box att during multimodal transformer
        self.mid_use_bbox_att = config.mid_use_bbox_att
        # print('m4c.py MMT.init_use_bbox_att', self.init_use_bbox_att)
        # print('m4c.py MMT.mid_use_bbox_att', self.mid_use_bbox_att)
        if self.mid_use_bbox_att:
            self.encoder = BertEncoderMid(config)
        elif self.init_use_bbox_att:
            self.encoder = BertEncoderPre(config)
        else:
            self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self,
                txt_emb,
                txt_mask,
                obj_emb,
                obj_mask,
                ocr_emb,
                ocr_mask,
                fixed_ans_emb,
                prev_inds, bbox_coor):
        # anwen hu 2020/10/7 add boox_coor containing obj_box, ocr_box
        # print('m4c.py MMT bbox_coor', bbox_coor)
        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary\
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)  # batch * L_seq(30) * D_hid

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(
            dec_emb.size(0),
            dec_emb.size(1),
            dtype=torch.float32,
            device=dec_emb.device
        )  # batch * L_seq(30)
        encoder_inputs = torch.cat(
            [txt_emb, obj_emb, ocr_emb, dec_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask, ocr_mask, dec_mask],
            dim=1
        )

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        # print('m4c.py text_end', txt_end)
        obj_begin = txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # batch * 1 * 1 * N
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )  #  batch * 1 * N * N
        # decoding step elements can attend to themselves in a causal manner
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = \
            _get_causal_mask(dec_max_num, encoder_inputs.device)

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        """encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )"""
        # # Anwen Hu 2020/10/6 revise self.encoder input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
        bbox_coor['obj_begin'] = obj_begin
        bbox_coor['ocr_end'] = ocr_end
        if self.init_use_bbox_att or self.mid_use_bbox_att:
            encoder_outputs = self.encoder(
                encoder_inputs,
                extended_attention_mask,
                head_mask,
                bbox_coor,
            )
        else:
            encoder_outputs = self.encoder(
                encoder_inputs,
                extended_attention_mask,
                head_mask=head_mask
            )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            'mmt_seq_output': mmt_seq_output,
            'mmt_txt_output': mmt_txt_output,
            'mmt_ocr_output': mmt_ocr_output,
            'mmt_dec_output': mmt_dec_output,
        }
        return results


class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2  # batch * N_ocr
        extended_attention_mask = extended_attention_mask.unsqueeze(1)  # batch * 1 * N_ocr

        query_layer = self.query(query_inputs)  # batch * N_seq * D_q
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)  # batch * N_ocr * _D_q

        scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        ) # batch * N_seq * N_or
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        # print('ans_emb(m4c.py)', ans_emb.size()) # 7853 * 768
        # print('oc_emb(m4c.py)', ocr_emb.size()) # 64 * 50 * 768
        # print('prev_inds(m4c.py)', prev_inds.size()) # 64 * 30
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)  # batch * N_ans * D_hid
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)  # batch * (N_ans + N_ocr) * D_hid
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)  # batch * 30 * D_hid

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=ocr_emb.device
        )  # [0,1,2, ..., seq_length-1]
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # batch * 30
        position_embeddings = self.position_embeddings(position_ids)  # batch * 30 * D_hid
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # batch * 30 * D_hid
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings  # batch * 30 * D_hid

        return dec_emb


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)  # batch * max_num
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))  # lt:less than
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results
