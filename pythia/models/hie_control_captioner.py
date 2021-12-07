import torch
from pythia.common.registry import registry
from pythia.models.m4c import M4C, MMT, TextBert, OcrPtrNet, ClassifierLayer, _get_mask, PrevPredEmbeddings, _get_causal_mask
from torch import nn
from pytorch_transformers_hu.modeling_bert import BertEncoderMid
from pytorch_transformers_hu.modeling_bert import BertEncoderPre, CrossBertEncoder
from pytorch_transformers_hu.modeling_bert import BertConfig as BertConfigHu
# fix the bug that BertConfigHu is not a instance of PretrainedConfig
from pytorch_transformers_hu.modeling_bert import BertPreTrainedModel as BertPreTrainedModelHu
from pytorch_transformers_hu.modeling_bert import BertLayerObjOcr
import torch.nn.functional as F
from pythia.models.base_model import BaseModel
from pythia.modules.encoders import ImageEncoder



from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)


@registry.register_model("hie_control_captioner")
class HieContorlCaptioner(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.vqammt_config = BertConfigHu(**self.config.vqammt)
        self.capmmt_config = BertConfigHu(**self.config.capmmt)
        self._datasets = registry.get("config").datasets.split(",")
        self.sc_learning = False
        self.remove_unk_in_pred = self.config.remove_unk_in_pred
        self.drop_obj = self.vqammt_config.drop_obj
        self.drop_ocr = self.vqammt_config.drop_ocr

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        if self.vqammt_config.init_use_bbox_att_purevision:
            self._build_obj_ocr_encoding()
        else:
            self._build_obj_encoding()
            self._build_ocr_encoding()
        self._build_vqammt()
        self._build_vqaoutput()
        self._build_capmmt()
        self._build_capoutput()

    def _build_vqammt(self):
        self.vqammt = VQAMMT(self.vqammt_config)
        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append({
            'module': self.vqammt,
            'lr_scale': self.config.lr_scale_mmt,
        })

    def _build_vqaoutput(self):
        # dynamic OCR-copying scores with pointer network
        self.vqa_ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)
        # TODO: answer processor?
        """self.answer_processor = registry.get(
            self._datasets[0] + "_answer_processor"
        )"""
        # print('m4c.py self.answer_processor', self.answer_processor)  pythia.datasets.processors.Processor

    def _build_capmmt(self):
        self.capmmt = CAPMMT(self.capmmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append({
            'module': self.capmmt,
            'lr_scale': self.config.lr_scale_mmt,
        })

    def _build_capoutput(self):
        # dynamic OCR-copying scores with pointer network
        self.cap_ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)

        # fixed answer vocabulary scores
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        num_choices -= self.config.classifier.ocr_max_num
        self.fixed_vocab_size = num_choices
        self.classifier = ClassifierLayer(
            self.config["classifier"]["type"],
            in_dim=self.capmmt_config.hidden_size,
            out_dim=num_choices,
            **self.config["classifier"]["params"]
        )
        self.answer_processor = registry.get(
            self._datasets[0] + "_answer_processor"
        )
        # print('m4c.py self.answer_processor', self.answer_processor)  pythia.datasets.processors.Processor\

    def _forward_txt_encoding(self, sample_list, fwd_results):
        # simple caption
        fwd_results['simple_cap_txt_inds'] = sample_list.simple_cap_text
        # binary mask of valid text (simple caption  words) vs padding
        fwd_results['simple_cap_txt_mask'] = _get_mask(
            sample_list.simple_cap_text_len, sample_list.simple_cap_text.size(1)
        )
        # auto question (concatenated)
        fwd_results['auto_q_txt_inds'] = sample_list.auto_q_text
        # binary mask of valid text (auto question words) vs padding
        fwd_results['auto_q_txt_mask'] = _get_mask(
            sample_list.auto_q_text_len, sample_list.auto_q_text.size(1)
        )
        # auto question (separate)
        fwd_results['auto_sep_q_txt_inds'] = sample_list.auto_sep_q_text # batch * 50 (5*10)
        fwd_results['auto_sep_q_txt_mask'] = sample_list.auto_sep_q_text_mask  # batch * 50 (5*10)
        fwd_results['auto_q_global_txt_mask'] = _get_mask(sample_list.auto_sep_q_num, 5) # 5: max question num


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
        if self.drop_obj:
            attention_mask = ocr_mask
            obj_ocr_hidden_states = ocr_feat
        elif self.drop_ocr:
            attention_mask = obj_mask
            obj_ocr_hidden_states = obj_feat
        else:
            attention_mask = torch.cat([obj_mask, ocr_mask], dim=1)
            obj_ocr_hidden_states = torch.cat([obj_feat, ocr_feat], dim=1)
        objocr_num = attention_mask.size(1)  # N
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # batch * 1 * 1 * N
        extended_attention_mask = extended_attention_mask.repeat(1, 1, objocr_num, 1)  # batch * 1 * N * N
        head_mask = [None] * self.vqammt_config.objocr_pre_encoding_layers
        obj_bbox = sample_list.obj_bbox_coordinates[:, :obj_fc6.shape[1], :]  # batch * features_max_len * 4
        ocr_bbox = sample_list.ocr_bbox_coordinates
        bbox_coor = {'ocr_bbox': ocr_bbox, 'obj_bbox': obj_bbox}

        # print('m4c.py obj_end', obj_end)
        # feed to obj and ocr encoding layers
        for i, layer_module in enumerate(self.objocr_bert):
            # Anwen Hu 2020/10/6 revise layer input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
            # print('pytorch_transformers_hu\modeling_bert.py BertEncoderPre bbox_ocr', bbox_coor)
            obj_ocr_layer_outputs = layer_module(obj_ocr_hidden_states, extended_attention_mask, head_mask[i], bbox_coor)
            obj_ocr_hidden_states = obj_ocr_layer_outputs[0]

        if self.drop_obj:
            obj_hidden_states = obj_feat
            ocr_hidden_states = obj_ocr_hidden_states
        elif self.drop_ocr:
            obj_hidden_states = obj_ocr_hidden_states
            ocr_hidden_states = ocr_feat
        else:
            obj_end = obj_mask.size(-1)
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
        assert self.capmmt_config.hidden_size == self.vqammt_config.hidden_size
        assert self.capmmt_config.hidden_size == TEXT_BERT_HIDDEN_SIZE
        if self.capmmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            self.writer.write(
                'Projecting text_bert output to {} dim'.format(
                    self.capmmt_config.hidden_size
                )
            )
            self.simple_cap_text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.capmmt_config.hidden_size
            )
            self.auto_q_text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.vqammt_config.hidden_size
            )
            # self.text_drop = nn.Dropout(self.config.text.dropout_prob)
        else:
            self.simple_cap_text_bert_out_linear = nn.Identity()
            self.auto_q_text_bert_out_linear = nn.Identity()
            # self.text_drop = nn.Dropout(self.config.text.dropout_prob)
            """if self.vqammt_config.sep_question_emb:
                self.auto_seq_q_text_gate = nn.Linear(TEXT_BERT_HIDDEN_SIZE, 1)"""


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
            self.config.obj.mmt_in_dim, self.vqammt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(
            4, self.vqammt_config.hidden_size
        )

        self.obj_feat_layer_norm = BertLayerNorm(self.vqammt_config.hidden_size)
        self.obj_bbox_layer_norm = BertLayerNorm(self.vqammt_config.hidden_size)
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
            self.config.ocr.mmt_in_dim, self.vqammt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(
            4, self.vqammt_config.hidden_size
        )

        self.ocr_feat_layer_norm = BertLayerNorm(self.vqammt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.vqammt_config.hidden_size)
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
            self.config.obj.mmt_in_dim, self.vqammt_config.hidden_size
        )
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.vqammt_config.hidden_size, self.vqammt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(
            4, self.vqammt_config.hidden_size
        )

        self.obj_feat_layer_norm = BertLayerNorm(self.vqammt_config.hidden_size)
        self.obj_bbox_layer_norm = BertLayerNorm(self.vqammt_config.hidden_size)
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
            self.config.obj.mmt_in_dim, self.vqammt_config.hidden_size
        )
        # 1672 = 300 (FastText) + 604 (PHOC) + 768 (ocr hidden)
        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            1672, self.vqammt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(
            4, self.vqammt_config.hidden_size
        )

        self.ocr_feat_layer_norm = BertLayerNorm(self.vqammt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.vqammt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

        # for fuse
        self.objocr_bert = nn.ModuleList([BertLayerObjOcr(self.vqammt_config) for _ in range(self.vqammt_config.objocr_pre_encoding_layers)])

    def forward(self, sample_list, **kwargs):

        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        # fetch simple caption inds and questions inds
        self._forward_txt_encoding(sample_list, fwd_results)
        if self.vqammt_config.init_use_bbox_att_purevision:
            self._forward_objocr_encoding(sample_list, fwd_results)
        else:
            self._forward_obj_encoding(sample_list, fwd_results)
            self._forward_ocr_encoding(sample_list, fwd_results)
        """print('pythia/models/hie_control_captioner.py simple_cap_txt: ', fwd_results['simple_cap_txt_inds'].size())
        print('pythia/models/hie_control_captioner.py auto_q_txt: ', fwd_results['auto_q_txt_inds'].size())
        print('pythia/models/hie_control_captioner.py obj_mmt_in: ', fwd_results['obj_mmt_in'].size())
        print('pythia/models/hie_control_captioner.py ocr_mmt_in: ', fwd_results['ocr_mmt_in'].size())"""

        self._forward_vqammt_and_vqaoutput(sample_list, fwd_results)
        """if 'sc' in kwargs and kwargs['sc']:
            self.sc_learning = True
            with torch.no_grad():
                self._forward_capmmt_and_capoutput(sample_list, fwd_results, decoding_strategy='greedy')  # first greedy
                self._forward_capmmt_and_capoutput(sample_list, fwd_results, decoding_strategy='sample')  # then sample, so prev_inds store the inds of sample decoding

            self._forward_capmmt_and_capoutput_sample(sample_list, fwd_results)
            results = {"scores": fwd_results["scores"], "scores_sample": fwd_results["scores_sample"],
                       'sampleseqLogprobs': fwd_results["sampleseqLogprobs"], 'prev_inds': fwd_results["prev_inds"]}
            # print('m4c.py sampleseqLogprobs', fwd_results["sampleseqLogprobs"])
            return results
        else:"""
        if 'beam_size' in kwargs and kwargs['beam_size'] > 0:
            self.beam_size = kwargs['beam_size']
            self._forward_capmmt_and_capoutput(sample_list, fwd_results, decoding_strategy='beam')
        else:
            self._forward_capmmt_and_capoutput(sample_list, fwd_results)
        # keep scores(cap scores) and vqa scores in the forward pass results
        # results = {"scores": fwd_results["scores"], "vqa_scores": fwd_results["vqa_scores"]}
        results = {"scores": fwd_results["scores"], 'prev_inds': fwd_results["prev_inds"]}
        if 'beam_size' in kwargs and kwargs['beam_size'] > 0:
            results['final_beams'] = fwd_results['final_beams']
        return results

    def _forward_vqammt_and_vqaoutput(self, sample_list, fwd_results):
        self._forward_vqammt(sample_list, fwd_results)
        # self._forward_vqaoutput(fwd_results)

    def _forward_vqammt(self, sample_list, fwd_results):
        auto_q_text_bert_out = self.text_bert(
            txt_inds=fwd_results['auto_q_txt_inds'],
            txt_mask=fwd_results['auto_q_txt_mask']
        ) # batch * 20 * hid
        auto_q_txt_emb = self.auto_q_text_bert_out_linear(auto_q_text_bert_out) # identity layer
        """if self.training: # text dropout only used in training
            auto_q_txt_emb = self.text_drop(auto_q_txt_emb)"""
        fwd_results['auto_q_txt_emb'] = auto_q_txt_emb
        fwd_results['auto_q_txt_mask'] = fwd_results['auto_q_txt_mask']
        vqammt_results = self.vqammt(
            que_emb=fwd_results['auto_q_txt_emb'],
            que_mask=fwd_results['auto_q_txt_mask'],
            obj_emb=fwd_results['obj_mmt_in'],
            obj_mask=fwd_results['obj_mask'],
            ocr_emb=fwd_results['ocr_mmt_in'],
            ocr_mask=fwd_results['ocr_mask'],
        )
        fwd_results.update(vqammt_results)

    def _forward_vqaoutput(self, fwd_results):
        vqammt_ans_output = fwd_results['vqammt_ans_output']  # batch * max_que_num * hid
        vqammt_ocr_output = fwd_results['vqammt_ocr_output']  # batch * max_ocr_length * hid
        ocr_mask = fwd_results['ocr_mask']
        vqa_dynamic_ocr_scores = self.vqa_ocr_ptr_net(
            vqammt_ans_output, vqammt_ocr_output, ocr_mask
        ) # batch * max_que_num * max_ocr_length
        # print('pythia/models/hie_control_captioner.py vqa_dynamic_ocr_scores:', vqa_dynamic_ocr_scores.size())
        fwd_results['vqa_scores'] = vqa_dynamic_ocr_scores

    def _forward_capmmt_and_capoutput(self, sample_list, fwd_results, decoding_strategy='greedy'):
        # if self.training and not self.sc_learning:
        if self.training and not self.sc_learning:
            # print('original training')
            fwd_results['prev_inds'] = sample_list.train_prev_inds.clone()
            self._forward_capmmt(sample_list, fwd_results)
            self._forward_capoutput(fwd_results)
        else:
            # decoding_strategy = 'beam'
            # print('inferences strategy', decoding_strategy)
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results['prev_inds'] = torch.zeros_like(
                sample_list.train_prev_inds
            )
            # anwen hu 2021/3/26: used for avoiding repeating same ocr tokens
            fwd_results['prev_ocrs_distri'] = torch.zeros([sample_list.train_prev_inds.size(0), self.config.classifier.ocr_max_num]).cuda()
            # print('self.answer_processor.BOS_IDX', self.answer_processor.BOS_IDX)
            fwd_results['prev_inds'][:, 0] = self.answer_processor.BOS_IDX
            if self.sc_learning:
                if decoding_strategy == 'sample':
                    fwd_results['sampleseqLogprobs'] = torch.zeros_like(
                        sample_list.train_prev_inds, dtype=torch.double
                    )
            # greedy decoding at test time
            # anwen hu 2020/5/16 move text bert out
            self._forward_capmmt_txt(fwd_results)
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
                self._forward_capmmt_mmt(sample_list, fwd_results)
                if decoding_strategy == 'sample':
                    self._forward_capoutput(fwd_results, sample=True)
                else:
                    self._forward_capoutput(fwd_results, sample=False)
                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                if decoding_strategy == 'greedy':
                    # print('step', t, ' m4c.py old prev_inds[7]', fwd_results['prev_inds'][7, 1:])
                    # print('step', t, ' m4c.py argmax_inds[7]', argmax_inds[7, :-1])
                    """fwd_results['scores'] = torch.sigmoid(fwd_results['scores'])
                    pre_argmax_inds = fwd_results["scores"].argmax(dim=-1)[:, :-1]  # 64*30*dim > 64 * 29
                    pre_argmax_inds = pre_argmax_inds.reshape(-1, 1) #  (64*29) * 1
                    pre_argmax_inds_onehot = torch.zeros(pre_argmax_inds.size(0), fwd_results["scores"].size(-1), dtype=torch.float32).cuda()
                    pre_argmax_inds_onehot = pre_argmax_inds_onehot.scatter_(1, pre_argmax_inds, 1)  # 64*29) * dim
                    pre_argmax_inds_onehot = pre_argmax_inds_onehot.reshape(fwd_results["scores"].size(0),
                                                                            fwd_results["scores"].size(1)-1,
                                                                            fwd_results["scores"].size(2)) # 64 * 29 * dim
                    pre_argmax_inds_onehot = torch.nn.functional.pad(pre_argmax_inds_onehot, [0, 0, 1, 0, 0, 0],
                                                                     "constant")
                    fwd_results["scores"] = torch.mul(fwd_results["scores"], 1.0 - pre_argmax_inds_onehot)
                    argmax_inds = fwd_results["scores"].argmax(dim=-1)
                    fwd_results['prev_inds'][:, 1:] = argmax_inds[:, :-1]  # prev_inds may be changed by following step"""
                    # anwen hu prev_inds won't be changed by following step, slightly worse than method above
                    if self.capmmt_config.avoid_repeat:
                        t_scores = torch.sigmoid(fwd_results["scores"][:, t, :])  # 64*30*dim > 64 * dim
                        t_input_index = fwd_results['prev_inds'][:, t].unsqueeze(1)  # 64 * 1
                        t_input_onehot = torch.zeros(t_scores.size(0), t_scores.size(1), dtype=torch.float32).cuda()
                        t_input_onehot = t_input_onehot.scatter_(1, t_input_index, 1) # 64*dim
                        # pre ocr inds won't be focused again
                        t_input_onehot = t_input_onehot + torch.nn.functional.pad(fwd_results['prev_ocrs_distri'], [self.fixed_vocab_size, 0, 0, 0], 'constant')
                        t_input_onehot = torch.clamp(t_input_onehot, max=1.0)
                        t_scores = torch.mul(t_scores, (1.0 - t_input_onehot))  # mask previous id
                        argmax_inds = t_scores.argmax(dim=-1)  # 64
                        # save ocr distribution
                        argmax_inds_onehot = torch.zeros(t_scores.size(0), t_scores.size(1), dtype=torch.float32).cuda().scatter_(1, argmax_inds.unsqueeze(1), 1)
                        argmax_ocr_onehot= argmax_inds_onehot[:, -self.config.classifier.ocr_max_num:]
                        fwd_results['prev_ocrs_distri'] = fwd_results['prev_ocrs_distri'] + argmax_ocr_onehot
                        if t != dec_step_num - 1:
                            # fwd_results['prev_inds'][:, t+1] = argmax_inds[:, t]
                            fwd_results['prev_inds'][:, t + 1] = argmax_inds
                    else:
                        argmax_inds = fwd_results["scores"].argmax(dim=-1)
                        if t != dec_step_num - 1:
                            fwd_results['prev_inds'][:, t+1] = argmax_inds[:, t]
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
                            if beam_seq[vix, t + 1] == 2 or t + 1 == dec_step_num - 1:
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

    def _forward_capmmt(self, sample_list, fwd_results):
        # forward the text BERT layers with simple caption as input
        self._forward_capmmt_txt(fwd_results)
        self._forward_capmmt_mmt(sample_list, fwd_results)

    def _forward_capmmt_txt(self, fwd_results):
        simple_cap_text_bert_out = self.text_bert(
            txt_inds=fwd_results['simple_cap_txt_inds'],
            txt_mask=fwd_results['simple_cap_txt_mask']
        )
        simple_cap_txt_emb = self.simple_cap_text_bert_out_linear(simple_cap_text_bert_out)
        """if self.training: # text dropout only used in training
            simple_cap_txt_emb = self.text_drop(simple_cap_txt_emb)"""
        fwd_results['simple_cap_txt_emb'] = simple_cap_txt_emb

    def _forward_capmmt_mmt(self, sample_list, fwd_results):
        # anwen hu 2020/10/7 add bbox_info containing obj_box, ocr_box
        ocr_bbox = sample_list.ocr_bbox_coordinates
        obj_bbox = sample_list.obj_bbox_coordinates
        # print('m4c.py forward mmt ocr_bbox shape', ocr_bbox.shape)
        # print('m4c.py forward mmt obj_bbox shape', obj_bbox.shape)
        # print('models/m4c_cntrol_captioner.py train txt_emb', fwd_results['txt_emb'].size())
        # print('models/m4c_cntrol_captioner.py val txt_mask', fwd_results['txt_mask'].size())
        if self.capmmt_config.use_vqa_obj and not self.drop_obj:
            obj_feat = fwd_results['vqammt_obj_output']
            obj_mask = fwd_results['obj_mask']
        else:
            obj_feat = None
            obj_mask = None

        if self.capmmt_config.use_vqa_ocr and not self.drop_ocr:
            ocr_feat = fwd_results['vqammt_ocr_output']
            ocr_mask = fwd_results['ocr_mask']
        else:
            ocr_feat = None
            ocr_mask = None

        if self.capmmt_config.drop_auto_question:
            que_feat = None
            que_mask = None
        elif self.capmmt_config.use_raw_que and self.capmmt_config.use_vision_que:
            if self.capmmt_config.que_vision_txt_fuse_type == 'cat':
                que_feat = torch.cat([fwd_results['vqammt_que_output'], fwd_results['auto_q_txt_emb']], dim=1)
                que_mask = torch.cat([fwd_results['auto_q_txt_mask'], fwd_results['auto_q_txt_mask']], dim=-1)
            else:
                que_feat = fwd_results['vqammt_que_output'] + fwd_results['auto_q_txt_emb']
                que_mask = fwd_results['auto_q_txt_mask']
        elif self.capmmt_config.use_vision_que:
            que_feat = fwd_results['vqammt_que_output']
            que_mask = fwd_results['auto_q_txt_mask']
        elif self.capmmt_config.use_raw_que:
            que_feat = fwd_results['auto_q_txt_emb']
            que_mask = fwd_results['auto_q_txt_mask']
        else:
            que_feat = None
            que_mask = None


        if self.capmmt_config.drop_simple_cap:
            cap_emb = None
            cap_mask = None
        else:
            cap_emb = fwd_results['simple_cap_txt_emb']
            cap_mask = fwd_results['simple_cap_txt_mask']

        ocr_emb = fwd_results['ocr_mmt_in']
        # anwen hu 2021/4/12 for the first step of beam search
        if fwd_results['prev_inds'].size(0) != fwd_results['vqammt_obj_output'].size(0):
            assert fwd_results['vqammt_obj_output'].size(0) == 1
            que_feat = que_feat.repeat(self.beam_size, 1, 1)
            que_mask = que_mask.repeat(self.beam_size, 1)
            cap_emb = cap_emb.repeat(self.beam_size, 1, 1)
            cap_mask = cap_mask.repeat(self.beam_size, 1)
            obj_feat = obj_feat.repeat(self.beam_size, 1, 1)
            obj_mask = obj_mask.repeat(self.beam_size, 1)
            ocr_feat = ocr_feat.repeat(self.beam_size, 1, 1)
            ocr_mask = ocr_mask.repeat(self.beam_size, 1)
            ocr_emb = ocr_emb.repeat(self.beam_size, 1, 1)

        cap_mmt_results = self.capmmt(
            que_feat=que_feat,
            que_mask=que_mask,
            cap_emb=cap_emb,
            cap_mask=cap_mask,
            obj_feat=obj_feat,
            obj_mask=obj_mask,
            ocr_feat=ocr_feat,
            ocr_mask=ocr_mask,
            ocr_emb=ocr_emb,  # used for fetch previous word embedding
            fixed_ans_emb=self.classifier.module.weight,
            prev_inds=fwd_results['prev_inds'],
            bbox_coor={'ocr_bbox': ocr_bbox, 'obj_bbox': obj_bbox}
        )
        fwd_results.update(cap_mmt_results)

    def _forward_capoutput(self, fwd_results, sample=False):
        # pre inds as query
        mmt_dec_output = fwd_results['capmmt_dec_output']
        # ocr ourput from VQA as key
        if fwd_results['capmmt_ocr_output'] is not None:
            mmt_ocr_output = fwd_results['capmmt_ocr_output']
        else:
            mmt_ocr_output = fwd_results['vqammt_ocr_output']
        ocr_mask = fwd_results['ocr_mask']
        # print('mmt_dec_output size', mmt_dec_output.size()) # 64 * 30 * 768
        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.cap_ocr_ptr_net(
            mmt_dec_output, mmt_ocr_output, ocr_mask
        )
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)  # 64*30*7903
        if sample:
            fwd_results['scores_sample'] = scores
        else:
            fwd_results['scores'] = scores

        if self.remove_unk_in_pred:
            # avoid outputting <unk> in the generated captions
            fwd_results["scores"][..., self.answer_processor.UNK_IDX] = -1e10

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


class VQAMMT(BertPreTrainedModelHu): # No pre inds embedding
    def __init__(self, config):
        super().__init__(config)
        self.only_attend_vision = config.only_attend_vision
        self.drop_obj = config.drop_obj
        self.drop_ocr = config.drop_ocr
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self,
                que_emb,
                que_mask,
                obj_emb,
                obj_mask,
                ocr_emb,
                ocr_mask):
        # anwen hu 2020/10/7 add boox_coor containing obj_box, ocr_box
        # print('m4c.py MMT bbox_coor', bbox_coor)

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        # offsets of each modality in the joint embedding space
        que_max_num = que_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + que_max_num
        # print('m4c.py text_end', txt_end)
        if self.drop_obj:
            encoder_inputs = torch.cat([que_emb, ocr_emb], dim=1)
            attention_mask = torch.cat([que_mask, ocr_mask], dim=1)
            ocr_begin = txt_end
            ocr_end = txt_end + ocr_max_num
            vision_num = ocr_max_num
        elif self.drop_ocr:
            encoder_inputs = torch.cat([que_emb, obj_emb], dim=1)
            attention_mask = torch.cat([que_mask, obj_mask], dim=1)
            obj_begin = txt_end
            obj_end = txt_end + obj_max_num
            vision_num = obj_max_num
        else:
            encoder_inputs = torch.cat([que_emb, obj_emb, ocr_emb],dim=1)
            attention_mask = torch.cat([que_mask, obj_mask, ocr_mask],dim=1)
            obj_begin = txt_end
            obj_end = txt_end + obj_max_num
            ocr_begin = obj_end
            ocr_end = obj_end + ocr_max_num
            vision_num = obj_max_num + ocr_max_num

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
        )  #  batch * 1 * N * Ns

        if self.only_attend_vision:
            # don't attend to que token
            extended_attention_mask[:, :, txt_begin:txt_end, txt_begin:txt_end] = torch.zeros([que_max_num, que_max_num])
            # dont't revise vision feat
            extended_attention_mask[:, :, txt_end:to_seq_length, 0:to_seq_length] = torch.zeros([vision_num, to_seq_length])
            # print('hie_control_caption.py extended attention mask: ', extended_attention_mask[0, :, txt_begin, :])
            # print('hie_control_caption.py extended attention mask: ', extended_attention_mask[0, :, txt_end-1, :])
            # print('hie_control_caption.py extended attention mask: ', extended_attention_mask[0, :, txt_end, :])


        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        mmt_seq_output = encoder_outputs[0]
        mmt_que_output = mmt_seq_output[:, txt_begin:txt_end]
        if self.only_attend_vision:
            mmt_obj_output = obj_emb
            mmt_ocr_output = ocr_emb
        else:
            if self.drop_obj:
                mmt_obj_output = obj_emb
                mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
            elif self.drop_ocr:
                mmt_ocr_output = ocr_emb
                mmt_obj_output = mmt_seq_output[:, obj_begin:obj_end]
            else:
                mmt_obj_output = mmt_seq_output[:, obj_begin:obj_end]
                mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        # get answer token output
        # ans_index: batch * max_que_num > batch * max_que_num * hid_dim
        # mmt_ans_output = mmt_que_output.gather(1, ans_index.unsqueeze(dim=2).repeat(1, 1, mmt_seq_output.size(-1)))
        results = {
            'mmt_seq_output': mmt_seq_output,
            'vqammt_que_output': mmt_que_output,
            # 'vqammt_ans_output': mmt_ans_output,
            'vqammt_obj_output': mmt_obj_output,
            'vqammt_ocr_output': mmt_ocr_output,
        }
        return results


class VQACrossMMT(BertPreTrainedModelHu): # No pre inds embedding
    def __init__(self, config):
        super().__init__(config)
        self.que_encoder = CrossBertEncoder(config)
        self.vision_encoder = CrossBertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self,
                que_emb,
                que_mask,
                obj_emb,
                obj_mask,
                ocr_emb,
                ocr_mask):

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        vision_inputs = torch.cat([obj_emb, ocr_emb], dim=1)
        vision_attention_mask = torch.cat([obj_mask, ocr_mask], dim=1) # batch * N_v

        # offsets of each modality in the joint embedding space
        que_max_num = que_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        # txt_begin = 0
        # txt_end = txt_begin + que_max_num
        # print('m4c.py text_end', txt_end)
        obj_begin = 0
        obj_end = obj_begin + obj_max_num
        ocr_begin = obj_end
        ocr_end = obj_end + ocr_max_num
        # first step: use question feat as query, vision feat as key/value
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_vision_attention_mask = vision_attention_mask.unsqueeze(1).unsqueeze(2) # batch * 1 * 1 * N_v
        extended_vision_attention_mask = extended_vision_attention_mask.repeat(1, 1, que_max_num, 1)  #  batch * 1 * N_q * N_v
        # flip the mask, so that invalid attention pairs have -10000.
        extended_vision_attention_mask = (1.0 - extended_vision_attention_mask) * -10000.0
        assert not extended_vision_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers
        question_encoder_outputs = self.que_encoder(
            que_emb, # query: batch * N_q * D_hid
            vision_inputs, # key, value :batch * N_v * D_hid
            extended_vision_attention_mask, # batch * 1 * N_q * N_v
            head_mask=head_mask
        )
        mmt_que_output = question_encoder_outputs[0]
        # second step: use vision feat as query, question feat as key/value
        extended_que_attention_mask = que_mask.unsqueeze(1).unsqueeze(2) # # batch * 1 * 1 * N_q
        extended_que_attention_mask = extended_que_attention_mask.repeat(1, 1, obj_max_num+ocr_max_num, 1)  # batch * 1 * N_v * N_q
        # flip the mask, so that invalid attention pairs have -10000.
        extended_que_attention_mask = (1.0 - extended_que_attention_mask) * -10000.0
        assert not extended_que_attention_mask.requires_grad
        vision_encoder_outputs = self.vision_encoder(
            vision_inputs,  # query: batch * N_v * D_hid
            mmt_que_output,  # key, value :batch * N_q * D_hid
            extended_que_attention_mask,  # batch * 1 * N_v * N_q
            head_mask=head_mask
        )
        vision_output = vision_encoder_outputs[0]

        mmt_obj_output = vision_output[:, obj_begin:obj_end]
        mmt_ocr_output = vision_output[:, ocr_begin:ocr_end]

        results = {
            'vqammt_que_output': mmt_que_output,
            'vqammt_obj_output': mmt_obj_output,
            'vqammt_ocr_output': mmt_ocr_output,
        }
        return results


class CAPMMT(BertPreTrainedModelHu):
    def __init__(self, config):
        super().__init__(config)
        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self,
                que_feat,
                que_mask,
                cap_emb,
                cap_mask,
                obj_feat,
                obj_mask,
                ocr_feat,
                ocr_mask,
                ocr_emb,
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
        if que_feat is None:
            encoder_inputs = torch.cat([cap_emb, obj_feat, ocr_feat, dec_emb], dim=1)
            attention_mask = torch.cat([cap_mask, obj_mask, ocr_mask, dec_mask], dim=1)
            txt_max_num = cap_mask.size(-1)
            obj_begin = txt_max_num
            ocr_begin = cap_mask.size(-1) + obj_mask.size(-1)
            ocr_end = ocr_begin + ocr_mask.size(-1)
        elif cap_emb is None:
            encoder_inputs = torch.cat([que_feat, obj_feat, ocr_feat, dec_emb], dim=1)
            attention_mask = torch.cat([que_mask, obj_mask, ocr_mask, dec_mask], dim=1)
            txt_max_num = que_mask.size(-1)
            obj_begin = txt_max_num
            ocr_begin = que_mask.size(-1) + obj_mask.size(-1)
            ocr_end = ocr_begin + ocr_mask.size(-1)
        elif obj_feat is None:
            encoder_inputs = torch.cat([que_feat, cap_emb, ocr_feat, dec_emb], dim=1)
            attention_mask = torch.cat([que_mask, cap_mask, ocr_mask, dec_mask], dim=1)
            txt_max_num = que_mask.size(-1) + cap_mask.size(-1)
            ocr_begin = que_mask.size(-1) + cap_mask.size(-1)
            ocr_end = ocr_begin + ocr_mask.size(-1)
        elif ocr_feat is None:
            encoder_inputs = torch.cat([que_feat, cap_emb, obj_feat, dec_emb], dim=1)
            attention_mask = torch.cat([que_mask, cap_mask, obj_mask, dec_mask], dim=1)
            txt_max_num = que_mask.size(-1) + cap_mask.size(-1)
            obj_begin = txt_max_num
        else:
            encoder_inputs = torch.cat([que_feat, cap_emb, obj_feat, ocr_feat, dec_emb], dim=1)
            attention_mask = torch.cat([que_mask, cap_mask, obj_mask, ocr_mask, dec_mask], dim=1)
            txt_max_num = que_mask.size(-1) + cap_mask.size(-1)
            obj_begin = txt_max_num
            ocr_begin = que_mask.size(-1) + cap_mask.size(-1) + obj_mask.size(-1)
            ocr_end = ocr_begin + ocr_mask.size(-1)

        # print('/pythia/models/hie_control_captioner.py encoder_inputs', encoder_inputs.size())
        # offsets of each modality in the joint embedding space

        # obj_max_num = obj_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        # print('m4c.py text_end', txt_end)
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

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]
        if ocr_feat is not None:
            mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        else:
            mmt_ocr_output = None

        results = {
            'capmmt_seq_output': mmt_seq_output,
            'capmmt_txt_output': mmt_txt_output,
            'capmmt_ocr_output': mmt_ocr_output,
            'capmmt_dec_output': mmt_dec_output,

        }
        return results






