import torch
from pythia.common.registry import registry
from pythia.models.m4c import M4C, TextBert, _get_mask, PrevPredEmbeddings, _get_causal_mask
from torch import nn
from pytorch_transformers_hu.modeling_bert import BertEncoderMid
from pytorch_transformers_hu.modeling_bert import BertEncoderPre
from pytorch_transformers_hu.modeling_bert import BertConfig as BertConfigHu
# fix the bug that BertConfigHu is not a instance of PretrainedConfig
from pytorch_transformers_hu.modeling_bert import BertPreTrainedModel as BertPreTrainedModelHu
from pytorch_transformers_hu.modeling_bert import BertLayerObjOcr


from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)


@registry.register_model("m4c_control_captioner")
class M4CContorlCaptioner(M4C):
    def __init__(self, config):
        super().__init__(config)
        self.remove_unk_in_pred = self.config.remove_unk_in_pred

    def _forward_output(self, sample_list, fwd_results, sample=False):
        super()._forward_output(sample_list, fwd_results, sample)

        if self.remove_unk_in_pred:
            # avoid outputting <unk> in the generated captions
            fwd_results["scores"][..., self.answer_processor.UNK_IDX] = -1e10

        return fwd_results

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append({
            'module': self.mmt,
            'lr_scale': self.config.lr_scale_mmt,
        })

    def _forward_txt_encoding(self, sample_list, fwd_results):
        # simple caption
        fwd_results['simple_cap_txt_inds'] = sample_list.simple_cap_text
        # binary mask of valid text (simple caption  words) vs padding
        fwd_results['simple_cap_txt_mask'] = _get_mask(
            sample_list.simple_cap_text_len, sample_list.simple_cap_text.size(1)
        )
        # auto question
        fwd_results['auto_q_txt_inds'] = sample_list.auto_q_text
        # binary mask of valid text (auto question words) vs padding
        fwd_results['auto_q_txt_mask'] = _get_mask(
            sample_list.auto_q_text_len, sample_list.auto_q_text.size(1)
        )

    def _forward_mmt_txt(self, sample_list, fwd_results):
        if not self.mmt_config.drop_simple_cap:
            simple_cap_text_bert_out = self.text_bert(
                txt_inds=fwd_results['simple_cap_txt_inds'],
                txt_mask=fwd_results['simple_cap_txt_mask']
            )
            simple_cap_txt_emb = self.simple_cap_text_bert_out_linear(simple_cap_text_bert_out)
        if not self.mmt_config.drop_auto_question:
            # forward the text BERT layers with auto question as input
            auto_q_text_bert_out = self.text_bert(
                txt_inds=fwd_results['auto_q_txt_inds'],
                txt_mask=fwd_results['auto_q_txt_mask']
            )
            auto_q_txt_emb = self.auto_q_text_bert_out_linear(auto_q_text_bert_out)
            # print('models/m4c_cntrol_captioner.py simple_cap_txt_emb', simple_cap_txt_emb.size())
            # print('models/m4c_cntrol_captioner.py auto_q_txt_emb', auto_q_txt_emb.size())
        if not self.mmt_config.drop_auto_question and not self.mmt_config.drop_simple_cap:
            fwd_results['txt_emb'] = torch.cat([simple_cap_txt_emb, auto_q_txt_emb], dim=1)
            fwd_results['txt_mask'] = torch.cat([fwd_results['simple_cap_txt_mask'], fwd_results['auto_q_txt_mask']],
                                                dim=1)
        elif not self.mmt_config.drop_simple_cap:
            fwd_results['txt_emb'] = simple_cap_txt_emb
            fwd_results['txt_mask'] = fwd_results['simple_cap_txt_mask']
        elif not self.mmt_config.drop_auto_question:
            fwd_results['txt_emb'] = auto_q_txt_emb
            fwd_results['txt_mask'] = fwd_results['auto_q_txt_mask']
        else:
            fwd_results['txt_emb'] = None
            fwd_results['txt_mask'] = None

        # print('models/m4c_cntrol_captioner.py val txt_emb', fwd_results['txt_emb'].size())
        # print('models/m4c_cntrol_captioner.py val txt_mask', fwd_results['txt_mask'].size())

    def _forward_mmt_mmt(self, sample_list, fwd_results):
        # anwen hu 2020/5/20 for the first step of beam search
        if fwd_results['prev_inds'].size(0) != fwd_results['obj_mmt_in'].size(0):
            assert fwd_results['obj_mmt_in'].size(0) == 1
            if fwd_results['txt_emb'] is not None:
                fwd_results['txt_emb'] = fwd_results['txt_emb'].repeat(fwd_results['prev_inds'].size(0), 1, 1)
                fwd_results['txt_mask'] = fwd_results['txt_mask'].repeat(fwd_results['prev_inds'].size(0), 1)
            fwd_results['obj_mmt_in'] = fwd_results['obj_mmt_in'].repeat(fwd_results['prev_inds'].size(0), 1, 1)
            fwd_results['obj_mask'] = fwd_results['obj_mask'].repeat(fwd_results['prev_inds'].size(0), 1)
            fwd_results['ocr_mmt_in'] = fwd_results['ocr_mmt_in'].repeat(fwd_results['prev_inds'].size(0), 1, 1)
            fwd_results['ocr_mask'] = fwd_results['ocr_mask'].repeat(fwd_results['prev_inds'].size(0), 1)
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
            self.simple_cap_text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
            self.auto_q_text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.simple_cap_text_bert_out_linear = nn.Identity()
            self.auto_q_text_bert_out_linear = nn.Identity()

    def _forward_mmt(self, sample_list, fwd_results):
        # forward the text BERT layers with simple caption as input
        self._forward_mmt_txt(sample_list, fwd_results)
        self._forward_mmt_mmt(sample_list, fwd_results)


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
        # anwen hu 2021/7/15 drop obj feats or ocr feats
        self.drop_obj = config.drop_obj
        self.drop_ocr = config.drop_ocr

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
        if txt_emb is None:
            encoder_inputs = torch.cat(
                [obj_emb, ocr_emb, dec_emb],
                dim=1
            )
            attention_mask = torch.cat(
                [obj_mask, ocr_mask, dec_mask],
                dim=1
            )
            # offsets of each modality in the joint embedding space
            obj_max_num = obj_mask.size(-1)
            ocr_max_num = ocr_mask.size(-1)
            dec_max_num = dec_mask.size(-1)
            # print('m4c.py text_end', txt_end)
            obj_begin = 0
            ocr_begin = obj_max_num
            ocr_end = ocr_begin + ocr_max_num
        elif self.drop_obj:
            encoder_inputs = torch.cat(
                [txt_emb, ocr_emb, dec_emb],
                dim=1
            )
            attention_mask = torch.cat(
                [txt_mask, ocr_mask, dec_mask],
                dim=1
            )
            # offsets of each modality in the joint embedding space
            txt_max_num = txt_mask.size(-1)
            ocr_max_num = ocr_mask.size(-1)
            dec_max_num = dec_mask.size(-1)
            ocr_begin = txt_max_num
            ocr_end = ocr_begin + ocr_max_num
        elif self.drop_ocr:
            encoder_inputs = torch.cat(
                [txt_emb, obj_emb, dec_emb],
                dim=1
            )
            attention_mask = torch.cat(
                [txt_mask, obj_mask, dec_mask],
                dim=1
            )
            # offsets of each modality in the joint embedding space
            # obj_max_num = obj_mask.size(-1)
            dec_max_num = dec_mask.size(-1)
            # print('m4c.py text_end', txt_end)
            obj_begin = 0
            ocr_begin = -1
            ocr_end = -1
        else:
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
            # txt_begin = 0
            # txt_end = txt_begin + txt_max_num
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
        # print('pythia/models/m4c_control_captioner.py encoder_inputs:', encoder_inputs.size())
        # # Anwen Hu 2020/10/6 revise self.encoder input ,add bbox_coor containing obj_box, ocr_box and obj_begin, ocr_end
        if self.init_use_bbox_att or self.mid_use_bbox_att:
            assert not self.drop_ocr and not self.drop_obj
            bbox_coor['obj_begin'] = obj_begin
            bbox_coor['ocr_end'] = ocr_end
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
        # mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]
        if not self.drop_ocr:
            mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        else:
            mmt_ocr_output = ocr_emb

        results = {
            'mmt_seq_output': mmt_seq_output,
            # 'mmt_txt_output': mmt_txt_output,
            'mmt_ocr_output': mmt_ocr_output,
            'mmt_dec_output': mmt_dec_output,
        }
        return results




