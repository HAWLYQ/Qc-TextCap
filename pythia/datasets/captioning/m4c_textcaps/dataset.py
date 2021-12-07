# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.datasets.vqa.m4c_textvqa.dataset import M4CTextVQADataset
from pythia.utils.objects_to_byte_tensor import enc_obj2bytes
from pythia.utils.text_utils import word_tokenize


class M4CTextCapsDataset(M4CTextVQADataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        print('pythia/datasets/captioning/m4c_textcaps/dataset.py  M4CTextCapsDdataset init')
        super().__init__(
            dataset_type, imdb_file_index, config, *args, **kwargs
        )
        self._name = "m4c_textcaps"

    def preprocess_sample_info(self, sample_info):
        # add dummy questions to train with M4C (for TextVQA)
        sample_info['question_str'] = ''  # empty question
        sample_info['question_id'] = sample_info['caption_id']
        return sample_info

    def postprocess_evalai_entry(self, entry):
        new_entry = {
            'caption_id': entry['question_id'],
            'image_id': entry['image_id'],
            'caption': entry['answer'],
            'pred_source': entry['pred_source'],
        }
        return new_entry

    def format_for_evalai_beam(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        # pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        beam_answers = report.final_beams
        answer_space_size = answer_processor.get_true_vocab_size()
        # print('answer_space_size', answer_space_size)

        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            context_tokens = report.context_tokens[idx]
            pred_answers = []
            scores = []
            preds_source = []
            for beam in beam_answers:
                answer_words = []
                pred_source = []
                # print('beam',beam)
                assert len(beam['seq']) <= 30
                for answer_id in beam['seq'].tolist():
                    if answer_id >= answer_space_size:
                        answer_id -= answer_space_size
                        answer_words.append(
                            word_tokenize(context_tokens[answer_id])
                        )
                        pred_source.append('OCR')
                    else:
                        if answer_id == answer_processor.EOS_IDX:
                            break
                        answer_words.append(
                            answer_processor.answer_vocab.idx2word(answer_id)
                        )
                        pred_source.append('VOCAB')
                # join all the answer tokens with space
                # (this should be correct for almost all cases)
                pred_answer = ' '.join(answer_words).replace(" 's", "'s")
                pred_answers.append(pred_answer)
                scores.append(beam['score'].numpy().tolist())
                preds_source.append(pred_source)

            entry = {
                "caption_id": question_id.item(),
                "image_id": report.image_id[idx],
                "captions": pred_answers,
                "preds_source": preds_source,
                "preds_score": scores,
            }
            # entry = self.postprocess_evalai_entry_(entry)
            # print('entry:', entry)
            predictions.append(entry)

        return predictions

    def add_answer_info(self, sample_info, sample):
        sample_has_caption = ('caption_str' in sample_info)
        # anwen hu 2020/11/18
        if sample_has_caption:
            sample_info['answers'] = [sample_info['caption_str']]
        """if sample_has_caption:
            print('pythia/datasets/captioning/m4c_textcaps/dataset.py caption_str', sample_info['caption_str'])
            # print('pythia/datasets/captioning/m4c_textcaps/dataset.py use spacy token list')
            spacy_token_caption_str = ' '.join(sample_info['spacy_token_list'][1:-1])
            print('pythia/datasets/captioning/m4c_textcaps/dataset.py spacy_token_caption_str', sample_info['caption_str'])
            sample_info['answers'] = [spacy_token_caption_str]  # [1:-1] means to drop the '<s>' and '</s>' token
            sample_info['answers_raw'] = [sample_info['caption_str']]"""

        sample = super().add_answer_info(sample_info, sample)

        if sample_has_caption:
            sample.caption_str = enc_obj2bytes(sample_info['caption_str'])
            sample.ref_strs = enc_obj2bytes(sample_info['reference_strs'])
            sample.pop('gt_answers_enc')

        return sample


