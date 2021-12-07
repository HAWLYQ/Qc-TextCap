# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.datasets.vqa.m4c_textvqa.dataset import M4CTextVQADataset
from pythia.utils.objects_to_byte_tensor import enc_obj2bytes
from pythia.common.sample import Sample
from pythia.utils.text_utils import word_tokenize
import numpy as np
import torch
import difflib
import random


class M4CControlTextCapsDataset(M4CTextVQADataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        print('pythia/datasets/captioning/m4c_control_textcaps/dataset.py M4CControlTextCapsDataset init')
        # in VQA2Dataset:
        super().__init__(
            dataset_type, imdb_file_index, config, *args, **kwargs
        )
        self._name = "m4c_control_textcaps"
        self.max_ocr_length = self.config.processors.answer_processor.params.max_length # ocr num
        self.max_question_num = self.config.processors.text_processor.params.max_question_num
        self.single_que_max_length = self.config.processors.text_processor.params.single_que_max_length
        self.max_ocrtoken_length = self.config.processors.text_processor.params.max_ocrtoken_length
        self.use_model_simple_cap = self.config.processors.text_processor.params.use_model_simple_cap
        self.use_model_and_auto_simpel_cap = self.config.processors.text_processor.params.use_model_and_auto_simpel_cap
        self.model_simple_cap_prob = self.config.processors.text_processor.params.model_simple_cap_prob
        self.simul_user_question = self.config.processors.text_processor.params.simul_user_question
        self.simul_cleanuser_question = self.config.processors.text_processor.params.simul_cleanuser_question
        self.simul_user_question_num = self.config.processors.text_processor.params.simul_user_question_num
        self.use_human_anno = self.config.processors.text_processor.params.use_human_anno
        self.obj_feat_length = self.config.obj_features_max_len
        self.ocr_feat_length = self.config.ocr_features_max_len

    def preprocess_sample_info(self, sample_info):
        # add dummy questions to train with M4C (for TextVQA)
        sample_info['question_str'] = ''  # empty question
        # anwenhu: question id will be used in sef.format_for_evalai, common/add_to_report
        sample_info['question_id'] = sample_info['caption_id']
        return sample_info

    """def postprocess_evalai_entry(self, entry):
        new_entry = {
            'caption_id': entry['question_id'],
            'image_id': entry['image_id'],
            'caption': entry['answer'],
            'pred_source': entry['pred_source'],
        }
        return new_entry"""

    def format_for_evalai(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        # pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        end_answers = report.scores.argmax(dim=-1).view(batch_size, -1)[:, -1].unsqueeze(1) # batch * 1
        pred_answers = report.prev_inds.view(batch_size, -1)[:, 1:] # 0 is start token
        pred_answers = torch.cat([pred_answers, end_answers], dim=-1)
        answer_space_size = answer_processor.get_true_vocab_size()

        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            context_tokens = report.context_tokens[idx]
            answer_words = []
            pred_source = []
            predicted_ocrs = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(
                        word_tokenize(context_tokens[answer_id])
                    )
                    pred_source.append('OCR')
                    # anwen hu 2021/4/13 for answer evaluation
                    predicted_ocrs.append(context_tokens[answer_id])
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
            # anwen hu 2021/4/13 for answer evaluation
            nopad_context_tokens = []
            for token in context_tokens:
                if token != '<pad>':
                    nopad_context_tokens.append(token)
            entry = {
                "caption_id": question_id.item(),
                "image_id": report.image_id[idx],
                "caption": pred_answer,
                "pred_source": pred_source,
                "pred_ocrs": predicted_ocrs,
                "processed_ocrs": nopad_context_tokens,
            }
            if self.simul_user_question or self.simul_cleanuser_question:
                entry['simul_question'] = report.simul_q_text_str[idx]
            # entry = self.postprocess_evalai_entry(entry)
            # print('entry(m4c_textvqa)', entry)

            predictions.append(entry)

        return predictions

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

    def get_ans_target(self, ocr_tokens, auto_answers):
        # auto_an_targets = []
        auto_an_targets = torch.zeros(self.max_question_num, self.max_ocr_length, dtype=torch.float)
        train_vqa_loss_mask = torch.zeros(self.max_question_num, dtype=torch.float)
        train_vqa_loss_mask[:len(auto_answers)] = 1.
        for a_id, auto_a in enumerate(auto_answers):
            ocr_sims = []
            for i, ocr_token in enumerate(ocr_tokens):
                sim = difflib.SequenceMatcher(None, auto_a.lower(), ocr_token.lower()).quick_ratio()
                ocr_sims.append((i, sim))
            sorted_sim = sorted(ocr_sims, key=lambda a: a[1], reverse=True)
            target = sorted_sim[0][0]
            auto_an_targets[a_id][target] = 1.0
            # auto_an_targets.append(target)
            # print('m4c_control_textcaps/dataset.py auto_a token: %s ; target %d ocr token: %s' % (auto_a, target, ocr_tokens[target]))
        # print('m4c_control_textcaps/dataset.py vqa target', auto_an_targets)
        # print('m4c_control_textcaps/dataset.py vqa mask', train_vqa_loss_mask)
        return auto_an_targets, train_vqa_loss_mask

    def simul_user_questions_by_objects(self, objects):
        category_objects = {'container': ['bottle', 'box', 'can', 'package', 'mug', 'pack'],
                         'cloth': ['jersey', 'uniform', 'shirt'],
                         'phone': ['phone', 'cellphone', 'computer', 'laptop'],
                         'screen': ['screen'],
                         'sign': ['sign', 'banner', 'poster', 'flyer', 'ad', 'advertisement', 'notice'],
                         'book': ['book', 'magazine'],
                         'player': ['player'],
                         'clock': ['clock', 'watch'],
                         'license': ['license', 'plate'],
                         }
        category_questions = {'container': [['what does the label on the <OBJ> say ', 'what word is on the <OBJ>'],
                                          'what is the brand on the <OBJ>',
                                          'what is in the <OBJ>'],
                            'cloth': ['what is the team name on the <OBJ>',
                                     ['what does the <OBJ> say on it', 'what is the word on the <OBJ>']],
                            'phone':['what is the date shown on the <OBJ>',
                                     'what app is installed on the <OBJ>', 'which website is displayed on the <OBJ>',
                                     'what is the operating system version of the <OBJ>',
                                     'what is the information shown on the <OBJ>', 'what is the brand of the <OBJ>',
                                     ['what does the <OBJ> say on the screen', 'what is the word on the <OBJ>']],
                            'screen': ['what is the operating system displayed on the <OBJ>',
                                       'what is the time displayed on the <OBJ>',
                                       'what is the type of <OBJ>',
                                       'what is the date displayed on the <OBJ>',
                                       'what does the <OBJ> display'],
                            'sign': ['what is the date on the <OBJ>',
                                     ['which company is the <OBJ> advertising for', 'what is the <OBJ> advertising for'],
                                     ['what does the <OBJ> state', 'what is the <OBJ> labeled',
                                      'what does the <OBJ> read', 'what words are on the <OBJ>',
                                      'what does the <OBJ> say on it'],
                                      'what does the <OBJ> show'],
                            'book': ['what is the title of the <OBJ>',
                                     ['who is the author of the <OBJ>', 'who wrote the <OBJ>'],
                                     ['what does the <OBJ> say on it', 'what words are on the <OBJ>'],
                                     'what is the <OBJ> about', 'what pages are the <OBJ> open to',
                                     ],
                            'player': ['what is the name of the <OBJ>', 'which team does the <OBJ> belong to',
                                       'who is the sponsor of the event', 'which state does the <OBJ> represent',
                                       'what is the number of the <OBJ>'],
                            'clock': ['what is the brand of the <OBJ>', 'what is the type of the <OBJ>',
                                      'what does the <OBJ> say on it', 'what is the date shown on the <OBJ>'],
                            'license': ['which state does the <OBJ> belong to',
                                        ['which does the <OBJ> read', 'which does the <OBJ> say on it'],
                                        'which is the name of the <OBJ>']}

        obj2cat = {}
        for cat, objs in category_objects.items():
            for obj in objs:
                obj2cat[obj] = cat
        simul_questions = []
        for i, object in enumerate(objects):
            # for each object, assign a question number
            if i == len(objects)-1:
                que_num = self.simul_user_question_num - (len(objects)-1)*int(round(self.simul_user_question_num/len(objects)))
            else:
                # 1.5 > 2
                que_num = int(round(self.simul_user_question_num/len(objects)))
            question_candidates = category_questions[obj2cat[object]]
            que_num = min(que_num, len(question_candidates))
            rand_indexes = random.sample(range(len(question_candidates)), que_num)
            for index in rand_indexes:
                if isinstance(question_candidates[index], str):
                    question_str = '<ANS> ' + question_candidates[index].replace('<OBJ>', object)
                    simul_questions.append(question_str)
                # if find a set of similar questions, randomly choose one
                elif isinstance(question_candidates[index], list):
                    question_str = '<ANS> ' + random.sample(question_candidates[index], 1)[0].replace('<OBJ>', object)
                    simul_questions.append(question_str)
            if len(simul_questions) >= self.simul_user_question_num:
                break
        simul_questions_str = ' '.join(simul_questions)
        # print('m4c_control_textcaps/dataset.py objects ', objects)
        # print('m4c_control_textcaps/dataset.py simu_questions ', simul_questions_str)
        return simul_questions_str

    def simul_user_questions_from_candidates(self, question_candidates):
        simul_questions = []
        que_num = min(self.simul_user_question_num, len(question_candidates))
        rand_indexes = random.sample(range(len(question_candidates)), que_num)
        for index in rand_indexes:
            if isinstance(question_candidates[index], str):
                question_str = '<ANS> ' + question_candidates[index]
                simul_questions.append(question_str)
            # if find a set of similar questions, randomly choose one
            elif isinstance(question_candidates[index], list):
                question_str = '<ANS> ' + random.sample(question_candidates[index], 1)[0]
                simul_questions.append(question_str)
        simul_questions_str = ' '.join(simul_questions)
        # print('m4c_control_textcaps/dataset.py objects ', objects)
        # print('m4c_control_textcaps/dataset.py simu_questions ', simul_questions_str)
        return simul_questions_str


    def add_sample_details(self, sample_info, sample):
        # 1. Load simple caption
        if self.use_model_simple_cap:
            simple_caption_str = sample_info['model_simple_caption_str']
            # print('pythia/datasets/m4c_control_textcaps/dataset.py use simple caption str: ', simple_caption_str)
            # print('pythia/datasets/m4c_control_textcaps/dataset.py raw simple caption str: ', sample_info['simple_caption_str'])
        elif self.use_model_and_auto_simpel_cap:
            pool = ['model']*int(self.model_simple_cap_prob*10)
            pool += ['auto']*int(10-self.model_simple_cap_prob*10)
            simple_cap_type = random.choice(pool)
            # print('pythia/datasets/captioning/m4c_control_textcaps.py use %s simple cap' % simple_cap_type)
            if simple_cap_type == 'model':
                simple_caption_str = sample_info['model_simple_caption_str']
            else:
                simple_caption_str = sample_info['simple_caption_str']
        elif self.use_human_anno:
            simple_caption_str = sample_info['human_simple_caption_str']
        else:
            simple_caption_str = sample_info['simple_caption_str']
        # text_processor is the BertTokenizerProcessor in pythia/datasets/processor.py,
        # originally designed for question in VQA, so key is 'question'
        processed_simple_cap = self.text_processor({"question": simple_caption_str})
        sample.simple_cap_text = processed_simple_cap['token_inds']
        sample.simple_cap_text_len = processed_simple_cap['token_num']  #

        # 2. Load  auto question
        if self.use_human_anno:
            auto_question_str = sample_info['human_question_str']
        elif self.simul_user_question:
            assert self.use_model_simple_cap
            auto_question_str = self.simul_user_questions_by_objects(sample_info['model_simple_caption_objects'])
            # used only for save
            sample.simul_q_text_str = auto_question_str
        elif self.simul_cleanuser_question:
            assert self.use_model_simple_cap
            auto_question_str = self.simul_user_questions_from_candidates(sample_info['clean_questions_candidates'])
            # used only for save
            sample.simul_q_text_str = auto_question_str
        else:
            auto_question_str = sample_info['auto_question_str']
        # concatenated questions
        # text_processor is the bert tokenizer, so use [SEP] as <ANS>
        processed_auto_question = self.text_processor({"question": auto_question_str.replace('<ANS>', '[SEP]')},
                                                      fetch_ans_index=True)
        sample.auto_q_text = processed_auto_question['token_inds']
        sample.auto_q_text_len = processed_auto_question['token_num']
        # 2021/2/24: fetch answer token index
        sample.ans_index = processed_auto_question['ans_index']
        # separated questions
        # print('m4c_control_textcaps/datasets.py auto_question_str ', auto_question_str)
        auto_question_strs = auto_question_str.split('<ANS>')[1:] # '0' is null str
        # print('m4c_control_textcaps/datasets.py auto_question_strs ', auto_question_strs)
        auto_sep_q_text = torch.zeros(self.single_que_max_length * self.max_question_num, dtype=torch.long)  # 50
        auto_sep_q_text_mask = torch.zeros(self.single_que_max_length * self.max_question_num, dtype=torch.long)  # 50
        for i in range(len(auto_question_strs)):
            processed_single_auto_question = self.text_processor({"question": auto_question_strs[i].strip()})
            auto_sep_q_text[i*self.single_que_max_length:(i+1)*self.single_que_max_length] = processed_single_auto_question['token_inds'][:self.single_que_max_length]
            seq_token_num = min(processed_single_auto_question['token_num'], self.single_que_max_length)
            auto_sep_q_text_mask[i*self.single_que_max_length: i*self.single_que_max_length+seq_token_num] = torch.ones(seq_token_num)
        sample.auto_sep_q_text = auto_sep_q_text  # 10*5=50
        sample.auto_sep_q_text_mask = auto_sep_q_text_mask.type(torch.float32) # 10*5=50
        # print('m4c_control_textcaps/datasets.py auto_sep_q_text ', auto_sep_q_text)
        # print('m4c_control_textcaps/datasets.py auto_sep_q_text_mask ', auto_sep_q_text_mask)
        sample.auto_sep_q_num = torch.tensor(len(auto_question_strs), dtype=torch.long)

        # print('m4c_control_textcaps\dataset.py simple caption str:', simple_caption_str)
        # print('m4c_control_textcaps\dataset.py simple caption:', sample.simple_cap_text, sample.simple_cap_text_len.item())
        # print('m4c_control_textcaps\dataset.py auto question str:', auto_question_str)
        # print('m4c_control_textcaps\dataset.py auto question:', sample.auto_q_text, sample.auto_q_text_len.item())
        # exit(0)

        # 3. Load object
        # object bounding box information
        sample.obj_bbox_coordinates = self.copy_processor(
            {"blob": sample_info["obj_normalized_boxes"]}
        )["blob"]
        # TODO: get_ans_target()
        # 4. Load OCR
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info['ocr_tokens'] = []
            sample_info['ocr_info'] = []
            if 'ocr_normalized_boxes' in sample_info:
                sample_info['ocr_normalized_boxes'] = np.zeros(
                    (0, 4), np.float32
                )
            # clear OCR visual features
            sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)

        # Preprocess OCR tokens
        ocr_tokens = []
        for token in sample_info["ocr_tokens"][:self.max_ocr_length]:
            sub_tokens = token.split(' ')[:self.max_ocrtoken_length]
            ocr_tokens.append(self.ocr_token_processor({"text": ' '.join(sub_tokens)})["text"])  # remove , ?
        # 2031/2/24 generate ans_target
        # tensor max_question_num * max_ocr_length ( 5 * 50); mask: max_question_num: 5
        if self.use_human_anno:
            sample.ans_target, sample.train_vqa_loss_mask = self.get_ans_target(ocr_tokens, sample_info["human_answers"])
        else:
            sample.ans_target, sample.train_vqa_loss_mask = self.get_ans_target(ocr_tokens, sample_info["auto_answers"])
        # Get FastText embeddings for OCR tokens
        # print('pythia/datasets/captioning/m4c_control_textcaps.py ocr tokens: ', len(ocr_tokens))
        context = self.context_processor({"tokens": ocr_tokens})  # fasttext processor
        sample.context = context["text"]
        sample.context_tokens = context["tokens"]  # add <pad> for ocr_tokens
        sample.context_tokens_enc = enc_obj2bytes(context["tokens"])
        sample.context_feature_0 = context["text"] # model ocr input
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = context["length"]
        # Get PHOC embeddings for OCR tokens
        context_phoc = self.phoc_processor({"tokens": ocr_tokens})
        sample.context_feature_1 = context_phoc["text"] # model ocr input
        sample.context_info_1 = Sample()
        sample.context_info_1.max_features = context_phoc["length"]
        # OCR order vectors
        # TODO remove order_vectors -- it is no longer needed in M4C
        order_vectors = np.eye(len(sample.context_tokens), dtype=np.float32)
        order_vectors = torch.from_numpy(order_vectors)
        order_vectors[context["length"]:] = 0
        sample.order_vectors = order_vectors
        # OCR bounding box information
        assert 'ocr_normalized_boxes' in sample_info
        # New imdb format: OCR bounding boxes are already pre-computed
        max_len = self.config.processors.answer_processor.params.max_length
        sample.ocr_bbox_coordinates = self.copy_processor(
            {"blob": sample_info['ocr_normalized_boxes']}
        )["blob"][:max_len]
        return sample

    def add_answer_info(self, sample_info, sample):
        sample_has_caption = ('caption_str' in sample_info)
        # anwen hu 2020/11/18
        if sample_has_caption:
            sample_info['answers'] = [sample_info['caption_str']]

        sample = super().add_answer_info(sample_info, sample)

        if sample_has_caption:
            sample.caption_str = enc_obj2bytes(sample_info['caption_str'])
            sample.ref_strs = enc_obj2bytes(sample_info['reference_strs'])
            sample.pop('gt_answers_enc')

        return sample

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        # breaking change from VQA2Dataset: load question_id
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )  # anwenhu: question_id is caption_id, only used for inference interface(inherited from VQA)

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = str(sample_info["image_id"])
        else:
            current_sample.image_id = sample_info["image_id"]

        if self._use_features is True:
            # contain obj features(image_feature_0 100*2048) and info, ocr features(image_feature_1 100*2048) and info
            features = self.features_db[idx]
            # anwen hu revise obj or ocr vision feat length
            features['image_feature_0'] = features['image_feature_0'][:self.obj_feat_length]
            features['image_feature_1'] = features['image_feature_1'][:self.ocr_feat_length]
            # print('/pythia/datasets/vqa/m4c_textvqa/dataset.py features.keys', features.keys())
            # print('/pythia/datasets/vqa/m4c_textvqa/dataset.py image_info_0', features['image_info_0'])
            # print('/pythia/datasets/vqa/m4c_textvqa/dataset.py image_info_1', features['image_info_1'])
            # print('/pythia/datasets/vqa/m4c_textvqa/dataset.py image_feature_1.shape', features['image_feature_1'].shape)
            current_sample.update(features)

        # add simple caption, auto question, obj bbox, ocr bbox
        current_sample = self.add_sample_details(sample_info, current_sample)
        # add caption
        current_sample = self.add_answer_info(sample_info, current_sample)

        # only the 'max_features' key is needed
        # (anwenhu: max_features stores the obj/ocr num before padding features to 100)
        # pop other keys to minimize data loading overhead
        for k in list(current_sample.image_info_0):
            if k != 'max_features':
                current_sample.image_info_0.pop(k)
        for k in list(current_sample.image_info_1):
            if k != 'max_features':
                current_sample.image_info_1.pop(k)

        return current_sample


