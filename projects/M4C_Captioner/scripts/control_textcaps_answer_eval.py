import sys
import json
import numpy as np
import os

sys.path.append(
    os.path.join(os.path.dirname(__file__), '../../../pythia/scripts/coco/')
)
import coco_caption_eval  # NoQA
import difflib


def print_metrics(res_metrics):
    print(res_metrics)
    keys = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'SPICE', 'CIDEr']
    print('\n\n**********\nFinal model performance:\n**********')
    for k in keys:
        print(k, ': %.1f' % (res_metrics[k] * 100))


def evaluate(gts, preds):
    print('gts length: %d, preds length:%d' % (len(gts), len(preds)))
    imgids = list(set(g['image_id'] for g in gts))
    print('imgids(captionids) length: ', len(imgids))

    metrics, img_metrics = coco_caption_eval.calculate_metrics(
        imgids, {'annotations': gts}, {'annotations': preds}
    )
    return metrics, img_metrics



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--set', type=str, default='val')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--reports_dir1', type=str, default='')
    parser.add_argument('--pred_file1', type=str, required=True)
    parser.add_argument('--reports_dir2', type=str, default='')
    parser.add_argument('--pred_file2', type=str, required=True)
    parser.add_argument('--align_answer2ocr', type=int, default=0)
    args = parser.parse_args()

    assert args.set in ['train', 'val', 'test', 'annotest', 'usertest', 'cleantest', 'mqusertest', 'cleanusertest']

    with open(args.reports_dir1+args.pred_file1) as f:
        preds1 = json.load(f)

    with open(args.reports_dir2+args.pred_file2) as f:
        preds2 = json.load(f)


    if args.dataset == 'ControlTextCaps':
        imdb_file = os.path.join(
            os.path.dirname(__file__),
            '../../../data/textcaps/imdb/m4c_control_textcaps/imdb_azure_{}_v2a.npy'.format(args.set)
        )
    elif args.dataset == 'ControlVizWiz':
        imdb_file = os.path.join(
            os.path.dirname(__file__),
            '../../../data/vizwiz/imdb/m4c_control_vizwiz/imdb_azure_{}_v2a.npy'.format(args.set)
        )
    else:
        print('invalid dataset')
        exit(0)
    print('load ', imdb_file)
    imdb = np.load(imdb_file, allow_pickle=True)
    # imdb = imdb[1:]

    # evaluate controllable captioning with a caption as an unit, thus use caption id as image id

    gt_capid_answers = {}
    for info in imdb:
        # print('gt caption: ', info['caption_str'])
        # print('auto questions: ', info['auto_question_str'])
        # print('auto answers: ', info['auto_answers'])
        # print('ocr tokens: ', info['ocr_tokens'])
        gt_answer_tokens = []
        for answer in info['auto_answers']:
            gt_answer_tokens += answer.split(' ')
        gt_capid_answers[info['caption_id']] = {'answers': info['auto_answers'], 'answer_tokens': gt_answer_tokens}

    capid_ocr_tokens = {}
    preds1_capid_answers = {}
    for pred in preds1:
        pred_answers = pred['pred_ocrs']
        pred_answer_tokens = []
        for answer in pred_answers:
            pred_answer_tokens += answer.split(' ')
        preds1_capid_answers[pred['caption_id']] = {'answers': pred_answers, 'answer_tokens': pred_answer_tokens}
        ocrs = []
        for ocr_phrase in pred['processed_ocrs']:
            ocrs += ocr_phrase.split(' ')
        capid_ocr_tokens[pred['caption_id']] = ocrs

    preds2_capid_answer = {}
    for pred in preds2:
        pred_answers = pred['pred_ocrs']
        pred_answer_tokens = []
        for answer in pred_answers:
            pred_answer_tokens += answer.split(' ')
        preds2_capid_answer[pred['caption_id']] = {'answers': pred_answers, 'answer_tokens': pred_answer_tokens}

    assert len(preds1_capid_answers.keys()) == len(gt_capid_answers.keys())
    assert len(preds2_capid_answer.keys()) == len(gt_capid_answers.keys())

    all_true_num1 = 0
    all_pred_num1 = 0
    all_true_num2 = 0
    all_pred_num2 = 0
    all_gt_num = 0
    for cap_id in gt_capid_answers.keys():
        # token level
        pred1 = preds1_capid_answers[cap_id]['answer_tokens']
        pred2 = preds2_capid_answer[cap_id]['answer_tokens']
        gt = gt_capid_answers[cap_id]['answer_tokens']
        if args.align_answer2ocr == 1:
            align_gt = []
            for token in gt:
                ocr_sims = []
                for ocr_token in capid_ocr_tokens[cap_id]:
                    sim = difflib.SequenceMatcher(None, token, ocr_token).quick_ratio()
                    ocr_sims.append((ocr_token, sim))
                sorted_sim = sorted(ocr_sims, key=lambda a: a[1], reverse=True)
                align_token = sorted_sim[0][0]
                align_gt.append(align_token)
            gt = align_gt
        true_tokens1 = []
        true_tokens2 = []
        """for token in pred1:
            if token in gt:
                true_tokens1.append(token)
        for token in pred2:
            if token in gt:
                true_tokens2.append(token)"""
        for token in gt:
            if token in pred1:
                true_tokens1.append(token)
            if token in pred2:
                true_tokens2.append(token)
        """print('caption: ', cap_id)
        print('gt: ', gt)
        print('pred1: ', pred1)
        print('pred2: ', pred2)
        print('true1: ', true_tokens1)
        print('true2: ', true_tokens2)
        print('\n')"""
        all_true_num1 += len(true_tokens1)
        all_pred_num1 += len(pred1)
        all_true_num2 += len(true_tokens2)
        all_pred_num2 += len(pred2)
        all_gt_num += len(gt)

    precision_1 = all_true_num1/all_pred_num1
    recall_1 = all_true_num1/all_gt_num
    if precision_1 + recall_1 > 0:
        f1_1 = 2 * (precision_1* recall_1) / (precision_1 + recall_1)
    else:
        f1_1 = 0
    print('1 precision: %.4f, recall: %.4f, f1:%.4f' % (precision_1, recall_1, f1_1))

    precision_2 = all_true_num2 / all_pred_num2
    recall_2 = all_true_num2 / all_gt_num
    if precision_2 + recall_2 > 0:
        f1_2 = 2 * (precision_2 * recall_2) / (precision_2 + recall_2)
    else:
        f1_2 = 0
    print('2 precision: %.4f, recall: %.4f, f1:%.4f' % (precision_2, recall_2, f1_2))











