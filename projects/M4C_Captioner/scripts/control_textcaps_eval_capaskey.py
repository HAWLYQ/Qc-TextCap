import sys
import json
import numpy as np
import os

sys.path.append(
    os.path.join(os.path.dirname(__file__), '../../../pythia/scripts/coco/')
)
import coco_caption_eval  # NoQA


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
    parser.add_argument('--reports_dir', type=str, default='')
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--cap_metrics_file', type=str, default='')
    parser.add_argument('--simple_cap_eval', type=int, default=0)
    parser.add_argument('--simple_cap_type', type=str, default='')
    args = parser.parse_args()

    assert args.set in ['train', 'val', 'test', 'annotest', 'usertest', 'cleantest', 'mqusertest', 'cleanusertest']

    with open(args.reports_dir+args.pred_file) as f:
        preds = json.load(f)
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
    gts = [
        {'image_id': str(info['caption_id']), 'caption': info['caption_str']}
        for info in imdb
    ]

    if args.simple_cap_type == 'auto':
        simple_caps = [
            {'image_id': str(info['caption_id']), 'caption': info['simple_caption_str']}
            for info in imdb
        ]
    elif args.simple_cap_type == 'model':
        simple_caps = [
            {'image_id': str(info['caption_id']), 'caption': info['model_simple_caption_str']}
            for info in imdb
        ]
    elif args.simple_cap_type == 'human':
        simple_caps = [
            {'image_id': str(info['caption_id']), 'caption': info['human_simple_caption_str']}
            for info in imdb
        ]
    else:
        print('invalid simple cap type ', args.simple_cap_type)
        exit(0)


    preds = [
        {'image_id': str(p['caption_id']), 'caption': p['caption']}
        for p in preds
    ]
    preds_metrics, preds_cap_metrics = evaluate(gts, preds)
    print('======preds=======')
    print_metrics(preds_metrics)

    # anwenhu 2021/1/20
    # write metrics of each caption for visualization
    if args.cap_metrics_file != '':
        cap_metrics_path = args.reports_dir+ 'preds_'+ args.cap_metrics_file
        print('write metrics of each predicted caption to', cap_metrics_path)
        json.dump(preds_cap_metrics, open(cap_metrics_path, 'w'))
    if args.simple_cap_eval == 1:
        simplecaps_metrics, simplecaps_cap_metrics = evaluate(gts, simple_caps)
        print('======simple caps=======')
        print_metrics(simplecaps_metrics)
        if args.cap_metrics_file != '':
            cap_metrics_path = args.reports_dir + args.simple_cap_type + '_simplecaps_' + args.cap_metrics_file
            print('write metrics of each simple caption to', cap_metrics_path)
            json.dump(simplecaps_cap_metrics, open(cap_metrics_path, 'w'))





