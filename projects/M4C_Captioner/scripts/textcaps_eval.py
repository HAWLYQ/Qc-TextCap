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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--set', type=str, default='val')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--img_metrics_path', type=str, default='')
    args = parser.parse_args()

    if args.set not in ['train', 'val']:
        raise Exception(
            'this script only supports TextCaps train and val set. '
            'Please use the EvalAI server for test set evaluation'
        )

    with open(args.pred_file) as f:
        preds = json.load(f)
    if args.dataset == 'TextCaps':
        imdb_file = os.path.join(
            os.path.dirname(__file__),
            '../../../data/textcaps/imdb/m4c_textcaps/imdb_{}.npy'.format(args.set)
        )
    elif args.dataset == 'VizWiz':
        imdb_file = os.path.join(
            os.path.dirname(__file__),
            '../../../data/vizwiz/imdb/imdb_{}.npy'.format(args.set)
        )
    else:
        print('invalid dataset')
        exit(0)

    imdb = np.load(imdb_file, allow_pickle=True)
    imdb = imdb[1:]

    gts = [
        {'image_id': str(info['image_id']), 'caption': info['caption_str']}
        for info in imdb
    ]
    preds = [
        {'image_id': p['image_id'], 'caption': p['caption']}
        for p in preds
    ]
    # print(len(gts), len(preds))
    imgids = list(set(g['image_id'] for g in gts))
    # print(len(imgids))

    metrics, img_metrics = coco_caption_eval.calculate_metrics(
        imgids, {'annotations': gts}, {'annotations': preds}
    )

    # anwenhu 2020/9/16
    # write metrics of each image for visualization
    if args.img_metrics_path != '':
        print('write metrics of each image to', args.img_metrics_path)
        json.dump(img_metrics, open(args.img_metrics_path, 'w'))

    print_metrics(metrics)
