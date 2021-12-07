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
    parser.add_argument('--metrics_path', type=str, default='')
    parser.add_argument('--eval_key', type=str, default='img')
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

    imdb = np.load(imdb_file, allow_pickle=True)
    # imdb = imdb[1:]

    # evaluate controllable captioning with a caption as an unit, thus use caption id as image id
    if args.eval_key == 'img':
        gts = [
            {'image_id': str(info['image_id']), 'caption': info['caption_str']}
            for info in imdb
        ]

        """simple_caps = [
            {'image_id': str(info['image_id']), 'caption': info['simple_caption_str']}
            for info in imdb
        ]"""
        preds_dict = {}
        for p in preds:
            # if there is multiple caption for one image, keep the last one
            preds_dict[str(p['image_id'])] = p['caption']
        preds = [{'image_id': img_id, 'caption': cap} for img_id, cap in preds_dict.items()]
        """preds = [
            {'image_id': str(p['image_id']), 'caption': p['caption']}
            for p in preds
        ]"""
    elif args.eval_key == 'cap':
        capid_to_imgid= {}
        for info in imdb:
            capid_to_imgid[str(info['caption_id'])] = str(info['image_id'])
        gts = [
            {'image_id': str(info['caption_id']), 'caption': info['caption_str']}
            for info in imdb
        ]
        preds_imgid_to_caption = {}
        for p in preds:
            preds_imgid_to_caption[str(p['image_id'])] = p['caption']
        preds = [
            {'image_id': capid, 'caption': preds_imgid_to_caption[capid_to_imgid[capid]]}
            for capid in capid_to_imgid.keys()
        ]
        """preds= [
            {'image_id': str(info['caption_id']), 'caption': info['simple_caption_str']}
            for info in imdb
        ]"""
    else:
        print('invalid evaluation key ', args.eval_key)
    preds_imgids = list(set(p['image_id'] for p in preds))
    print('preds %s num %d' % (args.eval_key, len(preds_imgids)))
    # print(len(gts), len(preds))
    imgids = list(set(g['image_id'] for g in gts))
    print('gts %s num %d' % (args.eval_key, len(imgids)))
    # print(len(imgids))
    if len(preds_imgids) != len(imgids):
        print('remove redundant img in preds or gts')
        common_imgids = set(preds_imgids)&set(imgids)
        print('commont img num ', len(list(common_imgids)))
        preds = [
            {'image_id': str(p['image_id']), 'caption': p['caption']}
            for p in preds if str(p['image_id']) in common_imgids
        ]
        gts = [
            {'image_id': str(g['image_id']), 'caption': g['caption']}
            for g in gts if str(g['image_id']) in common_imgids
        ]

    metrics, img_metrics = coco_caption_eval.calculate_metrics(
        imgids, {'annotations': gts}, {'annotations': preds}
    )

    # anwenhu 2020/9/16
    # write metrics of each image for visualization
    if args.metrics_path != '':
        print('write metrics of each %s to %s' % (args.eval_key, args.metrics_path))
        json.dump(img_metrics, open(args.metrics_path, 'w'))

    print_metrics(metrics)




