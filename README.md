# Question-controlled Text-aware Image Captioning (ACM MM2021)

By Anwen Hu, Shizhe Chen, Qin Jin

Task preview:

![Qc-TextCap](https://github.com/HAWLYQ/Qc-TextCap/blob/master/task_preview.pdf)

 
 ## Requirements
 This project is revised based on [M4C-Captioner](https://github.com/facebookresearch/mmf/tree/project/m4c/projects/M4C_Captioner).
 You can create your own conda environment and then get started as follows:
 ```
conda create --prefix=$virtual_env_path python=3.6 
source activate $virtual_env_path
git clone https://github.com/HAWLYQ/Qc-TextCap.git
cd Qc-TextCap
python setup.py develop
```
 If there is a error `error: Setup script exited with error in demjson setup command: use_2to3 is invalid.`, try:
 ```
pip install --upgrade setuptools==57.5.0
```
When you get the message 'Finished processing dependencies for pythia==0.3', 
the environment is ready. 
Then install pycocoevalcap for caption evaluation as follows:
```
pip install git+https://github.com/ronghanghu/coco-caption.git@python23
```

### Download detectron weights
```
mkdir data
cd data
wget http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf detectron_weights.tar.gz
```

## Dataset
### Download Qc-TextCap Datasets
Download ControlTextCaps and ControlVizWiz from baidu disk (https://pan.baidu.com/s/1gw5l6eFFGO2OFfWt9ZtFAg, pwd：ykyi).
Put corresponding imdb directories and vocabulary files under `data/$dataset_name`. 
Raw images can be downloaded from official sites of [TextCaps](https://textvqa.org/textcaps/dataset/) and [VizWiz-Captions](https://vizwiz.org/tasks-and-datasets/image-captioning/).
 
 
### Dataset introduction
The text-aware captions, questions and basic information of images are all stored in .npy files. 
Each sample in .npy file represent a tuple `<image, text-aware caption, automatic initial caption, pseudo initial caption, questions>`.
The file can be read
with following codes:

```
import numpy as np
data_path = ...
data = np.load(data_path, allow_pickle=True)
for sample in data:
    print(sample)
 ```
 Each `sample` is a dictionary where `caption_str` refers to the `text-aware caption`,`simple_caption_str` refers to the `automatic initial caption`, 
 `model_simple_caption_str` refers to the `pseudo initial caption` generated by a in-domain AoANet, 
 `auto_question_str` refers to `questions` joined with the token \<ANS>. 
 
 ### Input Feature Preparation
 Each `sample` also stores the image height, width, object bounding boxes and ocr bounding boxes in `image_height`, `image_width`, `obj_normalized_boxes` and `ocr_normalized_boxes`.
 Extract object-level features of these bounding boxes by [bottom-up-attention](https://github.com/MILVLG/bottom-up-attention.pytorch) (for the config file, we choose `configs/bua-caffe/extract-bua-caffe-r152.yaml`) and put these features
 under  `data/$dataset_name/bua_feats/obj` and `data/$dataset_name/bua_feats/ocr`.
 
 For ControlTextCaps, the feature file in obj or ocr directory for each image should be named as `$image_id+ .npy`.
 For ControlVizWiz, the feature file in obj or ocr directory for each image should be named as `$image_name+ .npy`.
 These feature files should be read as follows:
 ```
import numpy as np
import torch
feat_path =  ...
bua_feat = torch.from_numpy(np.load(feat_path))
print(bua_feat)
```
The output should be a vector whose size is  N x 2048, where N is the number of bounding boxes. 
 
 
## Experiments

 ### train
 To train GQAM from scratch (e.g. GQAM with rand training strategy on ControlTextCaps), run the shell script as follows:
 ```
CUDA_VISIBLE_DEVICES=0 sh run_train_controltextcaps.sh
```
 
 ### test 
 To test a model (e.g. GQAM trained with rand strategy on ControlTextCaps), run the shell script to get captions as follows:
 
  ```
CUDA_VISIBLE_DEVICES=0 sh run_test_controltextcaps.sh
```
Copy the path of prediction file to `eval_QcTextCap.sh`, run this shell script to calculate captioning metrics.
Attention: During test, make sure `use_model_and_auto_simpel_cap` is set False and `use_model_simple_cap` is set True in the 
corresponding config file.


 ### trained checkpoints 
Our checkpoints (M4CC, GQAM w/o GE, GQAM) trained on ControlTextCaps and ControlVizWiz can be download from baidu disk (https://pan.baidu.com/s/1g8GzWAu0gVRlxGiphgDmsg, pwd:w4a6).
 
 
 
 ## Citation

If you find this code useful for your research, please consider citing:
```bibtex
@inproceedings{DBLP:conf/mm/HuCJ21,
  author    = {Anwen Hu and
               Shizhe Chen and
               Qin Jin},
  title     = {Question-controlled Text-aware Image Captioning},
  booktitle = {{ACM} Multimedia},
  pages     = {3097--3105},
  publisher = {{ACM}},
  year      = {2021}
}
```
 
 
 
 
 
