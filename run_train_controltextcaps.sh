############## train GQAM with rand strategy #################
python -m torch.distributed.launch --nproc_per_node 1 --master_port 3140 tools/run.py --tasks captioning --datasets m4c_control_textcaps --model hie_control_captioner \
  --config configs/captioning/m4c_control_textcaps/GQAM_rand.yml \
  --save_dir save/GQAM_rand_controltextcaps \
  training_parameters.distributed True
