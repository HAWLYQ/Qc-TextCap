############## train GQAM w/o GE with rand strategy #################
python -m torch.distributed.launch --nproc_per_node 1 --master_port 2890 tools/run.py --tasks captioning --datasets m4c_control_textcaps --model hie_control_captioner \
  --config configs/captioning/m4c_control_vizwiz/GQAM_no_GE_rand.yml \
  --save_dir save/GQAM_no_GE_rand_controlvizwiz \
  training_parameters.distributed True
