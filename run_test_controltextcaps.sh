############## test M4CC trained by auto strategy #################
#python tools/run.py --tasks captioning --datasets m4c_control_textcaps --model m4c_control_captioner \
#  --config configs/captioning/m4c_control_textcaps/M4CC_auto.yml \
#  --save_dir checkpoints/M4CC_auto_controltextcaps \
#  --run_type inference --evalai_inference 1\
#  --resume_file checkpoints/M4CC_auto_controltextcaps/m4c_control_textcaps_m4c_control_captioner/best.ckpt \
#  --beam_size 0

############## test GQAM w/o GE trained by auto strategy #################
#python tools/run.py --tasks captioning --datasets m4c_control_textcaps --model hie_control_captioner \
#  --config configs/captioning/m4c_control_textcaps/GQAM_no_GE_auto.yml \
#  --save_dir checkpoints/GQAM_no_GE_auto_controltextcaps \
#  --run_type inference --evalai_inference 1 \
#  --resume_file checkpoints/GQAM_no_GE_auto_controltextcaps/m4c_control_textcaps_hie_control_captioner/best.ckpt \
#  --beam_size 0

############## test GQAM trained by auto strategy #################
#python tools/run.py --tasks captioning --datasets m4c_control_textcaps --model hie_control_captioner \
#  --config configs/captioning/m4c_control_textcaps/GQAM_auto.yml \
#  --save_dir checkpoints/GQAM_auto_controltextcaps \
#  --run_type inference --evalai_inference 1 \
#  --resume_file checkpoints/GQAM_auto_controltextcaps/m4c_control_textcaps_hie_control_captioner/best.ckpt \
#  --beam_size 0

############## test GQAM trained by rand(auto, pseudo) strategy #################
python tools/run.py --tasks captioning --datasets m4c_control_textcaps --model hie_control_captioner \
  --config configs/captioning/m4c_control_textcaps/GQAM_auto.yml \
  --save_dir checkpoints/GQAM_rand_controltextcaps \
  --run_type inference --evalai_inference 1 \
  --resume_file checkpoints/GQAM_rand_controltextcaps/m4c_control_textcaps_hie_control_captioner/best.ckpt \
  --beam_size 0


