export CUDA_VISIBLE_DEVICES=0
python3 main.py \
--epoch 100 \
--learning_rate .0001 \
--beta 0.5 \
--batch_size 36 \
--input_height 128 \
--output_height 128 \
--dataset 'wikiart' \
--input_fname_pattern '*/*.jpg' \
--checkpoint_dir 'checkpoint' \
--data_dir '../dataset' \
--sample_dir 'samples' \
--G_optim_times 2 \
--y_dim 4 \
--_lambda 1.0 \
--crop \
--use_tiny_can \
--use_resize_conv \
--use_label_smoothing \
#--train
# boolean flags (unused == FALSE): train, crop, use_can, use_slim_can, use_tiny_can
# crop vs. resize (--crop)
# resizeconv vs. deconv2d (--use_resize_conv)
# label_smoothing vs. not (--use_label_smoothing)
