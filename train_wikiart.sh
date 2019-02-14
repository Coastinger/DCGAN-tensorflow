export CUDA_VISIBLE_DEVICES=0
python3 main.py \
--epoch 100 \
--learning_rate .00005 \
--beta 0.5 \
--batch_size 36 \
--input_height 128 \
--output_height 128 \
--dataset 'wikiart' \
--input_fname_pattern '*/*.jpg' \
--checkpoint_dir 'checkpoint' \
--data_dir '../dataset' \
--sample_dir 'samples' \
--y_dim 4 \
--_lambda 1.0 \
--use_slim_can \
--train
# boolean flags (unused == FALSE): train, crop, use_can, use_slim_can, use_tiny_can
