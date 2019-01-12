export CUDA_VISIBLE_DEVICES=0
python3 main.py \
--epoch 25 \
--learning_rate .0001 \
--beta 0.5 \
--batch_size 64 \
--input_height 28 \
--output_height 28 \
--dataset mnist \
--input_fname_pattern */*.jpg \
--checkpoint_dir checkpoint \
--sample_dir samples \
--y_dim 10 \
--crop False \
--visualize False \
--generate_test_images 10 \
--use_can True \
--train
