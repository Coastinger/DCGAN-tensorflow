export CUDA_VISIBLE_DEVICES=0
python3 main.py \
--epoch 25 \
--learning_rate .0001 \
--beta 0.5 \
--batch_size 64 \
--input_height 64 \
--output_height 64 \
--dataset wikiart \
--data_dir ../dataset \
--input_fname_pattern */*.jpg \
--checkpoint_dir checkpoint \
--sample_dir samples \
--crop True \
--visualize False \
--use_can \
--y_dim 27 \
--generate_test_images 10 \
--train
