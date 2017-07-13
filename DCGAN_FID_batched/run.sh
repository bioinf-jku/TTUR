name=$1
lr_d=$2
lr_g=$3
script=$4
python3.5 $script \
--dataset celebA_cropped \
--input_height=64 \
--output_height=64 \
--is_crop False \
--is_train=True \
--batch_size=64 \
--checkpoint_dir="logs/checkpoints/checkpoint_${name}" \ #"checkpoints/DCGAN/celebA/checkpoint_${ts}_${num}" \
--log_dir="logs/tboard/${name}" \ #"tboard/celebA/${ts}_${num}" \
--sample_dir="logs/samples/samples_${name}" \
--fid_batch_size 100 \
--learning_rate_d $lr_d \
--learning_rate_g $lr_g \
--lr_decay_rate_d 1.0 \
--lr_decay_rate_g 1.0 \
--lr_decay_type 1 \
--lr_decay_step -1 \
--batch_size_m 1 \
--iter_m 1 \
--learning_rate_m 0.001 \
--epoch 5000 \
--load_checkpoint False \
--counter_start 0 \
--incept_path="inception-2015-12-05/classify_image_graph_def.pb" \
--data_path="/publicdata/image/celebA_cropped/" \
--stats_path="stats/stat_celeb_batched.pkl.gz" \
