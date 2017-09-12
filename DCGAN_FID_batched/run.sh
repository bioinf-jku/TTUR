incept_path="inception-2015-12-05/classify_image_graph_def.pb"
#dataset="lsun"
dataset="celebA"
#data_path="data/lsun_cropped"
data_path="data/celebA_cropped"
stats_path="stats/fid_stats_celeba.npz"
lr_d=$1
lr_g=$2
counter_start=0
load_checkpoint=false
if ! $load_checkpoint; then
  dwt=`date "+%m%d_%H%M%S"`
  run_id=${dwt}_${lr_d}_${lr_g}
else
  run_id="MMdd_hhmmss_lrd_lrg"
fi
python3 main.py \
--dataset=$dataset \
--input_height=64 \
--output_height=64 \
--is_crop False \
--is_train=True \
--batch_size=64 \
--checkpoint_dir="logs/${run_id}/checkpoints" \
--log_dir="logs/${run_id}/logs" \
--sample_dir="logs/${run_id}/samples" \
--fid_n_samples 50000 \
--fid_sample_batchsize 1000 \
--fid_batch_size 100 \
--fid_eval_steps 5000 \
--learning_rate_d $lr_d \
--learning_rate_g $lr_g \
--beta1 0.5 \
--epoch 81 \
--load_checkpoint $load_checkpoint \
--counter_start $counter_start \
--incept_path $incept_path \
--data_path $data_path \
--stats_path $stats_path \
