#!/bin/bash

dataset=iemocap
dataset_dir=data/${dataset}/dataset
ckpt_dir=data/${dataset}/ckpt
do_what=$1
d_rate=0.45
weight_decay=0.01
l_r=0.0001
bs=32
if [ "${do_what}" == "preprocess" ]; then
  python -u preprocess.py --data=${ckpt_dir}/data.pkl \
      --dataset=${dataset} > log/preprocess.${dataset}
elif [ "${do_what}" == "train" ]; then
  python -u train.py --data=${ckpt_dir}/data.pkl \
      --from_begin --device=cuda:1 --epochs=50 --drop_rate=$l_r \
      --weight_decay=$weight_decay --batch_size=$bs --learning_rate=$l_r > log/train.${dataset}.drop${d_rate}.wd${weight_decay}.lr${l_r}.bs${bs}
fi
