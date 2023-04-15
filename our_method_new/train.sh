#!/bin/bash

dataset=IEMOCAP
# data=/home/n/nguyenpk/CS6208/GNN_ERC/NP_preprocessing/data/${dataset}_token_fts.pkl
# embed=/home/n/nguyenpk/CS6208/GNN_ERC/NP_preprocessing/data/${dataset}_embedd.pkl

data=/home/h/huongntm/repo/GNN_data/${dataset}_token_fts.pkl
embed=/home/h/huongntm/repo/GNN_data/${dataset}_embedd.pkl

python -u train.py --data=${data} --pretrained_word_vectors=${embed}\
      --from_begin --device='cuda:0' --epochs=300 --drop_rate=0.4 \
      --weight_decay=0.001 --batch_size=32 --learning_rate=0.0003 --rnn=transformer > log.train.${dataset}_TM_nll_mask_weighted_rl_dr0.4
