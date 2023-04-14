#!/bin/bash

dataset=IEMOCAP
data=/home/n/nguyenpk/CS6208/GNN_ERC/NP_preprocessing/data/${dataset}_token_fts.pkl
embed=/home/n/nguyenpk/CS6208/GNN_ERC/NP_preprocessing/data/${dataset}_embedd.pkl

python -u train.py --data=${data} --pretrained_word_vectors=${embed}\
      --from_begin --device='cuda:0' --epochs=50 --drop_rate=0.45 \
      --weight_decay=0.0 --batch_size=32 --learning_rate=0.0003 --rnn=gru > log.train.${dataset}