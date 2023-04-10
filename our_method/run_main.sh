#!/bin/bash

dataset_path=/home/n/nguyenpk/CS6208/GNN_ERC/baseline/DialogueGCN-mianzhang/data/iemocap/ckpt/data.pkl

python -u  train.py --data=${dataset_path} --Rtype=MHA \
       --from_begin --device=cuda:0 --epochs=100 --drop_rate=0.45 \
       --weight_decay=0.0 --batch_size=32 --learning_rate=0.0003 --rnn=transformer --class_weight --lossfunc=entropy> log_train
       
# python -u train.py --data=${dataset_path} \
#        --from_begin --device=cuda:0 --epochs=100 --drop_rate=0.45 \
#        --weight_decay=0.0 --batch_size=32 --learning_rate=0.0003 --rnn=transformer --class_weight --lossfunc=entropy > $output_log/train_transfomer_RGTv01
       
# python -u train.py --data=${dataset_path} \
#       --from_begin --device=cuda:0 --epochs=50 --drop_rate=0.45 \
#       --weight_decay=0.0 --batch_size=32 --learning_rate=0.0003 --rnn=gru --class_weight --lossfunc=entropy > log/train_gru_RGTv01

# python -u train.py --data=${dataset_path} \
#       --from_begin --device=cuda:0 --epochs=50 --drop_rate=0.45 \
#       --weight_decay=0.0 --batch_size=32 --learning_rate=0.0003 --rnn=gru --class_weight --lossfunc=entropy > log/train_lstm_RGTv01
