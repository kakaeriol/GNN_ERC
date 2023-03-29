# python3 -u train_IEMOCAP.py \
#  	--base-model=GRU  --epochs=50 --dropout=0.45 \
#   	--batch-size=32 --lr=0.0003 > log_train_IEMOCAP

#python3 -u train_IEMOCAP.py \
#      --base-model=GRU > log_train_IEMOCAP

# python3 -u train_DailyDialog.py \
#  	--base-model=GRU  --epochs=50 --dropout=0.45 \
#   	--batch-size=32 --lr=0.0003 > log_train_DailyDialog

python3 -u train_MELD.py \
 	--labels=sentiment --base-model=LSTM  --epochs=80 --dropout=0.45 \
  	--batch-size=32 --lr=0.0003 > log_train_MELD