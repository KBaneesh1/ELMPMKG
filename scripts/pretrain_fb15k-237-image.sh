python main.py --gpus "0," --max_epochs=15  --num_workers=4 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --model_class UnimoKGC \
   --batch_size 64 \
   --pretrain 1 \
   --bce 0 \
   --check_val_every_n_epoch 1 \
   --overwrite_cache \
   --data_dir dataset/FB15k-237 \
   --task_name fb15k-237 \
   --eval_batch_size 64 \
   --max_seq_length 64 \
   --lr 5e-4    