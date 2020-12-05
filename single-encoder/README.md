
```bash
python ./train_natcat.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name natcat \
    --do_train \
    --do_lower_case \
    --data_dir ../data \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --save_total_limit 3 \
    --output_dir saved_checkpoints/roberta_single \
    --warmup_steps 750
```

```bash
python ./eval_downstream_task.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name eval \
    --do_eval \
    --do_lower_case \
    --eval_data_dir ../eval_data/agnews \
    --eval_data_file ../eval_data/agnews/test.csv \
    --max_seq_length 128 \
    --class_file_name ../eval_data/agnews/classes.txt.acl \
    --per_gpu_eval_batch_size=32   \
    --output_dir saved_checkpoints/roberta_single
```


To perform k-means based unsupervised label refinement
```bash
python compute_acc_kmeans.py ../eval_data/agnews/test.csv saved_checkpoints/roberta_single/agnews.preds.txt
```

```bash
python compute_acc_kmeans.py ../eval_data/agnews/test.csv model_predictions/agnews.preds.txt
```
