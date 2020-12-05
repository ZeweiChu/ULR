# RoBERTa dual encoder with L2 distance

```bash
python ./train_natcat.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name natcat \
    --do_train \
    --bert_representation avg \
    --seed 1 \
    --margin 0.5 \
    --do_lower_case \
    --data_dir ../../data \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --save_total_limit 2 \
    --output_dir saved_checkpoints/hinge_natcat_roberta_L2 \
    --warmup_steps 1313
```


After training, run eval script to produce roberta based embeddings
```bash
python eval_downstream_task.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name eval \
    --do_eval \
    --do_lower_case \
    --eval_data_dir ../../eval_data/agnews \
    --eval_data_file ../../eval_data/agnews/test.csv \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=64   \
    --output_dir saved_checkpoints/hinge_natcat_roberta_L2 \
    --bert_representation avg \
    --all_cats_file ../../eval_data/agnews/classes.txt.acl
```

To perform k-means based unsupervised label refinement
```bash
python compute_acc_kmeans.py  ../../eval_data/agnews/test.csv saved_checkpoints/hinge_natcat_roberta_L2/agnews.test.csv.text.txt saved_checkpoints/hinge_natcat_roberta_L2/agnews.test.csv.category.txt
```

```bash
python compute_acc_kmeans.py  ../../eval_data/agnews/test.csv model_predictions/agnews.text.txt model_predictions/agnews.category.txt
```
