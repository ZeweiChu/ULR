# Dual Encoder with Cosine Similarity


Finetune a RoBERTa dual encoder model with dot product and cross entropy loss
```bash
python ./train_natcat.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name natcat \
    --do_train \
    --bert_representation avg \
    --similarity_function dot \
    --seed 1 \
    --do_lower_case \
    --data_dir ../../data \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --save_total_limit 2 \
    --output_dir saved_checkpoints/bceloss_roberta_avg_dot \
    --warmup_steps 750
```


After training, run evaluation scripts to produce roberta based text representations
```bash
python ./eval_downstream_task.py \
    --model_type roberta \
    --model_name_or_path saved_checkpoints/bceloss_roberta_avg_dot \
    --task_name eval \
    --do_eval \
    --do_lower_case \
    --eval_data_dir ../../eval_data/agnews \
    --eval_data_file ../../eval_data/agnews/test.csv \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=64   \
    --output_dir saved_checkpoints/bceloss_roberta_avg_dot \
    --bert_representation avg \
    --similarity_function cosine \
    --all_cats_file ../../eval_data/agnews/classes.txt.acl

```

To perform k-means based unsupervised label refinement
```bash
python compute_acc_kmeans_cosine.py  ../../eval_data/agnews/test.csv saved_checkpoints/bceloss_roberta_avg_dot/agnews.test.csv.text.txt saved_checkpoints/bceloss_roberta_avg_dot/agnews.test.csv.category.txt
```

20NG
```bash
python compute_acc_kmeans_cosine.py  ../../eval_data/20newsgroups/test.csv model_predictions/20newsgroups.text.txt model_predictions/20newsgroups.category.txt
```
