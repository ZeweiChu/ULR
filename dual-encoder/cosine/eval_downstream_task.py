# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import re
import random
import shutil
import pickle
from typing import Dict, List, Tuple
import code

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from utils import dual_encoder_convert_examples_to_features as convert_examples_to_features
from utils import compute_metrics
from utils import output_modes
from utils import processors 
from utils import DataFiles 


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def evaluate(args, model, tokenizer, file_path, cat_file_path, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    cats = []
    with open(cat_file_path) as fin:
        for line in fin:
            cats.append(line.strip())

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, file_path, cat_file_path)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = []
        text_embeddings = []
        category_embeddings = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs_a = {"input_ids": batch[0], "attention_mask": batch[1]}
                inputs_b = {"input_ids": batch[3], "attention_mask": batch[4]}
                labels = batch[6]
                if args.model_type != "distilbert":
                    inputs_a["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    inputs_b["token_type_ids"] = (
                        batch[5] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs_a = model(**inputs_a)
                outputs_b = model(**inputs_b)

                # code.interact(local=locals())

                if args.bert_representation == "pool":
                    rep1 = outputs_a[1]
                    rep2 = outputs_b[1]
                elif args.bert_representation == "avg":
                    rep1 = torch.sum(outputs_a[0] * batch[1].unsqueeze(-1), 1) / (torch.sum(batch[1], 1, keepdim=True) + 1e-8)
                    rep2 = torch.sum(outputs_b[0] * batch[4].unsqueeze(-1), 1) / (torch.sum(batch[4], 1, keepdim=True) + 1e-8)

                if args.similarity_function == "dot":
                    score = torch.sum(rep1 * rep2, -1) # now the score will be between -1 and 1
                elif args.similarity_function == "cosine":
                    score = F.cosine_similarity(rep1, rep2) # now the score will be between -1 and 1

                category_embeddings.append(rep1.data.cpu().numpy())
                text_embeddings.append(rep2.data.cpu().numpy())

            nb_eval_steps += 1
            if preds is None:
                preds.append(score.detach().cpu().numpy().reshape((-1)))
            else:
                preds.append(score.detach().cpu().numpy().reshape((-1)))


        preds = np.concatenate(preds, 0)
        preds = preds.reshape((-1, len(cats)))

        logger.info("save prediction file to eval_output_dir")

        out_file_name = ".".join(file_path.split("/")[-2:])
        text_embeddings = np.concatenate(text_embeddings, 0)
        category_embeddings = np.concatenate(category_embeddings, 0)
        np.savetxt(os.path.join(eval_output_dir, out_file_name + ".text.txt"), text_embeddings[::len(cats)])
        np.savetxt(os.path.join(eval_output_dir, out_file_name + ".category.txt"), category_embeddings[:len(cats)])
        np.savetxt(os.path.join(eval_output_dir, out_file_name + ".preds.txt"), preds)


    return results


def load_and_cache_examples(args, task, tokenizer, file_path, cat_file_path):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task](cat_file_path=cat_file_path)
    output_mode = output_modes[task]
    # cached_features_file = os.path.join(
    #     args.data_dir,
    #     "cached_{}_{}_{}".format(
    #         file_path,
    #         str(args.max_seq_length),
    #         str(task),
    #     ),
    # )
    # Load data features from cache or dataset file
    # if os.path.exists(cached_features_file) and not args.overwrite_cache:
    #     logger.info("Loading features from cached file %s", cached_features_file)
    #     features = torch.load(cached_features_file)
    # else:
    logger.info("Loading from dataset file at %s", file_path)
    label_list = processor.get_labels()
    # if args.task_name == "nyt":
    if args.multi_class:
        examples = (processor.get_examples(file_path, args.label_filepath))
    else:
        examples = (processor.get_examples(file_path))
     
    logger.info("Encoding features from dataset file at %s", file_path)
    # if args.local_rank in [-1, 0]:
    #     logger.info("Saving features into cached file %s", cached_features_file)
    #     torch.save(features, cached_features_file)
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
    )

    # Convert to Tensors and build dataset
    all_input_ids_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
    all_attention_mask_a = torch.tensor([f.attention_mask_a for f in features], dtype=torch.long)
    all_token_type_ids_a = torch.tensor([f.token_type_ids_a if f.token_type_ids_a is not None else [0]*len(f.attention_mask_a) for f in features], dtype=torch.long)
    all_input_ids_b = torch.tensor([f.input_ids_b for f in features], dtype=torch.long)
    all_attention_mask_b = torch.tensor([f.attention_mask_b for f in features], dtype=torch.long)
    all_token_type_ids_b = torch.tensor([f.token_type_ids_b if f.token_type_ids_b is not None else [0]*len(f.attention_mask_b) for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, all_input_ids_b, all_attention_mask_b, all_token_type_ids_b, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--eval_data_dir",
        default=None,
        type=str,
        required=True,
        help="The directory containing the evaluation dataset",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        required=True,
        help="The directory containing the evaluation dataset",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--label_filepath",
        default=None,
        type=str,
        help="Path to the label file for the nyt dataset",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--bert_representation",
        default="pool",
        choices=["avg", "pool"],
        type=str,
        help="The BERT representation type",
    )
    parser.add_argument(
        "--similarity_function",
        default="pool",
        choices=["dot", "cosine"],
        type=str,
        help="The similarity scoring function",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--multi_class", action="store_true", help="Whether it is a multi class classfication task.")
    parser.add_argument(
        "--all_cats_file", default=None, type=str, help="The file containing all category names",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        # set to load the latest checkpoint for training
        args.model_name_or_path = args.output_dir
        all_model_checkpoints = [ckpt for ckpt in os.listdir(args.model_name_or_path) if os.path.isdir(os.path.join(args.model_name_or_path, ckpt))]
        all_model_checkpoints = [(ckpt.split("-")[-1] if "-" in ckpt else -1, ckpt) for ckpt in all_model_checkpoints]
        all_model_checkpoints.sort(reverse=True)
        args.model_name_or_path = os.path.join(args.model_name_or_path, all_model_checkpoints[0][1])
        logger.info("setting to load the model from %s", args.model_name_or_path)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    args.output_mode = output_modes[args.task_name]
    num_labels = 2

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        file_path = os.path.join(args.eval_data_dir, "test.csv")
        # if args.task_name == "nyt":
        if args.all_cats_file is not None:
            all_cats_file = args.all_cats_file
        else:
            all_cats_file = os.path.join(args.eval_data_dir, "classes.txt.acl")

        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            # result = evaluate(args, model, tokenizer, file_path, all_cats_file, prefix=prefix)
            result = evaluate(args, model, tokenizer, args.eval_data_file, all_cats_file, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
