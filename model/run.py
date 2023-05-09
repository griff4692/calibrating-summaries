#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
from collections import defaultdict
import json
import logging
import math
import regex as re
import os
import itertools
import random
from typing import Optional
import ujson
from glob import glob

import datasets
import nltk
import numpy as np
import torch
from datasets import load_from_disk, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.tracking import GeneralTracker
from filelock import FileLock
import torch.nn as nn
from transformers import (
    MODEL_MAPPING,
    DataCollatorForSeq2Seq,
    DataCollatorForContrastSeq2Seq,
    SchedulerType,
    get_scheduler,
    T5Tokenizer,
    LongT5ForConditionalGeneration,
    LongT5Config,
    LEDConfig,
    LEDForConditionalGeneration,
    AutoTokenizer
)
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import is_offline_mode
from transformers.utils.versions import require_version
import wandb
from tqdm import tqdm


# Need to update if you change abstract/evals/run
CONTRAST_METRIC_LIBRARY = {
    'rouge1',
    'rouge2',
    'rougeL',
    'rougeLsum',
    'coverage',
    'density',
    'compression',
    'bs_src_recall',
    'bs_src_precision',
    'bs_src_f1',
    'bs_ref_recall',
    'bs_ref_precision',
    'bs_ref_f1',
    'bart_score',
    'fact_score',
    'num_prediction_tokens'
}


def clean_uuid(uuid):
    clean = re.sub(r'\W+', '_', uuid)
    return re.sub(r'_+', '_', clean).strip('_')


def load_accelerator_state_relaxed(config, input_dir, models, optimizers, schedulers, process_index, scaler=None, load_optimizer=True):
    """
    Loads states of the models, optimizers, scaler, and RNG generators from a given directory.
    Args:
        input_dir (`str` or `os.PathLike`):
            The name of the folder to load all relevant weights and states.
        models (`List[torch.nn.Module]`):
            A list of model instances
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        process_index (`int`):
            The current process index in the Accelerator state
        scaler (`torch.cuda.amp.GradScaler`, *optional*):
            An optional *GradScaler* instance to load
    """
    # Model states
    MODEL_NAME = 'pytorch_model'
    OPTIMIZER_NAME = 'optimizer'
    SCHEDULER_NAME = 'scheduler'
    SCALER_NAME = 'scaler'
    RNG_STATE_NAME = 'random_states'

    for i, model in enumerate(models):
        weights_name = f"{MODEL_NAME}.bin" if i == 0 else f"{MODEL_NAME}_{i}.bin"
        input_model_file = os.path.join(input_dir, weights_name)
        # ONLY LINE CHANGED to allow for new parameters to be added to support contrastive learning -> strict = False
        state_dict = torch.load(input_model_file, map_location="cpu")
        if hasattr(models[i], 'module'):
            models[i].load_state_dict({'module.' + k: v for k, v in state_dict.items()}, strict=config.contrastive_classifier is False)
        else:
            models[i].load_state_dict(state_dict, strict=config.contrastive_classifier is False)
    logger.info("All model weights loaded successfully")

    if load_optimizer:
        # Optimizer states
        for i, opt in enumerate(optimizers):
            optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
            input_optimizer_file = os.path.join(input_dir, optimizer_name)
            optimizers[i].load_state_dict(torch.load(input_optimizer_file, map_location="cpu"))
        logger.info("All optimizer states loaded successfully")

        # Scheduler states
        for i, scheduler in enumerate(schedulers):
            scheduler_name = f"{SCHEDULER_NAME}.bin" if i == 0 else f"{SCHEDULER_NAME}_{i}.bin"
            input_scheduler_file = os.path.join(input_dir, scheduler_name)
            scheduler.load_state_dict(torch.load(input_scheduler_file))
        logger.info("All scheduler states loaded successfully")

        # GradScaler state
        if scaler is not None:
            input_scaler_file = os.path.join(input_dir, SCALER_NAME)
            scaler.load_state_dict(torch.load(input_scaler_file))
            logger.info("GradScaler state loaded successfully")
    else:
        logger.info('Skipping loading optimizer states. Probably starting a fresh run.')

    # Random states
    states = torch.load(os.path.join(input_dir, f"{RNG_STATE_NAME}_{process_index}.pkl"))
    random.setstate(states["random_state"])
    np.random.set_state(states["numpy_random_seed"])
    torch.set_rng_state(states["torch_manual_seed"])
    torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
    logger.info("All random states loaded successfully")


def label_smoothed_nll_loss(lprobs, labels, epsilon=0.1, ignore_index=-100):
    if labels.dim() == lprobs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(ignore_index)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    labels = torch.clamp(labels, min=0)
    nll_loss = lprobs.gather(dim=-1, index=labels)
    # works for fp16 input tensor too, by internally upcasting it to fp32
    smoothed_loss = lprobs.sum(dim=-1, keepdim=True, dtype=torch.float32)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * lprobs.shape[-1])
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


def label_smoothed_unlikelihood(probs, targets):
    probs = probs.view(-1, probs.size(-1))
    one_minus_probs = torch.clamp(1.0 - probs, min=1e-10)
    lprobs = -torch.log(one_minus_probs)
    targets = targets.view(-1, 1)
    return label_smoothed_nll_loss(lprobs, targets, ignore_index=-100)


# from model.utils import add_global_attention_mask
def add_global_attention_mask(batch):
    global_attention_mask = torch.zeros_like(batch['input_ids']).to(batch['input_ids'].device)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    batch['global_attention_mask'] = global_attention_mask


logger = get_logger(__name__)
dirname = os.path.dirname(__file__)
DATA_DIR = os.path.expanduser('~/data_tmp')
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
T5_MODEL = 'google/long-t5-tglobal-base'
PRIMERA_MODEL = 'allenai/PRIMERA'
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


class CustomWandBTracker(GeneralTracker):
    """
    A `Tracker` class that supports `wandb`. Should be initialized at the start of your script.
    Args:
        run_name (`str`):
            The name of the experiment run.
    """

    requires_logging_directory = False

    def __init__(self, run_name: str, experiment_name: str, entity: str):
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.entity = entity
        self.run = wandb.init(project=self.run_name, name=experiment_name, entity=self.entity)
        logger.info(
            f"Initialized WandB project {self.run_name} for entity {self.entity} with name {self.experiment_name}"
        )
        logger.info(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.
        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        wandb.config.update(values)
        logger.info("Stored initial configuration hyperparameters to WandB")

    def log(self, values: dict, step: Optional[int] = None):
        """
        Logs `values` to the current run.
        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        self.run.log(values, step=step)
        logger.info(f"Successfully logged to WandB for step={step}")

    def finish(self):
        """
        Closes `wandb` writer
        """
        self.run.finish()
        logger.info("WandB run closed")

    def name(self):
        return 'CustomWandB'

    def tracker(self):
        return self.run


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument('-skip_inference', default=False, action='store_true')
    parser.add_argument('-fp16', default=False, action='store_true')
    parser.add_argument('--log_every_n_steps', default=10, type=int)
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument('-use_deepspeed', default=False, action='store_true')
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=1024,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
        default='facebook/bart-base'
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        # help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default='linear',
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=2000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument('--dataset', default='pubmed', choices=['pubmed', 'clinical', 'chemistry'])
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(DATA_DIR, 'weights'),
        help="Where to store the final model."
    )
    parser.add_argument('--experiment', type=str, default='abstract_gen', help='Name of experiment')
    parser.add_argument('--wandb_project', default=None)
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('-disable_wandb', default=False, action='store_true')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        '--hf_model', default='primera', choices=['t5', 'primera']  # t5 = long T5
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument('-start_fresh', default=False, action='store_true')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('-no_gradient_clip', default=False, action='store_true')
    parser.add_argument('--validate_every_n_steps', default=1000, type=int)
    parser.add_argument('--max_val_examples', default=2048, type=int)
    parser.add_argument('-save_every_time', default=False, action='store_true')
    parser.add_argument('--exit_after_n_steps', default=999999999, type=int)

    # Contrast hyper-parameters
    parser.add_argument('-contrast', default=False, action='store_true')
    parser.add_argument('-build_contrast_exp_name', default=False, action='store_true')
    parser.add_argument('--contrast_ckpt', default=None)
    parser.add_argument('--contrast_metrics', default='faithful')
    parser.add_argument('--max_num_positive', default=3, type=int)
    parser.add_argument('--max_num_negative', default=3, type=int)
    parser.add_argument('--max_num_rank', default=3, type=int)
    parser.add_argument(
        '--contrast_intra_sample_strategy', default='random',
        choices=[
            'random', 'max_margin', 'min_margin', 'max_diversity', 'min_diversity', 'top_beam', 'bottom_beam',
            'wide_beam', 'min_metric', 'max_metric', 'max_gap', 'min_gap', 'avg_margin', 'max_length', 'min_length',
            'easy', 'hard', 'max_surprise', 'min_surprise', 'max_extractive_gap', 'max_faithful'
        ],
    )
    parser.add_argument('--contrast_objective', default='unlikelihood',
        choices=['unlikelihood', 'margin_rank', 'contrast', 'positive_distillation']
    )
    parser.add_argument('--positive_methods', default='all')
    parser.add_argument('--mixed_methods', default='all')
    parser.add_argument('-use_mixed_methods', default=False, action='store_true')
    parser.add_argument('--negative_methods', default='all')
    parser.add_argument('--reference_status', default='positive', choices=['ensure', 'remove', 'positive'])
    # For ranking objective
    # Table 13 lambda https://arxiv.org/pdf/2203.16804.pdf this is 0.001
    parser.add_argument('--contrast_rank_margin', default=0.001, type=float)
    parser.add_argument('--length_penalty', default=1.0, type=float)
    parser.add_argument('--margin_scale', type=float, default=0.01)
    parser.add_argument('--mle_weight', default=1.0, type=float)
    parser.add_argument('--contrast_weight', default=1.0, type=float)

    args = parser.parse_args()
    if args.contrast:
        args.log_every_n_steps = 2
        if args.contrast_ckpt is not None:
            args.resume_from_checkpoint = os.path.join(args.output_dir, args.contrast_ckpt, 'best_ckpt')
            logger.info(f'Starting contrastive fine-tuning from {args.resume_from_checkpoint}')

    args.output_dir = os.path.join(args.output_dir, args.experiment)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Saving all outputs to {args.output_dir}')

    return args


def main():
    args = parse_args()

    mp = 'fp16' if args.fp16 else None

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    if args.debug or args.disable_wandb:
        trackers = []
    else:
        trackers = [CustomWandBTracker(args.wandb_project, args.experiment, args.wandb_entity)]
    accelerator = Accelerator(log_with=trackers, logging_dir=args.output_dir, cpu=args.cpu, mixed_precision=mp)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    data_path = os.path.join(DATA_DIR, args.dataset, f'{args.hf_model}_splits')
    print(f'Loading custom dataset from {data_path}')
    raw_datasets = load_from_disk(data_path)

    contrast_dir = None
    if args.contrast:
        if args.dataset == 'chemistry':
            raw_datasets['train'] = raw_datasets['train'].map(lambda example: {'uuid': clean_uuid(example['uuid'])})
            raw_datasets['validation'] = raw_datasets['validation'].map(lambda example: {'uuid': clean_uuid(example['uuid'])})

        contrast_dir = os.path.join(DATA_DIR, args.dataset, 'corruptions')
        print(f'Loading in corruptions from {contrast_dir}')
        label_smoother = LabelSmoother(0.1)  # For margin rank contrastive learning

        def get_uuid_from_fn(fn):
            return fn.split('/')[-1].replace('.json', '')

        # Filter for available uuids
        train_pattern = os.path.join(contrast_dir, 'train', '*.json')
        val_pattern = os.path.join(contrast_dir, 'validation', '*.json')
        train_fns = list(glob(train_pattern))
        val_fns = list(glob(val_pattern))

        train_uuid_set = set(list(map(get_uuid_from_fn, train_fns)))
        val_uuid_set = set(list(map(get_uuid_from_fn, val_fns)))

        train_uuids = raw_datasets['train']['uuid']
        val_uuids = raw_datasets['validation']['uuid']

        keep_train_idxs = [i for i, uuid in enumerate(train_uuids) if uuid in train_uuid_set]
        keep_val_idxs = [i for i, uuid in enumerate(val_uuids) if uuid in val_uuid_set]
        raw_datasets['train'] = raw_datasets['train'].select(keep_train_idxs)
        raw_datasets['validation'] = raw_datasets['validation'].select(keep_val_idxs)

    if args.hf_model == 't5':
        tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
        config = LongT5Config.from_pretrained(T5_MODEL)
        config.contrastive_classifier = args.contrast and args.contrastive_objective == 'contrast'
        model = LongT5ForConditionalGeneration.from_pretrained(T5_MODEL, config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(PRIMERA_MODEL)
        config = LEDConfig.from_pretrained(PRIMERA_MODEL)
        config.contrastive_classifier = args.contrast and args.contrast_objective == 'contrast'
        model = LEDForConditionalGeneration.from_pretrained(PRIMERA_MODEL, config=config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    all_cols = list(raw_datasets['train'].features)
    keep_cols = ['input_ids', 'attention_mask', 'labels']
    if args.contrast:
        keep_cols.append('uuid')
    remove_cols = [x for x in all_cols if x not in keep_cols]
    train_dataset = raw_datasets['train'].remove_columns(remove_cols)
    eval_dataset = raw_datasets['validation'].remove_columns(remove_cols)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if args.hf_model == 'primera':
        pad_multiple = max(model.config.attention_window)
    else:
        pad_multiple = 8 if accelerator.use_fp16 else None

    if args.contrast:
        metric_norm_fn = os.path.join(DATA_DIR, f'{args.dataset}_metric_bounds.json')
        with open(metric_norm_fn, 'r') as fd:
            stats = ujson.load(fd)
        if args.contrast_metrics == 'faithful':
            contrast_metrics = ['bs_src_precision', 'fact_score', 'bart_score']
        elif args.contrast_metrics == 'relevance':
            contrast_metrics = ['bs_ref_f1', 'rouge1', 'rouge2']
        else:
            contrast_metrics = args.contrast_metrics.split(',')

        def score_candidate_fn(row):
            norm_vals = []
            for metric in contrast_metrics:
                stat = stats[metric]
                norm_vals.append((row[metric] - stat['mean']) / stat['std'])
            return sum(norm_vals) / len(norm_vals)
        logger.info(contrast_metrics)

        assert all([x in CONTRAST_METRIC_LIBRARY for x in contrast_metrics])

        train_data_collator = DataCollatorForContrastSeq2Seq(
            tokenizer,
            positive_methods=args.positive_methods,
            negative_methods=args.negative_methods,
            mixed_methods=args.mixed_methods,
            use_mixed_methods=args.use_mixed_methods,
            reference_status=args.reference_status,
            set_type='soft' if args.contrast_objective == 'margin_rank' else 'hard',
            contrast_dir=contrast_dir,
            split='train',
            score_candidate_fn=score_candidate_fn,
            max_num_positive=args.max_num_positive,
            max_num_negative=args.max_num_negative,
            max_num_rank=args.max_num_rank,
            max_target_length=args.max_target_length,
            contrast_sample_strategy=args.contrast_intra_sample_strategy,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=pad_multiple,
        )

        val_data_collator = DataCollatorForContrastSeq2Seq(
            tokenizer,
            positive_methods=args.positive_methods,
            negative_methods=args.negative_methods,
            mixed_methods=args.mixed_methods,
            use_mixed_methods=args.use_mixed_methods,
            reference_status=args.reference_status,
            set_type='soft' if args.contrast_objective == 'margin_rank' else 'hard',
            contrast_dir=contrast_dir,
            split='validation',
            score_candidate_fn=score_candidate_fn,
            max_num_positive=args.max_num_positive,
            max_num_negative=args.max_num_negative,
            max_num_rank=args.max_num_rank,
            max_target_length=args.max_target_length,
            contrast_sample_strategy=args.contrast_intra_sample_strategy,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=pad_multiple,
        )
    else:
        train_data_collator = val_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=pad_multiple,
        )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ['\n'.join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ['\n'.join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    num_workers = 0 if args.debug else 16
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=train_data_collator, batch_size=args.per_device_train_batch_size,
        num_workers=num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=val_data_collator, batch_size=args.per_device_eval_batch_size,
        num_workers=num_workers
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ['bias', 'LayerNorm.weight']
    non_contrast_np = [(n, p) for n, p in model.named_parameters() if 'contrast_projection' not in n]
    contrast_np = [(n, p) for n, p in model.named_parameters() if 'contrast_projection' in n]
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in non_contrast_np if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'lr': args.learning_rate
        },
        {
            'params': [p for n, p in non_contrast_np if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.learning_rate
        },
    ]

    if args.contrast and args.contrast_objective == 'contrast':
        assert len(contrast_np) > 0
        optimizer_grouped_parameters.append({
            'params': [p for n, p in contrast_np],
            'weight_decay': 0.0,
            'lr': 1e-3
        })
    else:
        assert len(contrast_np) == 0

    if args.use_deepspeed:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters)
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if accelerator.is_main_process:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers('recipe_structured', experiment_config)

    # Metrics
    metric = load_metric('rouge')

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    validation_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        logger.info(f"Loading states from {args.resume_from_checkpoint}")
        load_accelerator_state_relaxed(
            config, args.resume_from_checkpoint, accelerator._models, accelerator._optimizers, accelerator._schedulers,
            accelerator.state.process_index, accelerator.scaler,
            load_optimizer=not args.contrast and not args.start_fresh  # We are starting a new run
        )

        step_fn = os.path.join(args.resume_from_checkpoint, 'step.json')
        if args.contrast or not os.path.exists(step_fn) or args.start_fresh:
            resume_step = completed_steps = 0
        else:
            with open(step_fn, 'r') as fd:
                both_steps = ujson.load(fd)
                resume_step = both_steps['step']
                completed_steps = both_steps['completed_step']

    def contrast_step(outputs, gold_labels, contrast_labels, contrast_cutoff):
        contrast_losses = {}
        contrast_stats = defaultdict(list)

        encoder_h = outputs.encoder_last_hidden_state
        bsize, c_set_size, target_len = contrast_labels.size()
        num_contrast = bsize * c_set_size
        if args.contrast_objective == 'positive_distillation':
            contrast_inputs = {
                'encoder_outputs': [encoder_h],
                'labels': contrast_labels[:, 0],  # The most positive example is the teacher label
            }

            contrast_losses['distilled'] = model(**contrast_inputs).loss
            return contrast_losses, contrast_stats

        encoder_h_rep = encoder_h.unsqueeze(1).repeat(1, c_set_size, 1, 1).contiguous()
        _, _, encoder_seq_len, model_dim = encoder_h_rep.size()
        encoder_h_flat = encoder_h_rep.view(num_contrast, encoder_seq_len, model_dim)

        contrast_inputs = {
            'encoder_outputs': [encoder_h_flat],
            'decoder_input_ids': model.prepare_decoder_input_ids_from_labels(contrast_labels.view(num_contrast, -1)),
            'output_hidden_states': args.contrast_objective == 'contrast'
        }

        contrast_output = model(**contrast_inputs)
        contrast_logits = contrast_output.logits.view(bsize, c_set_size, target_len, -1)

        if contrast_cutoff is None:  # Split it down the middle
            contrast_cutoff = c_set_size // 2
        pos_idxs = list(range(0, contrast_cutoff))
        neg_idxs = list(range(contrast_cutoff, c_set_size))

        if args.contrast_objective == 'unlikelihood':
            contrast_losses['likelihood'] = label_smoother(
                {'logits': contrast_logits[:, pos_idxs].view(-1, contrast_logits.size()[-1])},
                contrast_labels[:, pos_idxs].view(-1)
            )
            probs_neg = torch.softmax(contrast_logits[:, neg_idxs], dim=-1)
            unlikelihood_smooth_loss = label_smoothed_unlikelihood(probs_neg.view(-1, probs_neg.size()[-1]), contrast_labels[:, neg_idxs].view(-1))
            contrast_losses['unlikelihood'] = unlikelihood_smooth_loss
        elif args.contrast_objective == 'contrast':
            all_contrast_nll = []
            pos_contrasts = list(itertools.combinations(pos_idxs, 2))
            # Heavily borrowed from CLIFF Github
            # https://github.com/ShuyangCao/cliff_summ/blob/8913d92f85457e030d77dc5dfa255bea7e226dc4/models/pegasus/contrastive_trainer.py
            decoder_states = contrast_output.decoder_hidden_states[-1]
            decoder_proj = model.contrast_projection(decoder_states).view(bsize, c_set_size, target_len, -1)
            decoder_proj_mask = contrast_labels.unsqueeze(-1) == -100

            decoder_proj.masked_fill_(decoder_proj_mask, 0)
            decoder_pooled = decoder_proj.sum(dim=2) / (contrast_labels != -100).sum(dim=-1, keepdim=True)
            for batch_idx in range(bsize):
                states = decoder_pooled[batch_idx]
                states_norm = states / states.norm(dim=-1, keepdim=True)
                cosine_sim = torch.matmul(states_norm, states_norm.transpose(0, 1))
                inverted_identity = 1 - torch.eye(len(cosine_sim), device=model.device)
                cosine_sim_exp = cosine_sim.exp() * inverted_identity
                denom = cosine_sim_exp.sum(dim=1)
                contrast_nll = 0.0
                for a, b in pos_contrasts:
                    exp_sim = cosine_sim_exp[a, b]
                    contrast_nll = contrast_nll - torch.log(exp_sim / denom[a])

                contrast_nll = contrast_nll / len(pos_contrasts)
                all_contrast_nll.append(contrast_nll)
            contrast_losses['contrast_nll'] = torch.stack(all_contrast_nll).mean()
        elif args.contrast_objective == 'margin_rank':  # BRIO https://arxiv.org/pdf/2203.16804.pdf
            # Code borrowed from https://github.com/yixinl7/brio
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            V = config.vocab_size
            nll = loss_fct(contrast_output.logits.view(-1, V), contrast_labels.view(-1)).view(
                bsize, c_set_size, target_len
            )
            seq_lens = (contrast_labels > -100).sum(dim=2)

            scores = (- nll.sum(dim=2) / seq_lens ** args.length_penalty) * args.margin_scale

            contrast_loss = 0
            for cand_idx in range(1, c_set_size):
                pos_score = scores[:, :-cand_idx]
                neg_score = scores[:, cand_idx:]
                pos_score = pos_score.contiguous().view(-1)
                neg_score = neg_score.contiguous().view(-1)
                ones = torch.ones_like(pos_score)
                loss_func = torch.nn.MarginRankingLoss(args.contrast_rank_margin * cand_idx)
                loss = loss_func(pos_score, neg_score, ones)
                contrast_loss += loss

            predicted_idx = np.mean(scores.argmax(dim=1).cpu().numpy())
            contrast_stats['predicted_rank_idx'].append(predicted_idx)

            if args.reference_status == 'ensure':
                gold_nll = loss_fct(outputs.logits.view(-1, V), gold_labels.view(-1)).view(bsize, -1)
                gold_lens = (gold_labels > -100).sum(dim=1, keepdim=True)
                gold_scores = (- gold_nll.sum(dim=1, keepdim=True) / gold_lens) * args.margin_scale
                pos_score = gold_scores.expand_as(scores)
                neg_score = scores
                pos_score = pos_score.contiguous().view(-1)
                neg_score = neg_score.contiguous().view(-1)
                ones = torch.ones_like(pos_score)
                gold_margin = 0
                loss_func = torch.nn.MarginRankingLoss(gold_margin)
                gold_weight = 1.0
                contrast_loss += gold_weight * loss_func(pos_score, neg_score, ones)
            contrast_losses['contrast_rank_loss'] = contrast_loss
        else:
            raise Exception('Not implemented yet!')

        for k, v in contrast_stats.items():
            if type(v) == list:
                contrast_stats[k] = np.mean(v)
        return contrast_losses, contrast_stats
    
    def run_validation(steps=None):
        print('Starting validation run...')
        logger.info('Starting validation run...')
        model.eval()
        gen_kwargs = {
            'max_length': args.max_target_length,
            'num_beams': args.num_beams, 'no_repeat_ngram_size': 3,
        }
        samples_seen = 0
        val_losses = []
        contrast_losses = defaultdict(list)
        contrast_stats = defaultdict(list)
        max_val_steps = (args.max_val_examples // validation_batch_size) + 1
        max_val_steps = max_val_steps if steps is None else min(max_val_steps, steps)
        for step, batch in tqdm(enumerate(eval_dataloader), total=max_val_steps):
            if args.contrast:
                batch, contrast_batch = batch
                contrast_labels = contrast_batch['labels']
                bsize = len(batch['input_ids'])
                num_contrast = len(contrast_labels)
                c_set_size = num_contrast // bsize
                assert num_contrast % bsize == 0
                contrast_labels = contrast_labels.view(bsize, c_set_size, -1)

            if args.hf_model == 'primera':
                add_global_attention_mask(batch)
                gen_kwargs['global_attention_mask'] = batch['global_attention_mask']
            with torch.no_grad():
                outputs = model(**batch)
                mle_loss = outputs.loss

                # Contrast Objective
                if args.contrast:
                    cl, cl_stats = contrast_step(
                        outputs, batch['labels'], contrast_labels, contrast_cutoff=args.max_num_positive
                    )
                    for k, v in cl.items():
                        contrast_losses[k].append(v.detach().float().item())
                    for k, v in cl_stats.items():
                        contrast_stats[k].append(float(v))

                val_losses.append(mle_loss.detach().float().item())

                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch['labels']
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(eval_dataloader) - 1:
                        decoded_preds = decoded_preds[: len(eval_dataloader.dataset) - samples_seen]
                        decoded_labels = decoded_labels[: len(eval_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += len(decoded_labels)

                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
                if step == max_val_steps:
                    break
        val_loss = np.mean(val_losses)
        result = metric.compute(use_stemmer=True)
        # Extract a few results from ROUGE
        result = {'validation/' + key: value.mid.fmeasure * 100 for key, value in result.items()}
        result['validation/loss'] = val_loss

        loss_cols = ['validation/loss']
        for k, v in contrast_losses.items():
            mean_v = np.mean(np.array(v))
            result['validation/' + k] = mean_v
            loss_cols.append('validation/' + k)
        for k, v in contrast_stats.items():
            mean_v = np.mean(np.array(v))
            result['validation/' + k] = mean_v
        result = {k: round(v, 4) for k, v in result.items()}

        logger.info(result)
        return result, loss_cols

    min_val_loss = 1e3
    if accelerator.is_main_process and not args.debug:
        tokenizer_dir = os.path.join(args.output_dir, 'tokenizer')
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)

    ckpt_dir = os.path.join(args.output_dir, 'best_ckpt')
    last_dir = os.path.join(args.output_dir, 'last_ckpt')

    # Sanity check the validation steps
    if not args.debug:
        result, _ = run_validation(steps=5)
        accelerator.log(result, step=0)
        logger.info(result)
    logger.info(f'Starting at epoch {starting_epoch}/{args.num_train_epochs}')
    is_done = False
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        last_n_losses = []
        logger.info(f'Starting epoch {epoch} now...')
        for step, batch in enumerate(train_dataloader):
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                progress_bar.update(1)
                completed_steps += 1

            contrast_labels = None
            if args.contrast:
                batch, contrast_batch = batch

                bsize = len(batch['input_ids'])
                contrast_labels = contrast_batch['labels']
                num_contrast = len(contrast_labels)
                c_set_size = num_contrast // bsize
                assert num_contrast % bsize == 0
                contrast_labels = contrast_labels.view(bsize, c_set_size, -1)

            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    continue
            if args.hf_model == 'primera':
                add_global_attention_mask(batch)
            outputs = model(**batch)
            mle_loss = outputs.loss

            detached_loss = mle_loss.detach().float().item()
            if len(last_n_losses) < args.log_every_n_steps:
                last_n_losses.append(detached_loss)
            else:
                last_n_losses = last_n_losses[1:] + [detached_loss]

            if args.debug:
                logger.info(f'Train MLE Loss: {mle_loss}')
            mle_loss = mle_loss / args.gradient_accumulation_steps
            optimizer_loss = args.mle_weight * mle_loss

            # Contrast Objective
            if args.contrast:
                contrast_losses, contrast_stats = contrast_step(
                    outputs, batch['labels'], contrast_labels, contrast_cutoff=args.max_num_positive
                )
                effective_coeff = args.contrast_weight / (len(contrast_losses) * args.gradient_accumulation_steps)
                for k, v in contrast_losses.items():
                    optimizer_loss += effective_coeff * v
                    if args.debug:
                        logger.info(f'Train Contrast {k} Loss: {v}')

            accelerator.backward(optimizer_loss)
            if not args.no_gradient_clip:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if completed_steps % args.log_every_n_steps == 0:
                    accelerator.log({'train/loss': np.mean(last_n_losses)}, step=completed_steps)
                    if args.contrast:
                        for k, v in contrast_losses.items():
                            accelerator.log({f'train/{k}': v.detach().float().item()}, step=completed_steps)
                        for k, v in contrast_stats.items():
                            accelerator.log({f'train/{k}': float(v)}, step=completed_steps)

                if completed_steps % args.validate_every_n_steps == 0:
                    result, loss_keys = run_validation()
                    accelerator.log(result, step=completed_steps)
                    monitor_val = np.mean([result[k] for k in loss_keys])
                    if monitor_val <= min_val_loss or args.save_every_time:
                        logger.info(
                            f'Validation loss improved from {min_val_loss} to {monitor_val}. '
                            f'Saving weights to {ckpt_dir}'
                        )
                        if not args.debug:
                            os.makedirs(ckpt_dir, exist_ok=True)

                            if args.save_every_time:
                                ckpt_dir = os.path.join(args.output_dir, f'ckpt_{completed_steps}_steps')
                                os.makedirs(ckpt_dir, exist_ok=True)
                            else:
                                ckpt_dir = os.path.join(args.output_dir, 'best_ckpt')
                            accelerator.save_state(ckpt_dir)

                            step_fn = os.path.join(ckpt_dir, 'step.json')
                            with open(step_fn, 'w') as fd:
                                ujson.dump({'step': step, 'completed_step': completed_steps, 'epoch': epoch}, fd)

                            min_val_loss = monitor_val
                            if args.save_every_time:
                                results_fn = os.path.join(args.output_dir, f'results_{completed_steps}_steps.json')
                            else:
                                results_fn = os.path.join(args.output_dir, 'best_results.json')
                            with open(results_fn, 'w') as f:
                                json.dump(
                                    {
                                        "eval_rouge1": result["validation/rouge1"],
                                        "eval_rouge2": result["validation/rouge2"],
                                        "eval_rougeL": result["validation/rougeL"],
                                        "eval_rougeLsum": result["validation/rougeLsum"],
                                    },
                                    f,
                                )
                    else:
                        logger.info(
                            f'Validation loss did not improve: from {min_val_loss} to {monitor_val}. Not saving.')
            if completed_steps >= args.max_train_steps or completed_steps >= args.exit_after_n_steps:
                logger.info(
                    f'Completed {completed_steps}/{args.max_train_steps} steps. Breaking out of training loop now.'
                )
                is_done = True
                break
        if is_done:
            break

    if not args.save_every_time:
        result, loss_keys = run_validation()
        accelerator.log(result, step=completed_steps)
        monitor_val = np.mean([result[k] for k in loss_keys])
        logger.info(f'Final validation loss from {min_val_loss} to {monitor_val}. Saving weights to {ckpt_dir}')
        if not args.debug:
            os.makedirs(last_dir, exist_ok=True)
            accelerator.save_state(last_dir)
            step_fn = os.path.join(last_dir, 'step.json')
            with open(step_fn, 'w') as fd:
                ujson.dump({'step': step, 'completed_step': completed_steps}, fd)
            with open(os.path.join(args.output_dir, 'last_results.json'), "w") as f:
                json.dump(
                    {
                        "eval_rouge1": result["validation/rouge1"],
                        "eval_rouge2": result["validation/rouge2"],
                        "eval_rougeL": result["validation/rougeL"],
                        "eval_rougeLsum": result["validation/rougeLsum"],
                    },
                    f,
                )


if __name__ == '__main__':
    main()
