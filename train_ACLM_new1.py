#coding=utf-8

import pickle
import os
import os.path
import argparse
from tqdm import tqdm
from itertools import count
from socket import gethostname
from tokenizers import Tokenizer
from statistics import mean
import json
import math
import copy
from datetime import timedelta
import numpy as np
import pandas as pd
import random

#from config import default_args
from lmprobs import TrigramSurprisalSpace
from lmprobs import AbstractSurprisalSpace
from sklearn.neighbors import KDTree

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel 
import torch.nn.functional as F

from lamb import Lamb
from model_extra import Bert
from utils import cosine_schedule_with_warmup_cooldown, is_main_process, get_rank, seed_everything, get_world_size
from dataset_ACLM_new import MaskedDataset, CausalDataset, ValidationDataset
from model_logging import ModelLogger


if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="/mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/babyLM2026/gpt-bert/data/train_10M_4k.txt", type=str, help="Path to the training data.")
    parser.add_argument("--valid_path", default="/mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/babyLM2026/gpt-bert/data/dev_10M_4k.txt", type=str, help="Path to the validation data.")
    parser.add_argument("--name", default="1_4_4k_bs_64_babylm_ACLM_10M", type=str, help="Name of the run.")
    parser.add_argument("--config_file", default="/mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/babyLM2026/gpt-bert/configs/small.json", type=str, help="The BERT model config")
    parser.add_argument("--tokenizer_path", default="/mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/babyLM2026/gpt-bert/tokenizers/tokenizer_10M_4k.json", type=str, help="Path to the tokenizer.")
    parser.add_argument("--output_dir", default="/mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/babyLM2026/gpt-bert/model_checkpoints", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--checkpoint_filename", default=None, type=str, help="The checkpoint filename to resume training.")
    parser.add_argument("--optimizer", default="lamb", type=str, help="The optimizer to use.")
    parser.add_argument("--hybrid_numerator", default=1, type=int, help="The numerator of the hybrid ratio.")
    parser.add_argument("--hybrid_denominator", default=4, type=int, help="The denominator of the hybrid ratio (the number of GPUs should be divisible by this number).")
    parser.add_argument("--seq_length", default=128, type=int, help="Sequence length for training.")
    parser.add_argument("--local_batch_size", default=64, type=int, help="Batch size for training per GPU.")
    parser.add_argument("--global_batch_size", default=8192, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--batch_reduction", default=4, type=int, help="The initial batch size reduction factor.")
    parser.add_argument("--learning_rate", default=1.41e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=150, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="Exponential moving average decay.")
    parser.add_argument("--validate_every", default=50, type=int, help="Run validation after every X training shards.")
    parser.add_argument("--validation_steps", default=1, type=int, help="Number of validation steps.")
    parser.add_argument("--log_stats_every", default=50, type=int, help="Log stats every X steps.")
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cooldown_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate cooldown for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--save_every', type=int, default=1_000, help="save every X steps")
    parser.add_argument("--mask_p_start", default=0.3, type=float, help="Initial masking probability.")
    parser.add_argument("--mask_p_end", default=0.15, type=float, help="Final masking probability.")
    parser.add_argument("--mask_random_p", default=0.1, type=float, help="Probability of replacing the masked token with a random token.")
    parser.add_argument("--mask_keep_p", default=0.1, type=float, help="Probability of keeping the masked token.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon.")
    parser.add_argument("--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument("--optimizer_beta2", default=0.98, type=float, help="Optimizer beta2.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction, help="Mixed precision training.")
    parser.add_argument('--n_special_tokens', default=16, type=int, help="Number of special tokens.")
    parser.add_argument('--z_loss_weight', default=1e-4, type=float, help="Weight for the z loss.")
    parser.add_argument('--token_weighted_loss', default=False, action=argparse.BooleanOptionalAction, help="Use token weighted loss.")

     # ACLM arguments
    parser.add_argument("--csv_path",default="/mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/babyLM2026/gpt-bert/data/train_10M_4k.csv",type=str,help="The path to the ACLM training data CSV file.",)
    parser.add_argument("--aclm_init_size",default=64,type=int,help="The number of intial sample batches.",)
    parser.add_argument("--aclm_tss_sample_size",default=64,type=int,help="The number of sample batches to compute surprisals.",)
    parser.add_argument("--aclm_sample_per_iter",default=32,type=int,help="The number of sample batches per iteration. ",)
    args = parser.parse_args()

    args.output_path = f"{args.output_dir}/{args.name}.bin"

    return args

def setup_training(args, tokenizer):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert args.gpus_per_node == torch.cuda.device_count()
    assert torch.cuda.device_count() > 0, "No GPUs available for training."
    print(f"Hello from rank {args.rank} of {args.world_size} on {gethostname()} where there are {torch.cuda.device_count()} allocated GPUs per node.", flush=True)

    assert args.world_size % args.hybrid_denominator == 0

    # if args.rank / args.world_size < args.hybrid_numerator / args.hybrid_denominator:
    if args.rank * args.hybrid_denominator < args.hybrid_numerator * args.world_size:
        args.dataset_type = "masked"
    else:
        args.dataset_type = "causal"

    print(f"Dataset type: {args.dataset_type}", flush=True)

    seed_everything(args.seed + args.rank)
    torch.distributed.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size,timeout=timedelta(minutes=60))
    if args.rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)
    #args.local_rank = args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node)
    #args.local_rank = int(os.getenv("LOCAL_RANK", os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", os.environ["SLURM_LOCALID"]))) #args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    print(f"RCCL started on device {args.device}", flush=True)
    print(f"host: {gethostname()}, rank: {args.rank}, local_rank: {args.local_rank}")

    if is_main_process():
        print(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs")
        print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({get_world_size()}) x 'batch_size'({args.local_batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * get_world_size() * args.local_batch_size * args.seq_length:,} subword instances")

    args.vocab_size = tokenizer.get_vocab_size()

    if is_main_process():
        wandb.init(
            name=args.name,
            project="BabyLM-Eleni-Try", ##
            entity="gusripama-" ##
        )
        # Define custom metrics with word_count as x-axis
        wandb.define_metric("word_count")
        wandb.define_metric("train/*", step_metric="word_count")
        wandb.define_metric("validation/*", step_metric="word_count")
        wandb.define_metric("stats/*", step_metric="word_count")


def load_config(args):
    with open(args.config_file, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args):
    args = load_config(args)
    model = Bert(args)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(args)
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(args.device)

    no_decay = ['bias', 'layer_norm']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if is_main_process():
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print()
        print("Parameters with weight decay:")
        for n, _ in decay_params:
            print(n)
        print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )

    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps,
        0.1
    )
    #print("starting..")
    model = DistributedDataParallel(
        model,
        device_ids= [args.device], #[args.local_rank],
        output_device= args.device, 
        bucket_cap_mb=torch.cuda.get_device_properties(args.device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )

    ema_model: nn.Module = copy.deepcopy(model.module)
    for param in ema_model.parameters():
        param.requires_grad = False

    global_step, epoch = 0, 0
    if args.checkpoint_filename is not None:
        state_dict = torch.load(args.checkpoint_filename, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        ema_model.load_state_dict(state_dict["ema_model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        global_step = state_dict["global_step"]
        epoch = state_dict["epoch"]

    return model, ema_model, optimizer, scheduler, global_step, epoch


class GPTBertSurprisalSpace(AbstractSurprisalSpace):
    def __init__(self,dims, dataloader,model):
        super().__init__(dims)
        self.dataloader = dataloader
        self.model=model

    def train(self, sequences):
        return None

    def fit(self):
        self.surprisalvecs = self.surprisalizer_()
        self.currentsurprisalvecs = self.surprisalvecs.copy()
        self.filtered_to_original = list(range(len(self.surprisalvecs)))
        print("Building KD Tree.")
        self.nnfinder = KDTree(self.surprisalvecs)

        return self.surprisalvecs

    def surprisalizer_(self):
        surprisals = []
        dataloader = iter(self.dataloader)
        self.model.eval()
        with torch.no_grad():
            for local_step in tqdm(range(len(dataloader)),desc="ACLM num steps"):
                input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(dataloader,args.device, 0)
                #print(len(dataloader))
                with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                    prediction = model(input_ids_, attention_mask_, target_ids_,return_all=True)
                    #print(prediction.size())
                    
                    target_ids= target_ids_.flatten()
                    
                    loss = F.cross_entropy(
                            prediction, target_ids, reduction='none')
                    #print(loss.size())
                    
                    loss_np = loss.detach().cpu().numpy()
                    surprisals.append(loss_np)
                    #print(len(surprisals))
        return surprisals

def get_batch(dataloader, device, global_step):
    dataloader._dataset.set_global_step(global_step)
    batch = next(dataloader)
    input_ids, target_ids, attention_mask, mask_p = [t.pin_memory().to(device, non_blocking=True) for t in batch]
    input_ids, target_ids = input_ids.t(), target_ids.t()
    mask_p = mask_p.mean()

    return input_ids, attention_mask, target_ids, mask_p

def count_words_from_tokens(input_ids):
    text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)
    words = text.strip().split()
    return len(words)

def training_epoch(model, ema_model, train_dataloader, valid_dataloader, optimizer, scheduler, global_step, epoch, total_word_count, target_word_count, args):
    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    # calculate the number of steps to perform in this epoch
    num_steps = min(len(train_dataloader), (args.max_steps - global_step) * args.accumulate_steps)
    #print(num_steps)
    #print(f"the length: {len(train_dataloader)}")
    print(f"the accumulation steps: {args.accumulate_steps}")
    
    # initialize the dataloader and the metrics
    train_dataset = train_dataloader.dataset ########### added this
    print(f" The dataset is: {train_dataset} and the number of steps are: {num_steps}, the rank is: {args.rank}")
    train_dataloader = iter(train_dataloader)
    #print(train_dataloader)
    total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0
    # get the first batch
    input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(train_dataloader, args.device, global_step)
    print(f" getting batch, rank: {args.rank}")
    
    # iterate over the steps
    for local_step in tqdm(range(num_steps), desc="Train iteration", initial=global_step, total=args.max_steps,disable=not is_main_process()):
        input_ids, attention_mask, target_ids, mask_p = input_ids_, attention_mask_, target_ids_, mask_p_

        # Note: Word counting happens in surprisal computation phases for accuracy and speed

        print(f"global_step: {global_step} , rank:  {args.rank}")
        #print(local_step)

        # forward pass, do a more detailed check of the model every 100 steps
        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            with ModelLogger(enable=global_step % 100 == 0, module=model):
                print(f" dataset type: {args.dataset_type} rank: {args.rank}" )
                predictions,loss, accuracy, z_loss, num_tokens = model(input_ids, attention_mask, target_ids,return_all=False)
                print(f" the shape {predictions.shape}")
        # get the next batch
        if local_step < num_steps - 1:
            input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(train_dataloader, args.device, global_step)
        # calculate the weight for the loss (either token-weighted or not)
        if args.token_weighted_loss:
            total_tokens = torch.tensor(num_tokens, device=args.device, dtype=torch.long)
            torch.distributed.all_reduce(total_tokens, torch.distributed.ReduceOp.SUM)
            weight = args.world_size * num_tokens / total_tokens / args.accumulate_steps
        else:
            weight = 1.0 / args.accumulate_steps

        print(f"backward pass , rank:  {args.rank}")

        # backward pass through both losses
        ((loss + args.z_loss_weight * z_loss) * weight).backward()
        print(f"backward done rank {args.rank}")
        # add the tracked metrics (for gradient accumulation)
        total_loss += loss.detach() * weight
        print(f"detach loss rank: {args.rank}")
        total_accuracy += accuracy * weight
        total_z_loss += z_loss * weight
        total_mask_p += mask_p * weight

        # gradient accumulation -- if we have accumulated enough gradients, we can perform the optimizer step; otherwise, we just continue and backpropagate through the next batch
        print(f"accumulate steps {args.accumulate_steps}, rank: {args.rank}")
        if (local_step + 1) % args.accumulate_steps != 0:
            continue

        # clip the gradients
        total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) * weight
        print(f"optimizer step, rank: {args.rank}")
        # optimizer step
        optimizer.step()
        print(f"scheduler step, rank:  {args.rank}")
        
        word_progress = min(total_word_count / target_word_count, 1.0)
        expected_scheduler_step = int(word_progress * args.max_steps)
        if not hasattr(args, 'last_scheduler_step'):
            args.last_scheduler_step = 0
        if expected_scheduler_step > args.last_scheduler_step:
            for _ in range(expected_scheduler_step - args.last_scheduler_step):
                scheduler.step()
            args.last_scheduler_step = expected_scheduler_step
        
        
        with torch.no_grad():

            # EMA update
            for param_q, param_k in zip(model.module.parameters(), ema_model.parameters()):
                param_k.data.mul_(args.ema_decay).add_((1.0 - args.ema_decay) * param_q.detach().data)
            #print("ema")
            print(f" {args.dataset_type}, rank: {args.rank}")
            # be careful here, not all GPUs work with the same training objective
            if args.dataset_type == "masked":
                total_mlm_loss = total_loss / (args.hybrid_numerator / args.hybrid_denominator)
                total_clm_loss = torch.zeros_like(total_mlm_loss)
                total_mask_p = total_mask_p / (args.hybrid_numerator / args.hybrid_denominator)
            else:
                print(f" ln 306 doing else, rank : {args.rank}")
                total_clm_loss = total_loss / (1 - args.hybrid_numerator / args.hybrid_denominator)
                total_mlm_loss = torch.zeros_like(total_clm_loss)
                total_mask_p = torch.zeros_like(total_mask_p)
            print(f"accumulating metrics, rank:  {args.rank}")
            # accumulate the metrics across GPUs
            metrics = torch.stack([total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_clm_loss])
            torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
        
            total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_clm_loss = metrics.tolist()
            print(f"all reduced done, rank: {args.rank}")
        
        # log the metrics
        if is_main_process():
            wandb.log(
                {
                    "word_count": total_word_count,
                    "epoch": epoch,
                    "train/loss": total_loss,
                    "train/z_loss": total_z_loss,
                    "train/perplexity": math.exp(total_loss),
                    "train/accuracy": total_accuracy * 100.0,
                    "train/mlm_loss": total_mlm_loss,
                    "train/clm_loss": total_clm_loss,
                    "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    "stats/grad_norm": total_grad_norm,
                    "stats/seq_length": train_dataset.seq_length,
                    "stats/global_batch_size": args.current_global_batch_size,
                    "stats/local_batch_size": args.current_local_batch_size,
                    "stats/accumulate_steps": args.accumulate_steps,
                    "stats/mask_p": total_mask_p,
                    "global_step": global_step
                },
                commit=True
            )
        
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0

        # checkpoint the model and the full training state

        if total_word_count in save_milestones:
            save(model, ema_model, optimizer, scheduler, global_step, epoch, total_word_count, args)
        print(f"saving model, rank: {args.rank}")
        
        # validate the model
        if (global_step + 1) % args.validate_every == 0:
            validation_epoch(model, valid_dataloader, epoch, total_word_count, args)
            model.train()
        print(f"validating model, rank: {args.rank}")
        
        # Note: Logging now handled above with custom word_count metric

        global_step += 1
        print(f"global step end of function: {global_step}, rank: {args.rank}")
        if total_word_count >= target_word_count:
            return global_step, total_word_count
        # Exiting the training due to hitting max steps
        #if global_step >= args.max_steps:
            #return global_step

    return global_step, total_word_count


@torch.no_grad()
def validation_epoch(model, valid_dataloader, epoch, total_word_count, args, commit=False):
    model = model.eval()
    print(f"start validation function rank: {args.rank}")
    losses, accuracies = [], []
    valid_dataloader = iter(valid_dataloader)
    input_ids, attention_mask, target_ids, _ = get_batch(valid_dataloader, args.device, 0)
    print("got the valid data")
    for local_step in tqdm(range(args.validation_steps), desc="Valid iteration", disable=not is_main_process()):
        print("going to model call")
        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            predictions,loss, accuracy, _, num_tokens = model(input_ids, attention_mask, target_ids)
            print("got the loss and accuracy")
        if local_step < args.validation_steps - 1:
            input_ids, attention_mask, target_ids, _ = get_batch(valid_dataloader, args.device, 0)
            print("got the next batch in validation")
        total_tokens = torch.tensor(num_tokens, device=args.device, dtype=torch.long)
        print("total_tokens")
        torch.distributed.all_reduce(total_tokens, torch.distributed.ReduceOp.SUM)
        weight = args.world_size * num_tokens / total_tokens
        print("weight")
        metrics = torch.stack([loss * weight, accuracy * weight])
        print("metrics")
        torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
        loss, accuracy = metrics.tolist()
        print("append loss")
        losses.append(loss)
        print("append accuracies")
        accuracies.append(accuracy)

    if is_main_process():
        wandb.log(
            {
                "word_count": total_word_count,
                "epoch": epoch,
                "validation/loss": mean(losses),
                "validation/accuracy": mean(accuracies) * 100.0,
                "validation/perplexity": math.exp(mean(losses))
            },
            commit=commit
        )


def save(model, ema_model, optimizer, scheduler, global_step, epoch, word_count, args):
    if not is_main_process():
        return

    def get_suffix(word_count):
        if word_count <= 10_000_000 and word_count % 1_000_000 == 0:
            return f"_at_{word_count // 1_000_000}M_words"
        elif word_count > 10_000_000 and word_count % 10_000_000 == 0 and word_count <= 100_000_000:
            return f"_at_{word_count // 1_000_000}M_words"
        elif 99_500_000 <= word_count < 100_000_000:
            return "_at_100M_words"
        else:
            return None

    suffix = get_suffix(word_count)
    if suffix is None:
        return

    checkpoint_dir = os.path.splitext(args.output_path)[0] + "_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.output_path))[0]
    checkpoint_base = os.path.join(checkpoint_dir, f"{base_name}{suffix}")

    print(f"[Rank 0] Saving checkpoint at {word_count} words to: {checkpoint_base}*.bin")

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), checkpoint_base + ".bin")
    torch.save(ema_model.state_dict(), checkpoint_base + "_ema.bin")
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch + 1,
        },
        checkpoint_base + "_state_dict.bin"
    )



def load_datasets(args,train_path, tokenizer, epoch, global_step, total_word_count, target_word_count, train_dataloader, valid_dataloader):
    print(f" loaded dataset begin, rank {args.rank}")
    train_seed = args.seed + get_rank() + epoch * get_world_size()

    #if (global_step + 1) / args.max_steps >= 0.9:
    #    seq_length = args.seq_length * 4
    #    global_batch_size = args.global_batch_size // 4
    #if (global_step + 1) / args.max_steps >= 0.7:
    #    seq_length = args.seq_length * 2
    #    global_batch_size = args.global_batch_size // 2
    #else:
    seq_length = args.seq_length
    global_batch_size = args.global_batch_size

    if train_dataloader is None or train_dataloader.dataset.seq_length != seq_length:
        if args.dataset_type == "masked":
            rank = args.rank
            world_size = args.world_size * args.hybrid_numerator // args.hybrid_denominator
            train_data = MaskedDataset(train_path, tokenizer, args, seq_length, rank, world_size)
            #print(len(train_data))
        else:
            rank = args.rank - args.world_size * args.hybrid_numerator // args.hybrid_denominator
            world_size = args.world_size * (args.hybrid_denominator - args.hybrid_numerator) // args.hybrid_denominator
            train_data = CausalDataset(train_path, tokenizer, args, seq_length, rank, world_size)
            #print(f"causal {len(train_data)}")
        if is_main_process():
            train_data.show_random_item(tokenizer)
    else:
        train_data = train_dataloader.dataset
        #print(f"else {len(train_data)}")

    # linear batch size scaling
    word_progress = min(total_word_count / target_word_count, 1.0)
    args.current_global_batch_size = int(global_batch_size / args.batch_reduction * (1 - word_progress) + global_batch_size * word_progress + 0.5)
    total_local_batch_size = int(args.current_global_batch_size / args.world_size + 0.5)
    args.accumulate_steps = int(math.ceil(total_local_batch_size / args.local_batch_size))
    args.current_local_batch_size = total_local_batch_size // args.accumulate_steps
    print(f" local : {args.current_local_batch_size} total : {total_local_batch_size}")
    min_length = torch.tensor(len(train_data) // total_local_batch_size, dtype=torch.long, device=args.device)
    
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.current_local_batch_size,
        num_workers=0,  # non-zero num_workers causes segmenation fault
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=True,
        pin_memory=True,
    )

    if valid_dataloader is None:
        valid_data = ValidationDataset(args.valid_path, tokenizer, args)

        valid_dataloader = DataLoader(
            valid_data,
            shuffle=False,
            batch_size=args.local_batch_size,
            num_workers=0,  # non-zero num_workers causes segmenation fault
            generator=torch.Generator().manual_seed(42),
            drop_last=True,
            pin_memory=True,
        )

    return train_dataloader, valid_dataloader


def reload_aclm_dataset_and_rebuild_tss(model, dataset_cycle, tokenizer, args, INITIAL_SAMPLE, SAMPLE_PER_ITER, device):
    """
    Reload the original dataset and rebuild TSS when current dataset is exhausted
    """
    print(f"DATASET CYCLE {dataset_cycle}: Reloading dataset and rebuilding TSS")
    
    
    # Reload the original training dataset
    aclm_data_df = pd.read_csv(args.csv_path)
    print(f"Reloaded original dataset with {len(aclm_data_df)} samples")
    
    # Create validation dataset for TSS rebuilding
    surprisal_data = ValidationDataset(aclm_data_df, tokenizer, args)
    surp_dataloader = DataLoader(
        surprisal_data,
        shuffle=False,
        batch_size=1,
        num_workers=0,
        generator=torch.Generator().manual_seed(42),
        drop_last=False,
        pin_memory=True,
    )
    
    # Only main process rebuilds TSS to avoid conflicts
    #if is_main_process():
        # Rebuild TSS using current model state
    print("Rebuilding TSS using current model state...")
    tss_gpt = GPTBertSurprisalSpace(0, surp_dataloader, model)
    tss_gpt.fit()
        
    # Save the rebuilt TSS
    tss_filename = f"surprisals_gpt_cycle_{dataset_cycle}.pkl"
    pickle.dump(tss_gpt, open(tss_filename, "wb"))
    print(f"TSS rebuilt and saved as {tss_filename}")
    print(f"TSS size: {len(tss_gpt.surprisalvecs)}")
    
    # Synchronize all GPUs - wait for main process to finish rebuilding TSS
    torch.distributed.barrier()
    print(f"Rank {args.rank}: TSS rebuild synchronization complete")
    
    # All processes load the rebuilt TSS
    #if not is_main_process():
    #    tss_filename = f"surprisals_gpt_cycle_{dataset_cycle}.pkl"
    #    with open(tss_filename, 'rb') as f:
    #        tss_gpt = pickle.load(f)
    #    print(f"Rank {args.rank}: Loaded TSS from {tss_filename}")
    #    print(f"Rank {args.rank}: TSS size: {len(tss_gpt.surprisalvecs)}")
    
    # Find the most confused sample instead of random sampling
    print("Computing surprisals for entire dataset to find most confused sample...")

    surp_dataloader = DataLoader(
        surprisal_data,
        shuffle=False,
        batch_size=64,
        num_workers=0,
        generator=torch.Generator().manual_seed(42),
        drop_last=False,
        pin_memory=True,
    )
    
    # Compute surprisals for the entire reloaded dataset
    surprisal_by_group = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(surp_dataloader, desc="Computing surprisals for dataset reload"):
            input_ids, target_ids, attention_mask, mask_p = [t.pin_memory().to(device, non_blocking=True) for t in batch]
            input_ids, target_ids = input_ids.t(), target_ids.t()
            
            with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                prediction = model(input_ids, attention_mask, target_ids, return_all=True)
                target_ids_flat = target_ids.flatten()
                
            loss = F.cross_entropy(prediction, target_ids_flat, reduction='none')
            seq_length, batch_size = target_ids.size()
            loss = loss.view(seq_length, batch_size)
            surprisal = loss.transpose(0, 1)
            mean_surprisal = surprisal.mean(dim=1)
            surprisals = mean_surprisal.tolist()
            surprisal_by_group.extend(surprisals)
    
    # Find the most confused sample
    surprisal_array = np.array(surprisal_by_group)
    most_confused_idx = surprisal_array.argmax()
    most_confused_index = aclm_data_df.index[most_confused_idx]
    
    print(f"Most confused sample index: {most_confused_index}")
    
    # Find similar samples using TSS
    _, initial_indices, _ = tss_gpt.find_index(most_confused_index, k=SAMPLE_PER_ITER)
    
    sampled_train_data_df = aclm_data_df.loc[initial_indices, :]
    remaining_df = aclm_data_df.drop(initial_indices, axis=0)
    
    # Remove initial samples from TSS
    tss_gpt.remove_from_space(initial_indices)
    
    print(f"Reset complete:")
    print(f"  - Most confused index: {most_confused_index}")
    print(f"  - Initial sample size: {len(sampled_train_data_df)}")
    print(f"  - Remaining samples: {len(remaining_df)}")
    print(f"  - TSS size after removal: {len(tss_gpt.nnfinder.data)}")
    
    return aclm_data_df, sampled_train_data_df, remaining_df, tss_gpt


if __name__ == "__main__":
    args = parse_arguments()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    setup_training(args, tokenizer)
    model, ema_model, optimizer, scheduler, global_step, start_epoch = prepare_model_and_optimizer(args)
    #print("done preparing")
    total_word_count = 0
    save_milestones = list(range(1_000_000, 10_000_001, 1_000_000))  # every 1M words up to 10M
    save_milestones += list(range(20_000_000, 100_000_001, 10_000_000))  # every 10M words from 20M to 100M
    last_saved_milestone = 0
    if total_word_count <= 99_960_000:
        train_dataloader, valid_dataloader = None, None    
        # ACLM hyper-parameters
        INITIAL_SAMPLE = args.aclm_init_size * args.local_batch_size
        TSS_SAMPLE_SIZE = args.aclm_tss_sample_size * args.local_batch_size
        SAMPLE_PER_ITER = args.aclm_sample_per_iter * args.local_batch_size

        # Load ACLM data
        aclm_data_df = pd.read_csv(args.csv_path)
        print(aclm_data_df.shape[0])
        max_iteration = float(aclm_data_df.shape[0] - INITIAL_SAMPLE) / SAMPLE_PER_ITER

        print("TSS_SAMPLE_SIZE: ", TSS_SAMPLE_SIZE)
        print("SAMPLE_PER_ITER: ", SAMPLE_PER_ITER)
        print("max_iteration: ", max_iteration)
        print("Initial sample", INITIAL_SAMPLE)
        
        # Add cycle tracking variables
        dataset_cycle = 0
        min_remaining_samples = SAMPLE_PER_ITER * 5  # Minimum samples needed to continue
        target_word_count = 99_960_000
        
        #with model.join(divide_by_initial_world_size=False):
        pool = aclm_data_df['index'].to_numpy()
        print(pool.shape)
        #tss = pickle.load(open("../surprisals_8.pkl", "rb"))
        with open('/mimer/NOBACKUP/groups/naiss2024-6-297/.to-arrhenius-disk/babyLM2026/gpt-bert/pretraining/surprisals_8_4k.pkl', 'rb') as f:
            tss = pickle.load(f)
    
    
        initial_indices = np.random.choice(len(aclm_data_df), INITIAL_SAMPLE, replace=False)
    
        with open('initial.txt', 'w') as file:
            for item in initial_indices:
                file.write(f"{item}\n")
        pool = np.delete(pool, initial_indices)
        tss.remove_from_space(initial_indices)
        print(f"tss size: {len(tss.nnfinder.data)}")
        sampled_train_data_df = aclm_data_df.loc[initial_indices,:]
        remaining_df = aclm_data_df.drop(initial_indices, axis=0)  
        remaining_df.to_csv('file1.csv', index=False)
        print(len(remaining_df))
        split = 0
        convergence_criterion_not_met = True
        tss_gpt = None  
        for epoch in count(start=start_epoch):
            print(f" epoch: {epoch} | dataset_cycle: {dataset_cycle} | total_word_count: {total_word_count}")
            print("## ************ start ACLM epoch: {} (cycle {}) **********".format(epoch, dataset_cycle))
        
            train_dataloader, valid_dataloader = load_datasets(args,sampled_train_data_df, tokenizer, epoch, global_step, total_word_count, target_word_count, train_dataloader, valid_dataloader)
            #print("loaded successfully")
            global_step, total_word_count = training_epoch(model, ema_model, train_dataloader, valid_dataloader, optimizer, scheduler, global_step, epoch, total_word_count, target_word_count, args)
            print(f"finished training epoch global step taken rank: {args.rank}")
            print("#### **** start ACLM surprisal computing ******")
            if epoch == 0:
            
                surprisal_by_group = []
            
                surprisal_data = ValidationDataset(sampled_train_data_df, tokenizer, args)
                sampled_dataloader = DataLoader(
                    surprisal_data,
                    shuffle=False,
                    batch_size=64,
                    num_workers=0,  # non-zero num_workers causes segmenation fault
                    generator=torch.Generator().manual_seed(42),
                    drop_last=False,
                    pin_memory=True,
                )
                sampled_seed = args.seed + get_rank() + epoch * get_world_size()
                
                model = model.eval()
                
                sampled_dataloader = iter(sampled_dataloader)

                with torch.no_grad():
                    for local_step in tqdm(range(len(sampled_dataloader)),desc="ACLM num steps"):
                        input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(sampled_dataloader, args.device, 0)
                        num_words = sum(count_words_from_tokens(seq) for seq in input_ids_.t())
                        num_words_tensor = torch.tensor(num_words, dtype=torch.long, device=args.device)
                        torch.distributed.all_reduce(num_words_tensor, op=torch.distributed.ReduceOp.SUM)
                        total_word_count += num_words_tensor.item()
                        print(f" {args.rank} {total_word_count}")
                        passed_milestones = sorted([m for m in save_milestones if last_saved_milestone < m <= total_word_count])
                        for milestone in passed_milestones:
                            save(model, ema_model, optimizer, scheduler, global_step, epoch, milestone, args)
                            last_saved_milestone = milestone
                        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                            prediction = model(input_ids_, attention_mask_, target_ids_,return_all=True)
                            #print(prediction.shape)
                            target_ids= target_ids_.flatten()
                        print(f"Prediction shape: {prediction.shape}, Target shape: {target_ids.shape}")      
                        loss = F.cross_entropy(
                            prediction, target_ids, reduction='none')

                        print(f" loss {loss.size()}")

                        seq_length, batch_size = target_ids_.size()
                        print(batch_size,seq_length)

                    
                        loss = loss.view(seq_length, batch_size)
                        surprisal = loss.transpose(0, 1)
                        print(f" surprisal after reshaping: {surprisal.shape}")
                    
                        
                        mean_surprisal = surprisal.mean(dim=1) 
                    
                        print(f"mean_surprisal: {mean_surprisal.shape}")

                        surprisals = mean_surprisal.tolist()
                        surprisal_by_group += surprisals
                        print("len(surprisal_by_group)",len(surprisal_by_group))
                        print("len(sampled_dataloader)",len(sampled_dataloader))
                        print("#### **** end ACLM surprisal computing ******")
                    
                    surprisal_array = np.array(surprisal_by_group)
                    print('surprisal_array.shape', surprisal_array.shape)
            
                    max_surprisal_idx = surprisal_array.argmax()
                    most_confused_index = initial_indices[max_surprisal_idx]
            
                    print('most_confused_index', most_confused_index)
            
                    print(f"size of tss: {len(tss.nnfinder.data)}")
                    _, indices, _ = tss.find_index(most_confused_index, k=SAMPLE_PER_ITER)
                
                    # Take things out of the space.
                    tss.remove_from_space(indices)
                    print('tss size', len(tss.nnfinder.data))
                    # indices = list(indices)
                    with open('indices.txt', 'w') as file:
                        for item in indices:
                            file.write(f"{item}\n")
                            #print(len(indices))
            
                    if any(item in initial_indices for item in indices):
                        print("At least one element of list1 is in list2.")
                    else:
                        print("No elements of list1 are in list2.")
                    if any(item in initial_indices for item in remaining_df['index']):
                        print(item)
                    else:
                        print("no element in remaining_df from initial")
                    missing = [idx for idx  in indices if idx not in remaining_df.index]
                    initial_in_missing = [idx for idx in missing if idx in initial_indices]
                    print(f" missing indices that are in initial: {initial_in_missing}")
                    sampled_train_data_df = remaining_df.loc[indices,:]
                    #explicit_indices = remaining_df.iloc[indices].index
                    remaining_df = remaining_df.drop(indices).reset_index(drop=True)
                    print(len(sampled_train_data_df))
                    print(len(remaining_df))
            elif epoch == 1:
                print(len(sampled_train_data_df))
                print("remaining df",len(remaining_df))
            
                surprisal_data = ValidationDataset(remaining_df, tokenizer, args)
                print(len(surprisal_data))
                surp_dataloader = DataLoader(
                    surprisal_data,
                    shuffle=False,
                    batch_size=1,
                    num_workers=0,  # non-zero num_workers causes segmenation fault
                    generator=torch.Generator().manual_seed(42),
                    drop_last=False,
                    pin_memory=True,
                )

                tss_gpt = GPTBertSurprisalSpace(0,surp_dataloader,model)
                #seq = remaining_df["Tokens"].tolist()
                tss_gpt.fit()
                pickle.dump(tss_gpt, open("surprisals_gpt.pkl", "wb"))
                print(f"Original size: {len(tss_gpt.surprisalvecs)}")
                
                # Synchronize all GPUs after TSS build
                torch.distributed.barrier()
                print(f"Rank {args.rank}: TSS build synchronization complete")

                

                surprisal_by_group = []
            
                surprisal_data = ValidationDataset(sampled_train_data_df, tokenizer, args)
                sampled_dataloader = DataLoader(
                    surprisal_data,
                    shuffle=False,
                    batch_size=64 ,
                    num_workers=0,  # non-zero num_workers causes segmenation fault
                    generator=torch.Generator().manual_seed(42),
                    drop_last=False,
                    pin_memory=True,
                )
                sampled_seed = args.seed + get_rank() + epoch * get_world_size()
                
                model = model.eval()
                
                sampled_dataloader = iter(sampled_dataloader)
                
                with torch.no_grad():
                    for local_step in tqdm(range(len(sampled_dataloader)),desc="ACLM num steps"):
                        input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(sampled_dataloader, args.device, 0)
                        num_words = sum(count_words_from_tokens(seq) for seq in input_ids_.t())
                        num_words_tensor = torch.tensor(num_words, dtype=torch.long, device=args.device)
                        torch.distributed.all_reduce(num_words_tensor, op=torch.distributed.ReduceOp.SUM)
                        total_word_count += num_words_tensor.item()
                        print(f" {args.rank} {total_word_count}")
                        passed_milestones = sorted([m for m in save_milestones if last_saved_milestone < m <= total_word_count])
                        for milestone in passed_milestones:
                            save(model, ema_model, optimizer, scheduler, global_step, epoch, milestone, args)
                            last_saved_milestone = milestone
                    
                        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                            prediction = model(input_ids_, attention_mask_, target_ids_,return_all=True)
                            #print(prediction.shape)
                        
                            target_ids= target_ids_.flatten()
                        
                            #print(f"Prediction shape: {prediction.shape}, Target shape: {target_ids.shape}")
                            loss = F.cross_entropy(
                                prediction, target_ids, reduction='none')

                            #print(f" loss {loss.size()}")
                            
                            seq_length, batch_size = target_ids_.size()
                            #print(batch_size,seq_length)
                            
                            loss = loss.view(seq_length, batch_size)
                            surprisal = loss.transpose(0, 1)
                            #print(f" surprisal after reshaping: {surprisal.shape}")

                            mean_surprisal = surprisal.mean(dim=1)
                            
                            #print(f"mean_surprisal: {mean_surprisal.shape}")

                            surprisals = mean_surprisal.tolist()
                            surprisal_by_group += surprisals
                        print("len(surprisal_by_group)",len(surprisal_by_group))
                        print("len(sampled_dataloader)",len(sampled_dataloader))
                        print("#### **** end ACLM surprisal computing ******")

                    surprisal_array = np.array(surprisal_by_group)
                    print('surprisal_array.shape', surprisal_array.shape)
                    max_surprisal_idx = surprisal_array.argmax()
                    print(max_surprisal_idx)
                    
                    most_confused_index = indices[max_surprisal_idx]
                    
                    print('most_confused_index', most_confused_index)
                    
                print(f"size of tss: {len(tss_gpt.nnfinder.data)}")
                _, indices, _ = tss_gpt.find_index(most_confused_index, k=SAMPLE_PER_ITER)
            
                # Take things out of the space.
                tss_gpt.remove_from_space(indices)
                print('tss size', len(tss_gpt.nnfinder.data))
                

                #print(len(remaining_df))
                if any(item in initial_indices for item in indices):
                    print("At least one element of list1 is in list2.")
                else:
                    print("No elements of list1 are in list2.")
                if any(item in initial_indices for item in remaining_df['index']):
                        print(item)
                else:
                    torch.distributed.barrier()
                print("no element in remaining_df from initial")
                missing = [idx for idx  in indices if idx not in remaining_df.index]
                initial_in_missing = [idx for idx in missing if idx in initial_indices]
                print(f" missing indices that are in initial: {initial_in_missing}")
                sampled_train_data_df = remaining_df.loc[indices,:]
                #explicit_indices = remaining_df.iloc[indices].index
                remaining_df = remaining_df.drop(indices)
                print(len(sampled_train_data_df))
                print(len(remaining_df))

            else:
                print(len(sampled_train_data_df))
                print(len(remaining_df))
                
                # Initialize tss_gpt if it doesn't exist (e.g., starting from epoch >= 2)
                if tss_gpt is None:
                    print("Initializing TSS for first time...")
                    surprisal_data = ValidationDataset(aclm_data_df, tokenizer, args)
                    surp_dataloader = DataLoader(
                        surprisal_data,
                        shuffle=False,
                        batch_size=1,
                        num_workers=0,
                        generator=torch.Generator().manual_seed(42),
                        drop_last=False,
                        pin_memory=True,
                    )
                    tss_gpt = GPTBertSurprisalSpace(0, surp_dataloader, model)
                    tss_gpt.fit()
                    pickle.dump(tss_gpt, open("surprisals_gpt_initial.pkl", "wb"))
                    print(f"TSS initialized with size: {len(tss_gpt.surprisalvecs)}")
                    
                    # Synchronize all GPUs after TSS build
                    torch.distributed.barrier()
                    print(f"Rank {args.rank}: TSS initialization synchronization complete")
                
                surprisal_by_group = []

                surprisal_data = ValidationDataset(sampled_train_data_df, tokenizer, args)
                sampled_dataloader = DataLoader(
                    surprisal_data,
                    shuffle=False,
                    batch_size=64 ,
                    num_workers=0,  # non-zero num_workers causes segmenation fault
                    generator=torch.Generator().manual_seed(42),
                    drop_last=False,
                    pin_memory=True,
                )
                sampled_seed = args.seed + get_rank() + epoch * get_world_size()
                
                model = model.eval()

                sampled_dataloader = iter(sampled_dataloader)

                with torch.no_grad():
                    for local_step in tqdm(range(len(sampled_dataloader)),desc="ACLM num steps"):
                        input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(sampled_dataloader, args.device, 0)
                        num_words = sum(count_words_from_tokens(seq) for seq in input_ids_.t())
                        num_words_tensor = torch.tensor(num_words, dtype=torch.long, device=args.device)
                        torch.distributed.all_reduce(num_words_tensor, op=torch.distributed.ReduceOp.SUM)
                        total_word_count += num_words_tensor.item()
                        print(f" {args.rank} {total_word_count}")
                        passed_milestones = sorted([m for m in save_milestones if last_saved_milestone < m <= total_word_count])
                        for milestone in passed_milestones:
                            save(model, ema_model, optimizer, scheduler, global_step, epoch, milestone, args)
                            last_saved_milestone = milestone
                        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                            prediction = model(input_ids_, attention_mask_, target_ids_,return_all=True)
                            print(prediction.shape)
                            target_ids= target_ids_.flatten()
                            
                        print(f"Prediction shape: {prediction.shape}, Target shape: {target_ids.shape}")
                        loss = F.cross_entropy(
                            prediction, target_ids, reduction='none')

                        print(f" loss {loss.size()}")

                        seq_length, batch_size = target_ids_.size()
                        print(batch_size,seq_length)

                        loss = loss.view(seq_length, batch_size)
                        surprisal = loss.transpose(0, 1)
                        print(f" surprisal after reshaping: {surprisal.shape}")

                    
                        mean_surprisal = surprisal.mean(dim=1)
                        
                        print(f"mean_surprisal: {mean_surprisal.shape}")

                        surprisals = mean_surprisal.tolist()
                        surprisal_by_group += surprisals
                        print("len(surprisal_by_group)",len(surprisal_by_group))
                        print("len(sampled_dataloader)",len(sampled_dataloader))
                        print("#### **** end ACLM surprisal computing ******")

                    surprisal_array = np.array(surprisal_by_group)
                    print('surprisal_array.shape', surprisal_array.shape)
                    
                    max_surprisal_idx = surprisal_array.argmax()
                    most_confused_index = indices[max_surprisal_idx]

                    print('most_confused_index', most_confused_index)

                print(f"size of tss: {len(tss_gpt.nnfinder.data)}")
                _, indices, _ = tss_gpt.find_index(most_confused_index, k=SAMPLE_PER_ITER)
                
                # Take things out of the space.
                #print(len(indices))
                #print(indices)
                tss_gpt.remove_from_space(indices)
                print('tss size', len(tss_gpt.nnfinder.data))
                # indices = list(indices)
                with open('indices.txt', 'w') as file:
                    for item in indices:
                        file.write(f"{item}\n")
                #print(len(indices))


                if any(item in initial_indices for item in indices):
                    print("At least one element of list1 is in list2.")
                else:
                    print("No elements of list1 are in list2.")
                if any(item in initial_indices for item in remaining_df['index']):
                    print(item)
                else:
                    print("no element in remaining_df from initial")
                missing = [idx for idx  in indices if idx not in remaining_df.index]
                initial_in_missing = [idx for idx in missing if idx in initial_indices]
                print(f" missing indices that are in initial: {initial_in_missing}")
                sampled_train_data_df = remaining_df.loc[indices,:]
                #explicit_indices = remaining_df.iloc[indices].index
                remaining_df = remaining_df.drop(indices)
                print(len(sampled_train_data_df))
                print(len(remaining_df))
        
        
            print("### ******** end ACLM split: {} ********".format(split))
            print(f"Progress: {total_word_count:,}/{target_word_count:,} words ({100*total_word_count/target_word_count:.1f}%)")
            print(f"Remaining samples in current cycle: {len(remaining_df):,}")
            split += 1
            epoch += 1

            validation_epoch(model, valid_dataloader, epoch, total_word_count, args, commit=True)
            if total_word_count in save_milestones:
                save(model, ema_model, optimizer, scheduler, global_step, epoch,total_word_count, args)
            
            # Check if we've reached the target word count
            if total_word_count >= target_word_count:
                print(f"Reached target word count: {total_word_count}")
                convergence_criterion_not_met = False
                break

            # Check if we need to reload the dataset
            if len(remaining_df) < min_remaining_samples:
                dataset_cycle += 1
                print(f"Dataset exhausted (remaining: {len(remaining_df)}). Starting cycle {dataset_cycle}")
                
                # Reload dataset and rebuild TSS
                aclm_data_df, sampled_train_data_df, remaining_df, tss_gpt = reload_aclm_dataset_and_rebuild_tss(
                    model, dataset_cycle, tokenizer, args, INITIAL_SAMPLE, SAMPLE_PER_ITER, args.device
                )
                
                # Reset split counter for new cycle
                split = 0
                print(f"Dataset cycle {dataset_cycle} initialized. Continuing training...")
                continue
    else:
        save(model, ema_model, optimizer, scheduler, global_step, epoch,total_word_count, args)
        validation_epoch(model, valid_dataloader, epoch, total_word_count, args, commit=True)
