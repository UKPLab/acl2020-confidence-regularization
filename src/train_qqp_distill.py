"""
Script to train BERT on QQP

Modified from the old "run_classifier" script from
https://github.com/huggingface/pytorch-transformer

Generally I tried not to change much, but I did add variable length sequence encoding
and parallel pre-processing for the sake of performance

Fair warning, I probably broke some of the multi-gpu stuff, I have only tested the single GPU version
"""

import argparse
import json
import logging
import os
import random
from collections import namedtuple
from os.path import join, exists
from typing import List, Dict, Iterable

# temporary hack for the pythonroot issue
import sys

import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, Sampler
from tqdm import trange, tqdm

from sklearn.metrics import f1_score

import config
import utils

import clf_distill_loss_functions
from bert_distill import BertDistill
from clf_distill_loss_functions import *

from predictions_analysis import visualize_predictions
from utils import Processor, process_par

LABEL_MAP = {"not_duplicate": 0, "duplicate": 1}
REV_LABEL_MAP = ["not_duplicate", "duplicate"]

TextPairExample = namedtuple("TextPairExample", ["id", "question1", "question2", "label"])


def load_qqp(dataset="qqp", split="train", sample=None, paws_num=0) -> List[TextPairExample]:
    if split in ["train", "dev", "test"]:
        if dataset == "qqp":
            filename = join(config.QQP_SOURCE, "{}.tsv".format(split))
        elif dataset == "qqp_paws":
            tsvname = "{}_qqp_paws_{}.tsv".format(split, paws_num) if split == "train" else "test_paws.tsv"
            filename = join(config.QQP_ADD_PAWS, tsvname)
        else:
            raise Exception("invalid dataset name")
    else:
        raise Exception("invalid qqp split")

    logging.info("Loading QQP from {}".format(filename))
    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample, replace=False)

    out = []

    for line in lines:
        line = line.split("\t")
        if len(line) != 6:
            continue
        out.append(
            TextPairExample(line[0], line[3], line[4], int(line[-1])))
    
    return out

def load_paws(is_train=False) -> List[TextPairExample]:
    if is_train:
        filename = join(config.QQP_PAWS_SOURCE, "qqp_train.tsv")
    else:
        filename = join(config.QQP_PAWS_SOURCE, "qqp_dev_and_test.tsv")

    logging.info("Loading PAWS... is_train: {}".format(str(is_train)))
    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    out = []
    for line in lines:
        line = line.split("\t")
        out.append(
            TextPairExample(line[0], line[1], line[2], int(line[-1])))
    return out

def load_teacher_probs(dataset="qqp", paws_num=0):
    file_path = None
    if dataset == "qqp":
        file_path = config.QQP_TEACHER_SOURCE
    elif dataset == "qqp_paws":
        file_path = config.QQP_PAWS_TEACHER_SOURCES["hans_{}".format(paws_num)]

    with open(file_path, "r") as teacher_file:
        all_lines = teacher_file.read()
        all_json = json.loads(all_lines)

    return all_json


def load_bias(bias_name, dataset="qqp", paws_num=0) -> Dict[str, np.ndarray]:
    """Load dictionary of example_id->bias where bias is a length 2 array
    of log-probabilities"""
    if bias_name == "qqp_hans_json":
        if dataset == "qqp":
            file_path = config.BIAS_SOURCES[bias_name]
        elif dataset == "qqp_paws":
            # TODO: possible rollback here
            file_path = config.QQP_PAWS_BIAS_SOURCES["hans_{}".format(paws_num)]
            # file_path = config.QQP_PAWS_BIAS_SOURCES["shallow_{}".format(paws_num)]

        with open(file_path, "r") as hypo_file:
            all_lines = hypo_file.read()
            bias = json.loads(all_lines)
            for k, v in bias.items():
                bias[k] = np.array(v)
        return bias
    else:
        raise Exception("invalid bias name")


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, input_ids, segment_ids, label_id, bias):
        self.example_id = example_id
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.bias = bias


class ExampleConverter(Processor):
    def __init__(self, max_seq_length, tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def process(self, data: Iterable):
        features = []
        tokenizer = self.tokenizer
        max_seq_length = self.max_seq_length

        for example in data:
            tokens_a = tokenizer.tokenize(example.question1)

            tokens_b = None
            if example.question2:
                tokens_b = tokenizer.tokenize(example.question2)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            features.append(
                InputFeatures(
                    example_id=example.id,
                    input_ids=np.array(input_ids),
                    segment_ids=np.array(segment_ids),
                    label_id=example.label,
                    bias=None
                ))
        return features


class InputFeatureDataset(Dataset):

    def __init__(self, examples: List[InputFeatures]):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def collate_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = np.zeros((sz, max_seq_len), np.int64)
    segment_ids = np.zeros((sz, max_seq_len), np.int64)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
    for i, ex in enumerate(batch):
        input_ids[i, :len(ex.input_ids)] = ex.input_ids
        segment_ids[i, :len(ex.segment_ids)] = ex.segment_ids
        mask[i, :len(ex.input_ids)] = 1

    input_ids = torch.as_tensor(input_ids)
    segment_ids = torch.as_tensor(segment_ids)
    label_ids = torch.as_tensor(np.array([x.label_id for x in batch], np.int64))

    # include example ids for test submission
    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = torch.zeros(len(batch)).long()

    if batch[0].bias is None:
        return example_ids, input_ids, mask, segment_ids, label_ids

    teacher_probs = torch.tensor([x.teacher_probs for x in batch])
    bias = torch.tensor([x.bias for x in batch])

    return example_ids, input_ids, mask, segment_ids, label_ids, bias, teacher_probs


class SortedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, seed):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.seed = seed
        if batch_size == 1:
            raise NotImplementedError()
        self._epoch = 0

    def __iter__(self):
        rng = np.random.RandomState(self._epoch + 601767 + self.seed)
        n_batches = len(self)
        batch_lens = np.full(n_batches, self.batch_size, np.int32)

        # Randomly select batches to reduce by size 1
        extra = n_batches * self.batch_size - len(self.data_source)
        batch_lens[rng.choice(len(batch_lens), extra, False)] -= 1

        batch_ends = np.cumsum(batch_lens)
        batch_starts = np.pad(batch_ends[:-1], [1, 0], "constant")

        if batch_ends[-1] != len(self.data_source):
            print(batch_ends)
            raise RuntimeError()

        bounds = np.stack([batch_starts, batch_ends], 1)
        rng.shuffle(bounds)

        for s, e in bounds:
            yield np.arange(s, e)

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def build_train_dataloader(data: List[InputFeatures], batch_size, seed, sorted):
    if sorted:
        data.sort(key=lambda x: len(x.input_ids))
        ds = InputFeatureDataset(data)
        sampler = SortedBatchSampler(ds, batch_size, seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_input_features)
    else:
        ds = InputFeatureDataset(data)
        return DataLoader(ds, batch_size=batch_size, sampler=RandomSampler(ds),
                          collate_fn=collate_input_features)


def build_eval_dataloader(data: List[InputFeatures], batch_size):
    ds = InputFeatureDataset(data)
    return DataLoader(ds, batch_size=batch_size, sampler=SequentialSampler(ds),
                      collate_fn=collate_input_features)


def convert_examples_to_features(
        examples: List[TextPairExample], max_seq_length, tokenizer, n_process=1):
    converter = ExampleConverter(max_seq_length, tokenizer)
    return process_par(examples, converter, n_process, chunk_size=2000, desc="featurize")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    f1_non = f1_score(y_true=labels, y_pred=preds, pos_label=0)
    return {
        "acc": acc,
        "f1": f1,
        "f1_non": f1_non
    }


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Seed for randomized elements in the training")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    ## Our arguements
    parser.add_argument("--mode", choices=["none", "distill", "smoothed_distill",
                                           "theta_smoothed_distill", "reweight_baseline",
                                           "bias_product_baseline", "learned_mixin_baseline"],
                        default="learned_mixin", help="Kind of debiasing method to use")
    parser.add_argument("--penalty", type=float, default=0.03,
                        help="Penalty weight for the learn_mixin model")
    parser.add_argument("--n_processes", type=int, default=4,
                        help="Processes to use for pre-processing")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sorted", action="store_true",
                        help='Sort the data so most batches have the same input length,'
                             ' makes things about 2x faster. Our experiments did not actually'
                             ' use this in the end (not sure if it makes a difference) so '
                             'its off by default.')
    parser.add_argument("--which_bias", choices=["qqp_hans_json"], default=None)
    parser.add_argument("--theta", type=float, default=0.1, help="for theta smoothed distillation loss")

    parser.add_argument("--qqp_dataset", type=str, default="qqp")
    parser.add_argument("--paws_num", type=int, default=0)

    args = parser.parse_args()

    utils.add_stdout_logger()

    if args.mode == "none":
        loss_fn = clf_distill_loss_functions.Plain()
    elif args.mode == "distill":
        loss_fn = clf_distill_loss_functions.DistillLoss()
    elif args.mode == "smoothed_distill":
        loss_fn = clf_distill_loss_functions.SmoothedDistillLoss()
    elif args.mode == "theta_smoothed_distill":
        loss_fn = clf_distill_loss_functions.ThetaSmoothedDistillLoss(args.theta)
    elif args.mode == "reweight_baseline":
        loss_fn = clf_distill_loss_functions.ReweightBaseline()
    elif args.mode == "bias_product_baseline":
        loss_fn = clf_distill_loss_functions.BiasProductBaseline()
    elif args.mode == "learned_mixin_baseline":
        loss_fn = clf_distill_loss_functions.LearnedMixinBaseline(args.penalty)
    else:
        raise RuntimeError("invalid mode")

    output_dir = args.output_dir

    if args.do_train:
        if exists(output_dir):
            if len(os.listdir(output_dir)) > 0:
                logging.warning("Output dir exists and is non-empty")
        else:
            os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(output_dir) and os.listdir(output_dir) and args.do_train:
        logging.warning(
            "Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Its way ot easy to forget if this is being set by a command line flag
    if "-uncased" in args.bert_model:
        do_lower_case = True
    elif "-cased" in args.bert_model:
        do_lower_case = False
    else:
        raise NotImplementedError(args.bert_model)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=do_lower_case)

    num_train_optimization_steps = None
    train_examples = None
    if args.do_train:
        train_examples = load_qqp(dataset=args.qqp_dataset, split="train", paws_num=args.paws_num,
                                  sample=2000 if args.debug else None)
        num_train_optimization_steps = int(
            len(
                train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(
        str(PYTORCH_PRETRAINED_BERT_CACHE),
        'distributed_{}'.format(args.local_rank))

    model = BertDistill.from_pretrained(
        args.bert_model, cache_dir=cache_dir, num_labels=2, loss_fn=loss_fn)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        train_features: List[InputFeatures] = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer, args.n_processes)

        bias_map = None

        if args.which_bias is None:
            raise Exception("bias source must be specified")

        if args.which_bias == "mix":
            hypo_bias_map = load_bias("hypo")
            hans_bias_map = load_bias("hans")
            bias_map = {}
            def compute_entropy(probs, base=2):
                return -(probs * (np.log(probs) / np.log(base))).sum()
            for key in hypo_bias_map.keys():
                hypo_ent = compute_entropy(np.exp(hypo_bias_map[key]))
                hans_ent = compute_entropy(np.exp(hans_bias_map[key]))
                if hypo_ent < hans_ent:
                    bias_map[key] = hypo_bias_map[key]
                else:
                    bias_map[key] = hans_bias_map[key]
        else:
            bias_map = load_bias(args.which_bias, args.qqp_dataset, args.paws_num)

        logging.info("**** filtering down the training example ****")
        logging.info("original len: {}".format(str(len(train_features))))
        train_features = [x for x in train_features if x.example_id in bias_map]
        logging.info("filtered len: {}".format(str(len(train_features))))

        if args.mode != "none":
            for fe in train_features:
                fe.bias = bias_map[fe.example_id].astype(np.float32)
            teacher_probs_map = load_teacher_probs(args.qqp_dataset, args.paws_num)
            for fe in train_features:
                fe.teacher_probs = np.array(teacher_probs_map[fe.example_id]).astype(
                    np.float32)
        else:
            bias_map = None


        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = build_train_dataloader(train_features, args.train_batch_size,
                                                  args.seed, args.sorted)

        model.train()
        loss_ema = 0
        total_steps = 0
        decay = 0.99

        for _ in trange(int(args.num_train_epochs), desc="Epoch", ncols=100):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            pbar = tqdm(train_dataloader, desc="loss", ncols=100)
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                if bias_map is not None:
                    example_ids, input_ids, input_mask, segment_ids, label_ids, bias, teacher_probs = batch
                else:
                    bias = None
                    teacher_probs = None
                    example_ids, input_ids, input_mask, segment_ids, label_ids = batch

                logits, loss = model(input_ids, segment_ids, input_mask, label_ids, bias, teacher_probs)

                total_steps += 1
                loss_ema = loss_ema * decay + loss.cpu().detach().numpy() * (1 - decay)
                descript = "loss=%.4f" % (loss_ema / (1 - decay ** total_steps))
                pbar.set_description(descript, refresh=False)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Record the args as well
        arg_dict = {}
        for arg in vars(args):
            arg_dict[arg] = getattr(args, arg)
        with open(join(output_dir, "args.json"), 'w') as out_fh:
            json.dump(arg_dict, out_fh)

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertDistill(config, num_labels=2, loss_fn=loss_fn)
        model.load_state_dict(torch.load(output_model_file))
    else:
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(output_config_file)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        model = BertDistill(config, num_labels=2, loss_fn=loss_fn)
        model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if not args.do_eval:
        return
    if not (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        return

    model.eval()

    if args.do_eval:
        eval_datasets = [("qqp_dev", load_qqp(split="dev"))]
        eval_datasets += [("paws_test", load_qqp(dataset=args.qqp_dataset, split="test"))]
        #eval_datasets += [("paws_test", load_paws(is_train=True))]
        # eval_datasets += [("qqp_train", load_qqp(dataset=args.qqp_dataset, split="train", paws_num=args.paws_num))]

    for ix, (name, eval_examples) in enumerate(eval_datasets):
        logging.info("***** Running evaluation on %s *****" % name)
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", args.eval_batch_size)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer)
        eval_features.sort(key=lambda x: len(x.input_ids))
        all_label_ids = np.array([x.label_id for x in eval_features])
        eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

        eval_loss = 0
        nb_eval_steps = 0
        probs = []
        test_subm_ids = []

        for example_ids, input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                               desc="Evaluating",
                                                                               ncols=100):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            probs.append(torch.nn.functional.softmax(logits, 1).detach().cpu().numpy())
            test_subm_ids.append(example_ids.cpu().numpy())

        probs = np.concatenate(probs, 0)
        test_subm_ids = np.concatenate(test_subm_ids, 0)
        eval_loss = eval_loss / nb_eval_steps

        preds = np.argmax(probs, axis=1)

        result = acc_and_f1(preds, all_label_ids)
        result["duplicate_acc"] = ((preds == all_label_ids).astype(np.int32)*all_label_ids).sum() / all_label_ids.sum()
        result["non-duplicate_acc"] = ((preds == all_label_ids).astype(np.int32)*(all_label_ids==0)).sum() / \
                                      (all_label_ids==0).sum()

        output_eval_file = os.path.join(output_dir, "eval_%s_results.txt" % name)
        output_all_eval_file = os.path.join(output_dir, "eval_all_results.txt")
        with open(output_eval_file, "w") as writer, open(output_all_eval_file, "a") as all_writer:
            logging.info("***** Eval results *****")
            all_writer.write("eval results on %s:\n" % name)
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
                all_writer.write("%s = %s\n" % (key, str(result[key])))

        output_answer_file = os.path.join(output_dir, "eval_%s_answers.json" % name)
        answers = {ex.example_id: [float(x) for x in p] for ex, p in
                   zip(eval_features, probs)}
        with open(output_answer_file, "w") as f:
            json.dump(answers, f)


if __name__ == "__main__":
    main()
