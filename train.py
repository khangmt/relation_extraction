from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from model import BERT_BiLSTM
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizerFast)
from transformers import AdamW, get_scheduler
logger = logging.getLogger(__name__)
from utils import EvalObject, get_label_map, get_Dataset
def evaluate(args, data, model, deviceType):
    score = EvalObject()
    model.eval()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)
    logger.info("***** Running eval *****")
    predict = np.array([])

    groundtruth_class = np.array([])

    for b_i, (input_ids, segment_ids, input_mask, class_ids) in enumerate(tqdm(dataloader, desc="Evaluating")):
        
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        label_ids = label_ids.to(args.device)

        with torch.no_grad():
            logits = model.predict(input_ids, segment_ids, input_mask)
            predictions = torch.argmax(logits, dim=-1)
            #convert to numpy to use sklearn metrics
            if deviceType == "cuda":
                np_predictions = predictions.cpu().detach().numpy()
                class_id = class_ids.cpu().detach().numpy()
            else:
                np_predictions = predictions.detach().numpy()
                class_id = class_ids.detach().numpy()
            predict = np.concatenate([predict, np_predictions], axis = 0)
        groundtruth_class = np.concatenate([groundtruth_class, class_id ], axis = 0)
    y_true = np.array()
    assert len(groundtruth_class) == predict.shape[0]
    from sklearn.metrics import precision_score, recall_score, f1_score
    score.precision = precision_score(groundtruth_class, predict)
    score.recall = recall_score(groundtruth_class, predict)
    score.f1 = f1_score(groundtruth_class, predict, average='weighted')

    return score
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--rel2id_file", default=None, type=str)
    parser.add_argument("--eval_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--input_dir", default=None, type=str)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--num_train_epochs", default=2, type=float)
    parser.add_argument("--warmup_proprotion", default=0.1, type=float)
    parser.add_argument("--use_weight", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--logging_steps", default=500, type=int)
    parser.add_argument("--clean", default=False, action="store_true", help="clean the output dir")

    parser.add_argument("--need_birnn", default=False, action="store_true")
    parser.add_argument("--rnn_dim", default=128, type=int)

    args = parser.parse_args()
    print(args)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        logger.info(f"device: {device} n_gpu: {n_gpu}")
    else :
        device = torch.device("cpu")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(),"output")
    #clean output_dir for retrain
    if args.clean and args.do_train:
        # logger.info("清理")
            if os.path.exists(args.output_dir):
                def del_file(path):
                    ls = os.listdir(path)
                    for i in ls:
                        c_path = os.path.join(path, i)
                        print(c_path)
                        if os.path.isdir(c_path):
                            del_file(c_path)
                            os.rmdir(c_path)
                        else:
                            os.remove(c_path)
            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
    
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if not os.path.exists(os.path.join(args.output_dir, "eval")):
        os.makedirs(os.path.join(args.output_dir, "eval"))
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, 
                    do_lower_case=args.do_lower_case)
    tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
    #get tag mapping and number of tags, number of classes
    parser.add_argument("--train_path", default=None, type=str)
    args.train_path = os.path.join(args.input_dir,args.train_file)
    rel2id, id2rel = get_label_map(args.rel2id_file)
    
    if args.do_train:
        config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        model = BERT_BiLSTM.from_pretrained(args.model_name_or_path, config=config, 
                need_birnn=args.need_birnn, rnn_dim=args.rnn_dim, num_rel = len(rel2id))
        model.to(device)
        
        if device.type =="cuda" and n_gpu > 1:
            model = torch.nn.DataParallel(model)
        writer = SummaryWriter(logdir=os.path.join(args.output_dir, "eval"), comment="Linear")
        
        #return pytorch Dataset
        train_data = get_Dataset(tokenizer, args.train_path, rel2id)
        print(train_data[0])
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        if args.do_eval:
            parser.add_argument("--eval_path", default=None, type=str)
            args.eval_path = os.path.join(args.input_dir,args.eval_file)
            eval_data = get_Dataset(tokenizer,args.eval_path)
        
        if args.max_steps > 0:
            num_training_steps = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        loss_function = nn.CrossEntropyLoss() 
        scheduler = get_scheduler(name="cosine", optimizer= optimizer, num_warmup_steps= args.warmup_steps, num_training_steps= num_training_steps)
        #save infor
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Total optimization steps = %d", num_training_steps)
        
        model.train()
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_f1 = 0.0
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, mask_ids, class_ids = batch

                logits = model(input_ids, segment_ids, mask_ids)
                loss = loss_function(logits,class_ids)
                if device.type =="cuda" and n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        tr_loss_avg = (tr_loss-logging_loss)/args.logging_steps
                        writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                        logging_loss = tr_loss
            if args.do_eval:
                score = evaluate(args, eval_data, model, device.type)
                
                # add eval result to tensorboard
                f1_score = score.f1
                writer.add_scalar("Eval/precision", score.precision, ep)
                writer.add_scalar("Eval/recall", score.recall, ep)
                writer.add_scalar("Eval/f1_score", score.f1, ep)
                # save the best performs model
                if f1_score > best_f1:
                    logger.info(f"----------the best class f1 is {f1_score}")
                    best_f1 = f1_score
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)

                # Good practice: save your training arguments together with the trained model
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

            # logger.info(f'epoch {ep}, train loss: {tr_loss}')
        # writer.add_graph(model)
        writer.close()

if __name__ == "__main__":
    main()
    pass