import os
import random
import datetime
import logging
import sys
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import Counter
from ast import literal_eval
from tqdm import tqdm, trange
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, label_ranking_average_precision_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext as tt
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.nn.modules.distance import CosineSimilarity
from torch.nn.modules import HingeEmbeddingLoss

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel,BertPreTrainedModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME

from data.coherency import CoherencyDataSet, UtterancesWrapper, BertWrapper, GloveWrapper, CoherencyPairDataSet, GlovePairWrapper, CoherencyPairBatchWrapper
from model.mtl_models import CosineCoherence, MTL_Model3, MTL_Model4
from data_preparation import get_dataloader

def main():
    args = parse_args()
    init_logging(args)
    # Init randomization
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cuda_device_name = "cuda:{}".format(args.cuda)
    device = torch.device(cuda_device_name if torch.cuda.is_available() else 'cpu')

    datasetfile = os.path.join(args.datadir, "coherency_dset_{}.txt".format(str(args.task)))
    dataloader = get_dataloader(datasetfile, args)

    if args.model == "cosine":
        if args.do_train:
            assert False, "cannot train the cosine model!"
        model = CosineCoherence(args, device)
    elif args.model == "random":
        if args.do_train:
            assert False, "cannot train the random model!"
        model = None
    elif args.model == "model-3":
        model = MTL_Model3(args, device).to(device)
    elif args.model == "model-4":
        assert False, "model-4 not yet supported"
    else:
        assert False, "specified model not supported"

    logging.info("Used Model: {}".format(str(model)))
    output_model_file =os.path.join(args.datadir, "{}_task-{}.ckpt".format(str(model), str(args.task)))
    logging.info("Model output file: {}".format(output_model_file))

    if args.do_train:

        live_data = open("live_data_{}.csv".format(str(args.task)), 'w', buffering=1)
        live_data.write("{},{},{}\n".format('step', 'loss', 'score'))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        hinge = HingeEmbeddingLoss(reduction='none', margin=0.0).to(device)

        for epoch in trange(args.epochs, desc="Epoch"):
            for i,((utts_left, utts_right), (coh_ixs, (acts_left, acts_right))) in tqdm(enumerate(dataloader),
                    total=len(dataloader), desc='Iteration', postfix="LR: {}".format(args.learning_rate)):
                if args.test and i > 3: break

                coh1, loss1 = model(utts_left.to(device), acts_left.to(device))
                coh2, loss2 = model(utts_right.to(device), acts_right.to(device))
                loss = loss1 + loss2 + hinge(coh1, coh2)

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

                if i % 10 == 0: # write to live_data file
                    _, pred = torch.max(torch.cat([coh1.unsqueeze(1), coh1.unsqueeze(1)], dim=1), dim=1)
                    pred = pred.detach().cpu().numpy()
                    score = accuracy_score(coh_ixs, pred)
                    live_data.write("{},{},{}\n".format(((epoch*len(dataloader))+i)/10, loss.mean().item(), score))
                    live_data.flush()

            torch.cuda.empty_cache()

            #save after every epoch
            torch.save(model.state_dict(), output_model_file)

    if args.do_eval:
        if model != None: # do non random evaluation
            if args.model != "cosine" and not args.test:
                model.load_state_dict(torch.load(output_model_file))
            model.to(device)
            model.eval()
        rankings = []
        da_rankings = []

        for i,((utts_left, utts_right), (coh_ixs, (acts_left, acts_right))) in tqdm(enumerate(dataloader),
                total=len(dataloader), desc='Iteration', postfix="LR: {}".format(args.learning_rate)):
            if args.test and i > 3: break

            if model == None: #generate random values
                pred = [random.randint(0,1) for _ in range(len(coh_ixs))]
            else:
                coh1, lda1 = model(utts_left.to(device), acts_left.to(device))
                coh2, lda2 = model(utts_right.to(device), acts_right.to(device))

                _, pred = torch.max(torch.cat([coh1.unsqueeze(1), coh2.unsqueeze(1)], dim=1), dim=1)
                pred = pred.detach().cpu().numpy()

                if lda1 != None and lda2 != None:
                    da1 = lda1[0].detach().cpu().numpy()
                    da2 = lda2[0].detach().cpu().numpy()
                    acts_left = acts_left.view(acts_left.size(0)*acts_left.size(1)).detach().cpu().numpy()
                    acts_right = acts_right.view(acts_right.size(0)*acts_right.size(1)).detach().cpu().numpy()
                    da_rankings.append(accuracy_score(da1, acts_left))
                    da_rankings.append(accuracy_score(da2, acts_right))

            rankings.append(accuracy_score(coh_ixs, pred))

            torch.cuda.empty_cache()

        print("Coherence Accuracy: ", np.array(rankings).mean())
        logging.info("Coherence Accuracy Result: {}".format(np.array(rankings).mean()))
        if len(da_rankings) > 0:
            print("DA Accuracy: ", np.array(da_rankings).mean())
            logging.info("DA Accuracy Result: {}".format(np.array(da_rankings).mean()))

def init_logging(args):
    now = datetime.datetime.now()
    logfile = os.path.join(args.logdir, 'coherency_{}_task_{}_{}.log'.format(args.model, args.task, now.strftime("%Y-%m-%d_%H:%M:%S")))
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG, format='%(levelname)s:%(message)s')
    print("Logging to ", logfile)

    logging.info("Used Hyperparameters:")

    logging.info("learning_rate = {}".format(args.learning_rate))
    logging.info("num_epochs = {}".format(args.epochs))
    logging.info("lstm_hidden_size = {}".format(args.lstm_hidden_size))
    logging.info("lstm_layers = {}".format(args.lstm_layers))
    logging.info("batch_size = {}".format(args.batch_size))
    logging.info("========================")
    logging.info("task = {}".format(args.task))
    logging.info("seed = {}".format(args.seed))
    logging.info("embedding = {}".format(args.embedding))

def parse_args():
    parser = argparse.ArgumentParser()
    # This also serves as a kind of configuration object, so some parameters are not ought to be changed (listed below)
    parser.add_argument("--outputdir",
                        required=False,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--datadir",
                        required=True,
                        type=str,
                        help="""The input directory where the files of daily
                        dialog are located. the folder should have
                        train/test/validation as subfolders""")
    parser.add_argument("--logdir",
                        default="./logs",
                        type=str,
                        help="the folder to save the logfile to.")
    parser.add_argument('--seed',
                        type=int,
                        default=80591,
                        help="random seed for initialization")
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help="")
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help="amount of epochs")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help="")
    parser.add_argument('--lstm_hidden_size',
                        type=int,
                        default=150,
                        help="hidden size for the lstm models")
    parser.add_argument('--embedding',
                        type=str,
                        default="glove",
                        help="""from which embedding should the word ids be used.
                                alternatives: bert|elmo|glove """)
    parser.add_argument('--model',
                        type=str,
                        default="cosine",
                        help="""with which model the dataset should be trained/evaluated.
                                alternatives: random | cosine | model-3 | model-4""")
    parser.add_argument('--task',
                        required=True,
                        type=str,
                        default="up",
                        help="""for which task the dataset should be created.
                                alternatives: up (utterance permutation)
                                              us (utterance sampling)
                                              hup (half utterance petrurbation) """)
    parser.add_argument('--test',
                        action='store_true',
                        help= "just do a test run on small amount of data")
    parser.add_argument('--cuda',
                        type=int,
                        default = 0,
                        help= "which cuda device to take")
    parser.add_argument('--do_train',
                        action='store_true',
                        help= "just do a test run on small amount of data")
    parser.add_argument('--do_eval',
                        action='store_true',
                        help= "just do a test run on small amount of data")
    ### usually unmodified parameters, keept here to have a config like object
    parser.add_argument('--num_classes',
                        type=int,
                        default=5,
                        help="DONT CHANGE. amount of classes 1-4 for DA acts, 0 for none")
    parser.add_argument('--lstm_layers',
                        type=int,
                        default=1,
                        help="DONT CHANGE. amount of layers for LSTM models")
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=300,
                        help="DONT CHANGE. embedding dimension")

    return parser.parse_args()


if __name__ == '__main__':
    main()
