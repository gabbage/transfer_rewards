import os
import operator
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
from torch.autograd import Variable
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

    train_datasetfile = os.path.join(args.datadir,"train", "coherency_dset_{}.txt".format(str(args.task)))
    val_datasetfile = os.path.join(args.datadir, "validation", "coherency_dset_{}.txt".format(str(args.task)))
    test_datasetfile = os.path.join(args.datadir, "test", "coherency_dset_{}.txt".format(str(args.task)))

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
        model = MTL_Model4(args, device).to(device)
    else:
        assert False, "specified model not supported"

    logging.info("Used Model: {}".format(str(model)))

    if args.do_train:

        if args.live:
            live_data = open("live_data_{}.csv".format(str(args.task)), 'w', buffering=1)
            live_data.write("{},{},{}\n".format('step', 'loss', 'score'))

        train_dl = get_dataloader(train_datasetfile, args)
        val_dl = get_dataloader(val_datasetfile, args)

        if args.loss == "mtl":
            sigma_1 = nn.Parameter(torch.tensor(1.0, requires_grad=True).to(device))
            sigma_2 = nn.Parameter(torch.tensor(1.0, requires_grad=True).to(device))
            optimizer = torch.optim.Adam(list(model.parameters())+[sigma_1,sigma_2], lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        hinge = torch.nn.MarginRankingLoss(reduction='none', margin=0.1).to(device)
        epoch_scores = dict()

        for epoch in trange(args.epochs, desc="Epoch"):
            output_model_file_epoch = os.path.join(args.datadir, "{}_task-{}_loss-{}_epoch-{}.ckpt".format(str(model), str(args.task),str(args.loss), str(epoch)))

            for i,((utts_left, utts_right), 
                    (coh_ixs, (acts_left, acts_right)), (len_u1, len_u2, len_d1, len_d2)) in tqdm(enumerate(train_dl),
                    total=len(train_dl), desc='Training', postfix="LR: {}".format(args.learning_rate)):
                if args.test and i >= 10: break

                coh_ixs = coh_ixs.to(device)
                coh1, (_,loss1) = model(utts_left.to(device),
                        acts_left.to(device),
                        (len_u1.to(device), len_d1.to(device)))
                coh2, (_,loss2) = model(utts_right.to(device),
                        acts_right.to(device), 
                        (len_u2.to(device), len_d2.to(device)))

                # coh_ixs is of the form [0,1,1,0,1], where 0 indicates the first one is the more coherent one
                # for this loss, the input is expected as [1,-1,-1,1,-1], where 1 indicates the first to be coherent, while -1 the second
                # therefore, we need to transform the coh_ixs accordingly
                loss_coh_ixs = torch.add(torch.add(coh_ixs*(-1), torch.ones(coh_ixs.size()).to(device))*2, torch.ones(coh_ixs.size()).to(device)*(-1))
                if args.loss == "da":
                    loss = torch.div(loss1 + loss2, 2)
                elif args.loss == "coh":
                    loss = hinge(coh1, coh2, loss_coh_ixs)
                elif args.loss == "mtl":
                    loss = torch.div(torch.div(loss1 + loss2, 2), sigma_1**2) + torch.div(hinge(coh1, coh2, loss_coh_ixs), sigma_2**2) + torch.log(sigma_1) + torch.log(sigma_2)

                optimizer.zero_grad()
                loss.sum().backward()
                optimizer.step()

                if i % 10 == 0 and args.live: # write to live_data file
                    _, pred = torch.max(torch.cat([coh1.unsqueeze(1), coh2.unsqueeze(1)], dim=1), dim=1)
                    pred = pred.detach().cpu().numpy()
                    coh_ixs = coh_ixs.detach().cpu().numpy()
                    score = accuracy_score(coh_ixs, pred)
                    live_data.write("{},{},{}\n".format(((epoch*len(train_dl))+i)/10, loss.mean().item(), score))
                    live_data.flush()

            # torch.cuda.empty_cache()

            #save after every epoch
            torch.save(model.state_dict(), output_model_file_epoch)

            # evaluate
            rankings = []
            da_rankings = []
            with torch.no_grad():
                for i,((utts_left, utts_right), 
                        (coh_ixs, (acts_left, acts_right)), (len_u1, len_u2, len_d1, len_d2)) in tqdm(enumerate(val_dl),
                        total=len(val_dl), desc='Evaluation', postfix="LR: {}".format(args.learning_rate)):
                    if args.test and i >= 10: break

                    coh1, lda1 = model(utts_left.to(device), acts_left.to(device), (len_u1.to(device), len_d1.to(device)))
                    coh2, lda2 = model(utts_right.to(device), acts_right.to(device), (len_u2.to(device), len_d2.to(device)))

                    _, pred = torch.max(torch.cat([coh1.unsqueeze(1), coh2.unsqueeze(1)], dim=1), dim=1)
                    pred = pred.detach().cpu().numpy()

                    if lda1 != None and lda2 != None:
                        da1 = lda1[0].detach().cpu().numpy()
                        da2 = lda2[0].detach().cpu().numpy()
                        acts_left = acts_left.view(acts_left.size(0)*acts_left.size(1)).detach().cpu().numpy()
                        acts_right = acts_right.view(acts_right.size(0)*acts_right.size(1)).detach().cpu().numpy()
                        da_rankings.append(accuracy_score(da1, acts_left))
                        da_rankings.append(accuracy_score(da2, acts_right))

                    coh_ixs = coh_ixs.detach().cpu().numpy()
                    rankings.append(accuracy_score(coh_ixs, pred))

                if args.loss == "mtl":
                    epoch_scores[epoch] = (np.array(rankings).mean() + np.array(da_rankings).mean())
                    logging.info("epoch {} has Coh. Acc: {} ; DA Acc: {}".format(epoch, np.array(rankings).mean(), np.array(da_rankings).mean()))
                elif args.loss == "da":
                    epoch_scores[epoch] = (np.array(da_rankings).mean())
                    logging.info("epoch {} has DA Acc: {}".format(epoch, np.array(da_rankings).mean()))
                elif args.loss == "coh":
                    epoch_scores[epoch] = (np.array(rankings).mean())
                    logging.info("epoch {} has Coh. Acc: {}".format(epoch, np.array(rankings).mean()))

        # get maximum epoch
        best_epoch = max(epoch_scores.items(), key=operator.itemgetter(1))[0]
        print("Best Epoch, ie final Model Number: {}".format(best_epoch))
        logging.info("Best Epoch, ie final Model Number: {}".format(best_epoch))

    if args.do_eval:
        if model != None: # do non random evaluation
            if args.model != "cosine" and  args.model != "random" and not args.test:
                output_model_file_epoch = os.path.join(args.datadir, "{}_task-{}_loss-{}_epoch-{}.ckpt".format(str(model), str(args.task),str(args.loss), str(args.best_epoch)))
                model.load_state_dict(torch.load(output_model_file_epoch))
            model.to(device)
            model.eval()

        train_dl = get_dataloader(train_datasetfile, args)
        test_dl = get_dataloader(test_datasetfile, args)

        def _eval_datasource(dl):
            rankings = []
            da_rankings = []
            da_y_pred = []
            da_y_true = []

            for i,((utts_left, utts_right), 
                    (coh_ixs, (acts_left, acts_right)), (len_u1, len_u2, len_d1, len_d2)) in tqdm(enumerate(dl),
                    total=len(dl), desc='Iteration Train', postfix="LR: {}".format(args.learning_rate)):
                if args.test and i > 5: break

                if model == None: #generate random values
                    pred = [random.randint(0,1) for _ in range(coh_ixs.size(0))]
                else:
                    coh1, lda1 = model(utts_left.to(device), acts_left.to(device), (len_u1.to(device), len_d1.to(device)))
                    coh2, lda2 = model(utts_right.to(device), acts_right.to(device), (len_u2.to(device), len_d2.to(device)))

                    _, pred = torch.max(torch.cat([coh1.unsqueeze(1), coh2.unsqueeze(1)], dim=1), dim=1)
                    pred = pred.detach().cpu().numpy()

                    if lda1 != None and lda2 != None:
                        da1 = lda1[0].detach().cpu().numpy()
                        da2 = lda2[0].detach().cpu().numpy()
                        acts_left = acts_left.view(acts_left.size(0)*acts_left.size(1)).detach().cpu().numpy()
                        acts_right = acts_right.view(acts_right.size(0)*acts_right.size(1)).detach().cpu().numpy()
                        da_y_pred = da_y_pred + acts_left.tolist() + acts_right.tolist()
                        da_y_true = da_y_true + da1.tolist() + da2.tolist()

                coh_ixs = coh_ixs.detach().cpu().numpy()
                rankings.append(accuracy_score(coh_ixs, pred))

                torch.cuda.empty_cache()

            return rankings, (da_y_true, da_y_pred)

        rankings_train, da_vals_train = _eval_datasource(train_dl)
        rankings_test, da_vals_test = _eval_datasource(test_dl)

        print("Coherence Accuracy Train: ", np.array(rankings_train).mean())
        logging.info("Coherence Accuracy Train: {}".format(np.array(rankings_train).mean()))
        print("Coherence Accuracy test: ", np.array(rankings_test).mean())
        logging.info("Coherence Accuracy test: {}".format(np.array(rankings_test).mean()))
        if len(da_vals_train[0]) > 0:
            print("DA Accuracy Train: ", accuracy_score(da_vals_train[0], da_vals_train[1]))
            logging.info("DA Accuracy Train: {}".format(accuracy_score(da_vals_train[0], da_vals_train[1])))
            print("DA MicroF1 Train: ", f1_score(da_vals_train[0], da_vals_train[1], average='micro'))
            logging.info("DA MicroF1 Train: {}".format(f1_score(da_vals_train[0], da_vals_train[1], average='micro')))

        if len(da_vals_test[0]) > 0:
            print("DA Accuracy test: ", accuracy_score(da_vals_test[0], da_vals_test[1]))
            logging.info("DA Accuracy test: {}".format(accuracy_score(da_vals_test[0], da_vals_test[1])))
            print("DA MicroF1 test: ", f1_score(da_vals_test[0], da_vals_test[1], average='micro'))
            logging.info("DA MicroF1 test: {}".format(f1_score(da_vals_test[0], da_vals_test[1], average='micro')))

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
    logging.info("loss = {}".format(args.loss))
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
    parser.add_argument('--loss',
                        type=str,
                        default="mtl",
                        help="""with which loss the dataset should be trained/evaluated.
                                alternatives: mtl | da | coh """)
    parser.add_argument('--task',
                        required=True,
                        type=str,
                        default="up",
                        help="""for which task the dataset should be created.
                                alternatives: up (utterance permutation)
                                              us (utterance sampling)
                                              hup (half utterance petrurbation) """)
    parser.add_argument('--best_epoch',
                        type=int,
                        default = 0,
                        help= "when evaluating, tell the best epoch to choose the file")
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
    parser.add_argument('--live',
                        action='store_true',
                        help= "whether to do live output of accuracy and loss values")
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
