import os
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

from data.coherency import CoherencyDataSet, UtterancesWrapper, BertWrapper, GloveWrapper
from model.coh_random import RandomCoherenceRanker
from model.cos_ranking import CosineCoherenceRanker
from model.coh_model3 import MTL_Model3
from model.coh_model4 import MTL_Model4

### Hyper Parameters ###
batch_size = 32
learning_rate = 5e-2 # inc!
num_epochs = 10
lstm_hidden_size = 150
lr_schedule = [2,4]
lstm_layers = 1 #keep 1
max_seq_len = 285 # for glove using the nltk tokenizer

########################

# def ranking_score(model, orig_sents, orig_DAs, permutations_sents, permutations_DAs):
def ranking_score(model, all_dialogues, all_acts, len_dialog):
    # if len(permutations_sents) == 0:
        # print("caught permutations with length 0")
        # return 0.0
    scores, loss = model(all_dialogues, all_acts, len_dialog)
    score = 0.0
    
    coh_scores = scores.cpu().detach().numpy().tolist()
    orig_score = coh_scores[0]
    perturb_amount = len(coh_scores[1:])
    
    if perturb_amount == 0:
        return None

    for perm_score in coh_scores[1:]:
        if orig_score >= perm_score:
            score += 1.0

    return score/perturb_amount

    # for (perm_sent, perm_DA) in zip(permutations_sents, permutations_DAs):
        # score += model.compare(orig_sents, orig_DAs, perm_sent, perm_DA)
    # # for (perm_sent, perm_DA) in zip(permutations_sents, permutations_DAs):
        # # score += model( perm_sent, perm_DA, orig_sents, orig_DAs)

    # return score/(len(permutations_sents)) # len *2 if top uncommented

def ranking_score_live(scores, loss, len_dialog):
    # if len(permutations_sents) == 0:
        # print("caught permutations with length 0")
        # return 0.0
    score = 0.0
    
    coh_scores = scores.cpu().detach().numpy().tolist()
    orig_score = coh_scores[0]
    perturb_amount = len(coh_scores[1:])

    for perm_score in coh_scores[1:]:
        if orig_score >= perm_score:
            score += 1.0

    return score/perturb_amount

def insertion_score(model, orig_sents):
    max_i = len(orig_sents)-1
    values = []
    for i, sent in enumerate(orig_sents):
        for y in range(len(orig_sents)):
            curr_removed = orig_sents[0:i] + [] if i == max_i else orig_sents[i+1:]
            curr_removed.insert(y, sent)
            score = model(curr_removed, [])
            values.append((0 if y != i else 1, score))

    # values = values + [(0, 0.0)]*(pad_len-len(values))
    true = np.array([[i for (i,x) in values]])
    score = np.array([[x for (i,x) in values]])
    return label_ranking_average_precision_score(true, score)
    # return values

def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

def main():
    args = parse_args()
    init_logging(args)
    # Init randomization
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    cuda_device_name = "cuda:{}".format(args.cuda)
    device = torch.device(cuda_device_name if torch.cuda.is_available() else 'cpu')

    output_model_file =os.path.join(args.datadir, "model_{}.ckpt".format(args.task))
    logging.info("Model output file: {}".format(output_model_file))

    stop = [x for x in stopwords.words('english')]
    stop = [i for sublist in stop for i in sublist]
    # dset = CoherencyDataSet(args.datadir, args.task, word_filter=lambda c: c not in stop)
    dset = CoherencyDataSet(args.datadir, args.task, word_filter=None)

    if args.embedding == 'bert':
        embed_dset = BertWrapper(dset, device, True)
    elif args.embedding == 'glove':
        embed_dset = GloveWrapper(dset, device, max_seq_len)
    elif args.embedding == 'elmo':
        assert False, "elmo not yet supported!"

    dataloader = DataLoader(embed_dset, batch_size=1, shuffle=False, num_workers=4)

    # model = RandomCoherenceRanker(args.seed)
    # model = CosineCoherenceRanker(args.seed)
    model = MTL_Model3(embed_dset.embed_dim, lstm_hidden_size, lstm_layers, 4, device).to(device)
    # model.load_state_dict(torch.load(output_model_file))
    # model = MTL_Model4(embed_dset.embed_dim, lstm_hidden_size, lstm_layers, 4, device).to(device)

    logging.info("Used Model: {}".format(str(model)))

    if args.do_train:

        live_data = open("live_data_{}.csv".format(args.task), 'w', buffering=1)
        live_data.write("{},{},{}\n".format('step', 'loss', 'score'))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = MultiStepLR(optimizer, milestones=lr_schedule, gamma=0.1)
        hinge = HingeEmbeddingLoss(reduction='none', margin=0.0).to(device)

        for epoch in trange(num_epochs, desc="Epoch"):
            scheduler.step()
            # for i,((d,a), (pds, pas)) in tqdm(enumerate(embed_dset), total=len(embed_dset), desc='Iteration'):
            for i,(all_dialogues, all_acts, len_dialog) in tqdm(enumerate(dataloader), total=len(embed_dset), desc='Iteration'):
                if args.test and i > 3: break

                all_dialogues = all_dialogues.squeeze(0).to(device)
                logging.info("all_dialogue size: {}".format(all_dialogues.size()))
                all_acts = all_acts.squeeze(0).to(device)
                len_dialog = len_dialog.squeeze(0).to(device)

                if all_dialogues.size(0) < 2*len_dialog:
                    continue # sometimes for task HUP, the dialog is just to short to create permutations

                coh_base, loss_base = model(all_dialogues, all_acts, len_dialog)
                hinge_pred = coh_base[1:]
                hinge_target = torch.cat([coh_base[0].unsqueeze(0) for _ in range(hinge_pred.size(0))], 0)
                h = hinge(hinge_target, hinge_pred)
                m = torch.tensor([1.0]+([1.0/hinge_pred.size(0)] * hinge_pred.size(0))).to(device)
                loss = torch.dot(loss_base, m) + torch.sum(h)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 10 == 0: # write to live_data file
                    score = ranking_score_live(coh_base, loss_base, len_dialog)
                    live_data.write("{},{},{}\n".format(((epoch*len(embed_dset))+i)/10, loss.item(), score))
                    live_data.flush()

            torch.cuda.empty_cache()

        torch.save(model.state_dict(), output_model_file)

    if args.do_eval:

        # model.load_state_dict(torch.load(output_model_file))
        model.eval()
        rankings = []

        # for i,((d,a), (pds, pas)) in tqdm(enumerate(embed_dset), total=len(embed_dset)):
        for i,(all_dialogues, all_acts, len_dialog) in tqdm(enumerate(embed_dset), total=len(embed_dset), desc='Iteration'):
            if args.test and i > 3: break

            score = ranking_score(model, all_dialogues, all_acts, len_dialog)
            if score:
                rankings.append(score)

            # for ten in pds:
                # ten.detach()
            # d.detach()
            torch.cuda.empty_cache()

        # if model.collect_da_predictions:
            # da_pred = model.da_predictions
            # target = [y for (x,y) in da_pred]
            # pred = [x for (x,y) in da_pred]
            # print("DA accuracy: ", accuracy_score(target, pred))
            # logging.info("Accuracy DA: {}".format(accuracy_score(target, pred)))

        print("Coherence Accuracy: ", np.array(rankings).mean())
        logging.info("Accuracy Result: {}".format(np.array(rankings).mean()))

def init_logging(args):
    now = datetime.datetime.now()
    logfile = os.path.join(args.logdir, 'coherency_task_{}_{}.log'.format(args.task, now.strftime("%Y-%m-%d_%H:%M:%S")))
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG, format='%(levelname)s:%(message)s')
    print("Logging to ", logfile)

    logging.info("Used Hyperparameters:")

    logging.info("learning_rate = {}".format(learning_rate))
    logging.info("num_epochs = {}".format(num_epochs))
    logging.info("lstm_hidden_size = {}".format(lstm_hidden_size))
    logging.info("lr_schedule = {}".format(lr_schedule))
    logging.info("lstm_layers = {}".format(lstm_layers))
    logging.info("max_seq_len = {}".format(max_seq_len))
    logging.info("========================")
    logging.info("task = {}".format(args.task))
    logging.info("seed = {}".format(args.seed))
    logging.info("embedding = {}".format(args.embedding))

def parse_args():
    parser = argparse.ArgumentParser()
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
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--embedding',
                        required=True,
                        type=str,
                        default="bert",
                        help="""from which embedding should the word ids be used.
                                alternatives: bert|elmo|glove """)
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
    return parser.parse_args()


class MTL_Loss(nn.Module):
    # see: https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
    def __init__(self):
        super(MTL_Loss,self).__init__()
        self.sigma1 = torch.tensor(1.0, requires_grad=True)
        self.sigma2 = torch.tensor(1.0, requires_grad=True)

    """ both x and y need to be torch Variables ! """
    def forward(self,x,y):
        #TODO: define the loss function with the two sigmas based on MSE and crossentropy
        pass

def bert_experiment():
    #_shuf BERT cache dir
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    bert = BertModel.from_pretrained(BERT_MODEL_NAME,
              cache_dir=cache_dir).to(device)

    sent = "[CLS] hello there, how are you? [SEP] hey, i am fine, and you? [SEP] me too, i just had a great breakfast. [SEP] ok, that sounds nice [SEP]"
    sent = bert_tok.tokenize(sent)
    print(len(sent))
    sent = torch.tensor(bert_tok.convert_tokens_to_ids(sent), dtype=torch.long)
    print(sent.unsqueeze(0).size())
    logits, cls = bert(sent.unsqueeze(0), output_all_encoded_layers=False)
    print(logits.size())



if __name__ == '__main__':
    main()
