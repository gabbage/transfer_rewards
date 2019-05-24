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
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, BatchSampler,
                              TensorDataset, Dataset)
from torch.nn.modules.distance import CosineSimilarity

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel,BertPreTrainedModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer


class CoherencyDataSet(Dataset):
    def __init__(self, data_dir, task, word_filter=None):
        super(CoherencyDataSet, self).__init__()
        assert task == 'up' or task =='us' or task == 'hup' or task == 'ui'
        data_file_shuf = os.path.join(data_dir, "coherency_dset_{}.txt".format(task))
        assert os.path.isfile(data_file_shuf), "could not find dataset file: {}".format(data_file_shuf)

        self.word_filter = word_filter
        self.task = task 

        self.dialogues = []
        self.acts = []
        self.permutations = []

        self.indices_convert = dict()
        utt_idx = 0

        self.max_dialogue_len = 0

        with open(data_file_shuf, 'r') as f:
            coh_df = pd.read_csv(f, sep='|', names=['acts', 'utts', 'perm'])

        for (idx, row) in coh_df.iterrows():

            r_acts = [int(x) for x in row['acts'].split(' ')]
            self.acts.append(r_acts)

            for i in range(len(r_acts)):
                self.indices_convert[utt_idx] = idx
                utt_idx += 1

            r_utts = literal_eval(row['utts'])
            self.dialogues.append(r_utts)
            self.max_dialogue_len = max(self.max_dialogue_len, len(r_utts))
            
            r_perms = literal_eval(row['perm'])
            self.permutations.append(r_perms)

    def create_permutations(self, idx):
        acts = self.acts[idx]
        sents = self.dialogues[idx]
        perms = self.permutations[idx]

        perm_sents = []
        perm_acts = []
        if self.task == 'us':
            for perm in perms:
                dialogue_ix, curr_ix = perm[0], perm[1]

                (act, utt) = self.get_utt_by_idx(dialogue_ix)
                perm_sents.append(deepcopy(sents))
                perm_acts.append(deepcopy(acts))
                perm_sents[-1][curr_ix] = deepcopy(utt)
                perm_acts[-1][curr_ix] = act

        elif self.task == 'up' or self.task == 'hup' or self.task == 'ui':
            for perm in perms:
                perm_sents.append(deepcopy(sents))
                perm_acts.append(deepcopy(acts))
                for (i_to, i_from) in zip(list(range(len(perm))), perm):
                    perm_sents[-1][i_to] = sents[i_from]
                    perm_acts[-1][i_to] = acts[i_from]
        return perm_sents, perm_acts
    
    def get_utt_ix(self, idx):
        return [i for (i, b) in self.indices_convert.items() if b == idx]

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, idx):
        dialog = self.dialogues[idx]
        acts = self.acts[idx]
        perms = self.permutations[idx]
        perm_utts, perm_acts = self.create_permutations(idx)
        if self.word_filter is not None:
            dialog = [list(filter(self.word_filter, x)) for x in dialog]
            for p in perm_utts:
                p = [list(filter(self.word_filter, x)) for x in p]

        return ((dialog, acts), (perm_utts, perm_acts))
        # return ((self.coherences[idx], self.permutations[idx]), self.dialogues[idx])
    
    def get_utt_by_idx(self, idx):
        base_idx = self.indices_convert[idx]
        min_base_idx = min(self.get_utt_ix(base_idx))
        j = idx - min_base_idx
        utt = self.dialogues[base_idx][j]
        act = self.acts[base_idx][j]
        return (act, utt)

class CoherencyPairDataSet(Dataset):
    def __init__(self, data_dir, task, word_filter=None):
        super(CoherencyPairDataSet, self).__init__()
        assert task == 'up' or task =='us' or task == 'hup' or task == 'ui'
        data_file_shuf = os.path.join(data_dir, "coherency_dset_{}.txt".format(task))
        assert os.path.isfile(data_file_shuf), "could not find dataset file: {}".format(data_file_shuf)

        self.word_filter = word_filter
        self.task = task 
        self.max_dialogue_len = 0

        self.acts = []
        self.perm_acts = []
        self.utts = []
        self.perm_utts = []

        with open(data_file_shuf, 'r') as f:
            coh_df = pd.read_csv(f, sep='|', names=['acts', 'utts', 'perm_acts', 'perm_utts'])

        for (idx, row) in coh_df.iterrows():
            r_acts = [int(x) for x in row['acts'].split(' ')]
            self.acts.append(r_acts)
            
            r_perm_acts = [int(x) for x in row['perm_acts'].split(' ')]
            self.perm_acts.append(r_perm_acts)

            self.utts.append(literal_eval(row['utts']))
            self.perm_utts.append(literal_eval(row['perm_utts']))
            self.max_dialogue_len = max(self.max_dialogue_len, len(self.utts[-1]))

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, idx):
        return (self.acts[idx], self.utts[idx], self.perm_acts[idx], self.perm_utts[idx])

class UtterancesWrapper(Dataset):
    """ This Wrapper can be used to walk through the DailyDialog corpus by sentence, not by dialog"""
    def __init__(self, coherency_dset):
        super(UtterancesWrapper, self).__init__()
        self.base = coherency_dset
        assert False, "This class is currently not supported"

    def __len__(self):
        return len(self.base.indices_convert)

    def __getitem__(self, idx):
        base_idx = self.base.indices_convert[idx]
        min_base_idx = min(self.base.get_utt_ix(base_idx))
        j = idx - min_base_idx
        utt = self.base.dialogues[base_idx][j]
        act = self.base.acts[base_idx][j]
        return (act, [utt])

class BertWrapper(Dataset):
    def __init__(self, base_dset, device, return_embeddding=True):
        super(BertWrapper, self).__init__()
        assert isinstance(base_dset, CoherencyDataSet)
        assert False, "This class is currently not supported"

        self.base = base_dset
        self.device = device
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME,
                  cache_dir=cache_dir).to(device)
        self.bert.eval()
        
        self.bert_tok = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=True)
        self.word2id = lambda x: self.bert_tok.convert_tokens_to_ids(x)
        self.embed_dim = self.bert.config.hidden_size
        
        cls_id, sep_id = self.word2id(["[CLS]"])[0], self.word2id(["[SEP]"])[0]

        self.cls = [cls_id] 
        self.sep = [sep_id]
        self.embed = return_embeddding #whether to return the embedding or the classification vector

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        labels, utts = self.base[idx]
        outputs = []

        max_len = max( map(len, utts))
        for utt in utts:
            bert_input = self.cls + self.word2id(utt) + (max_len-len(utt))*[0]
            bert_input = torch.LongTensor(bert_input).unsqueeze(0).to(self.device)
            encoding, cls_vec = self.bert(bert_input, output_all_encoded_layers=False)

            if self.embed:
                outputs.append(encoding.squeeze(0)) # remove the 'batch_size' dimension of the tensor
            else:
                outputs.append(cls_vec.squeeze(0))

        #TODO later: add type and attention vectors
        return (labels, outputs)

class GlovePairWrapper(Dataset):
    def __init__(self, base_dset, datadir, max_seq_len, max_dialogue_len, batch_size):
        super(GlovePairWrapper, self).__init__()
        assert isinstance(base_dset, CoherencyPairDataSet)

        self.base = base_dset
        self.datadir = datadir
        self.max_seq_len = max_seq_len
        self.max_dialogue_len = max_dialogue_len
        self.batch_size = batch_size
        self.embed_dim = 300
        self.vocab = self._build_vocab()
        self.vocab.load_vectors("glove.42B.300d")
        self.embed = nn.Embedding(len(self.vocab), self.embed_dim)
        self.embed.weight.data.copy_(self.vocab.vectors)

        self.sampler = list(BatchSampler(SequentialSampler(self.base), batch_size=self.batch_size, drop_last=True))

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        indices = self.sampler[idx]
        pad_word = self.vocab.stoi["<pad>"]

        def _embed_dialog(d):
            x = [self.embed(
                torch.tensor([w for w in utt], dtype=torch.long))
                    for utt in d]
            dial_ten = torch.cat([y.unsqueeze(0) for y in x], 0)
            return dial_ten

        batch_acts, batch_utts, batch_perm_acts, batch_perm_utts = [], [], [], []
        for i in indices:
            acts, utts, perm_acts, perm_utts = self.base[i]
            assert len(acts) == len(utts) and len(utts) == len(perm_acts) and len(perm_acts) == len(perm_utts), "not all lengths equal!"

            utts = [utt + [pad_word]*(self.max_seq_len-len(utt)) for utt in utts]
            perm_utts = [perm_utt + [pad_word]*(self.max_seq_len-len(perm_utt)) for perm_utt in perm_utts]
            #pad dialogues
            for i in range(self.max_dialogue_len - len(acts)):
                acts.append(0)
                perm_acts.append(0)
                utts.append([pad_word]*self.max_seq_len)
                perm_utts.append([pad_word]*self.max_seq_len)
            # assert len(acts) == self.max_dialogue_len, "hwuiae"

            batch_acts.append(torch.tensor(acts))
            batch_utts.append(_embed_dialog(utts))
            batch_perm_acts.append(torch.tensor(perm_acts))
            batch_perm_utts.append(_embed_dialog(perm_utts))

        #shuffle the data in a structured way
        cnd = idx%2 == 1
        utts1 = (batch_utts if cnd else batch_perm_utts)[0::2] + (batch_utts if not cnd else batch_perm_utts)[1::2]
        utts2 = (batch_utts if not cnd else batch_perm_utts)[0::2] + (batch_utts if cnd else batch_perm_utts)[1::2]
        acts1 = (batch_acts if cnd else batch_perm_acts)[0::2] + (batch_acts if not cnd else batch_perm_acts)[1::2]
        acts2 = (batch_acts if not cnd else batch_perm_acts)[0::2] + (batch_acts if cnd else batch_perm_acts)[1::2]
        coh_values = torch.tensor([idx%2, (idx+1)%2]*int(self.batch_size/2))

        utts1 = torch.stack(utts1, 0).detach()
        utts2 = torch.stack(utts2, 0).detach()
        acts1 = torch.stack(acts1, 0).detach()
        acts2 = torch.stack(acts2, 0).detach()

        return utts1, utts2, acts1, acts2, coh_values

    def _build_vocab(self):
        cnt = Counter()
        with open(os.path.join(self.datadir, "itos.txt"), "r") as f:
            for i, word in enumerate(f):
                cnt[word[:-1]] += 1
        return tt.vocab.Vocab(cnt)

class GloveWrapper(Dataset):
    def __init__(self, base_dset, device, max_seq_len):
        super(GloveWrapper, self).__init__()
        assert isinstance(base_dset, CoherencyDataSet)

        self.base = base_dset
        self.device = device
        self.max_seq_len = max_seq_len
        self.embed_dim = 300
        self.vocab = self._build_vocab()
        self.vocab.load_vectors("glove.42B.300d")
        self.embed = nn.Embedding(len(self.vocab), self.embed_dim)
        self.embed.weight.data.copy_(self.vocab.vectors)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        (dialog, acts), (perm_utts, perm_acts) = self.base[idx]

        pad_dialogue = [[utt + ["<pad>"]*(self.max_seq_len-len(utt)) for utt in dialog]]
        pad_perm_utts = [[utt + ["<pad>"]*(self.max_seq_len-len(utt)) for utt in p] for p in perm_utts]

        def _embed_dialog(d):
            # return [self.embed(
                # torch.tensor([self.vocab.stoi[w] for w in utt], dtype=torch.long)).to(self.device)
                    # for utt in d]
            x = [self.embed(
                torch.tensor([self.vocab.stoi[w] for w in utt], dtype=torch.long))
                    for utt in d]
            dial_ten = torch.cat([y.unsqueeze(0) for y in x], 0)
            return dial_ten

        glove_perm_utts = [_embed_dialog(d).detach() for d in pad_perm_utts+pad_dialogue]
        all_dialogues = torch.cat( glove_perm_utts, 0)
        all_dialogues.detach()

        # act-1 since the input classes are [1.4], but we get [0..3] predicted
        acts = torch.tensor([act-1 for act in acts])
        perm_acts = [torch.tensor([act-1 for act in pact]) for pact in perm_acts]
        all_acts = torch.cat([acts] + perm_acts, 0)
        all_acts.detach()

        return (all_dialogues, all_acts, torch.tensor(len(dialog)))
        # return (glove_dialogue, acts), (glove_perm_utts, perm_acts)
    
    def get_word2id(self):
        return self.vocab.stoi

    def _build_vocab(self):
        cnt = Counter()
        for d in self.base.dialogues:
            for sent in d:
                for word in sent:
                    cnt[word.lower()] += 1
        return tt.vocab.Vocab(cnt)

