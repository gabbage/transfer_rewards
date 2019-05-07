import os
import datetime
import logging
import sys
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from ast import literal_eval
from tqdm import tqdm, trange
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, label_ranking_average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext as tt
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.nn.modules.distance import CosineSimilarity

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel,BertPreTrainedModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer

### Hyper Parameters ###
BERT_MODEL_NAME = "bert-base-uncased"
batch_size = 32

########################

class CoherencyDataSet(Dataset):
    def __init__(self, data_dir, task, word_filter=None):
        super(CoherencyDataSet, self).__init__()
        assert task == 'up' or task =='us' or task == 'hup' 
        data_file_shuf = os.path.join(data_dir, "coherency_dset_{}_shuf.txt".format(task))
        assert os.path.isfile(data_file_shuf), "could not find dataset file: {}".format(data_file_shuf)

        self.word_filter = word_filter

        self.dialogues = []
        self.acts = []
        self.coherences = []

        self.indices_convert = dict()
        utt_idx = 0

        with open(data_file_shuf, 'r') as f:
            coh_df = pd.read_csv(f, sep='|', names=['coh', 'acts', 'utts'])

        for (idx, row) in coh_df.iterrows():
            self.coherences.append(int(row['coh']))

            r_acts = [int(x) for x in row['acts'].split(' ')]
            self.acts.append(r_acts)

            for i in range(len(r_acts)):
                self.indices_convert[utt_idx] = idx
                utt_idx += 1

            r_utts = literal_eval(row['utts'])
            self.dialogues.append(r_utts)
    
    def get_utt_ix(self, idx):
        return [i for (i, b) in self.indices_convert.items() if b == idx]

    def __len__(self):
        return len(self.coherences)

    def __getitem__(self, idx):
        if self.word_filter is not None:
            filter_dialogues = [list(filter(self.word_filter, x)) for x in self.dialogues[idx]]
            return ((self.coherences[idx], self.acts[idx]), filter_dialogues)

        return ((self.coherences[idx], self.acts[idx]), self.dialogues[idx])

class UtterancesWrapper(Dataset):
    """ This Wrapper can be used to walk through the DailyDialog corpus by sentence, not by dialog"""
    def __init__(self, coherency_dset):
        super(UtterancesWrapper, self).__init__()
        self.base = coherency_dset

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

        self.base = base_dset
        self.device = device
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME,
                  cache_dir=cache_dir).to(device)
        self.bert.eval()
        
        self.bert_tok = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=True)
        self.word2id = lambda x: self.bert_tok.convert_tokens_to_ids(x)
        
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

class GloveWrapper(Dataset):
    def __init__(self, base_dset, device):
        super(GloveWrapper, self).__init__()
        assert isinstance(base_dset, CoherencyDataSet)

        self.base = base_dset
        self.vocab = self._build_vocab()
        self.vocab.load_vectors("glove.42B.300d")
        self.embed = nn.Embedding(len(self.vocab), 300)
        self.embed.weight.data.copy_(self.vocab.vectors)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        (label, dialogue) = self.base[idx]
        seq_len = max([len(utt) for utt in dialogue])
        pad_dialogue = [utt + ["<pad>"]*(seq_len-len(utt)) for utt in dialogue]
        glove_dialogue = [self.embed(
                torch.tensor([self.vocab.stoi[w] for w in utt], dtype=torch.long))
                    for utt in pad_dialogue]

        return (label, glove_dialogue)
    
    def get_word2id(self):
        return self.vocab.stoi

    def _build_vocab(self):
        cnt = Counter()
        for d in self.base.dialogues:
            for sent in d:
                for word in sent:
                    cnt[word.lower()] += 1
        return tt.vocab.Vocab(cnt)

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

def main():
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
                        default="/home/sebi/code/transfer_rewards/sub_rewards",
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
                        type=bool,
                        default = False,
                        help= "just do a test run on small amount of data")
    args = parser.parse_args()

    # Init randomization
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # BERT cache dir

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    stop = [x for x in stopwords.words('english')]
    stop = [i for sublist in stop for i in sublist]
    dset = CoherencyDataSet(args.datadir, args.task, word_filter=lambda c: c not in stop)

    if args.embedding == 'bert':
        embed_dset = BertWrapper(dset, device, True)
    elif args.embedding == 'glove':
        embed_dset = GloveWrapper(dset, device)
    elif args.embedding == 'elmo':
        assert False, "elmo not yet supported!"

    cos = CosineSimilarity(dim=0).to(device)
    cos_values = []
    coh_values = []
    print(len(dset))

    for i,(lbl, output) in tqdm(enumerate(embed_dset), total=len(embed_dset)):
        if args.test and i > 10: break

        dialog_coherencies = []
        for j in range(len(output)-1):
            vec1 = output[j]
            vec2 = output[j+1]
            c = cos(vec1.mean(0), vec2.mean(0)).cpu().item()
            dialog_coherencies.append(c)

        coh = float(lbl[0])
        cos_values.append( np.array(c).mean())
        coh_values.append(coh)

        for ten in output:
            ten.detach()
        torch.cuda.empty_cache()

    # print(mean_squared_error(coh_values, cos_values))
    cos_pred = list(map(round, cos_values))
    #TODO: print accuracy for both classes, see how good it can discriminate!
    print("accuracy = ", accuracy_score(coh_values, cos_pred))
    print("F1 score = ", f1_score(coh_values, cos_pred, average='macro'))
    print("MRR = ", label_ranking_average_precision_score(coh_values, cos_values))

if __name__ == '__main__':
    main()

    # output_model_file = os.path.join(args.outputdir, WEIGHTS_NAME)
    # output_config_file = os.path.join(args.outputdir, CONFIG_NAME)

    # # Logging file
    # now = datetime.datetime.now()
    # logfile = os.path.join(args.logdir, 'MTL_{}.log'.format(now.strftime("%Y-%m-%d_%H:%M:%S")))
    # logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG, format='%(levelname)s:%(message)s')
    # print("Logging to ", logfile)

    # logging.info("Used Hyperparameters:")
    # logging.info("BERT_MODEL_NAME = {}".format(BERT_MODEL_NAME))
    # logging.info("batch_size = {}".format(batch_size))

def bert_experiment():
    # BERT cache dir
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
