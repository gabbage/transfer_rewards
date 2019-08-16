import os
import sys
import pandas as pd
import argparse
from copy import deepcopy
from collections import Counter
import torchtext as tt
from nltk.corpus import stopwords
from ast import literal_eval
from tqdm import tqdm
import torch
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader, Dataset
from allennlp.modules.elmo import batch_to_ids

test_amount = 20

def get_batches(filename, batch_size):
    coh_ixs = []
    acts1 = []
    acts2 = []
    utts1 = []
    utts2 = []

    with open(filename, 'r') as f:
        coh_df = pd.read_csv(f, sep='|', names=['coh_idx', 'acts1', 'utts1', 'acts2', 'utts2'])

    for (idx, row) in tqdm(coh_df.iterrows(), desc="Data Loading"):
        a1 = [int(x) for x in row['acts1'].split(' ')]
        acts1.append(a1)
        a2 = [int(x) for x in row['acts2'].split(' ')]
        acts2.append(a2)

        utts1.append(literal_eval(row['utts1']))
        utts2.append(literal_eval(row['utts2']))

        coh_ixs.append(int(row['coh_idx']))

    batch_indices = list(BatchSampler(SequentialSampler(range(len(acts1))), batch_size=batch_size, drop_last=False))
    batches = []
    for batch_ixs in batch_indices:
        batch = []
        for i in batch_ixs:
            batch.append((coh_ixs[i],
                acts1[i], utts1[i],
                acts2[i], utts2[i]))
        batches.append(batch)

    return batches

class CoherencyPairDataSet(Dataset):
    def __init__(self, filename, args):
        super(CoherencyPairDataSet, self).__init__()
        assert os.path.isfile(filename), "could not find dataset file: {}".format(filename)

        self.max_dialogue_len = 0

        self.coh_ixs = []
        self.acts1 = []
        self.acts2 = []
        self.utts1 = []
        self.utts2 = []
        self.word2id = None

        if args.embedding == 'glove':
            self.word2id = load_vocab(args).stoi
        elif args.embedding == 'elmo':
            self.word2id = None
            self.vocab = load_vocab(args).stoi.keys()
        else:
            assert False, "wrong or not supported embedding"

        if args.model == 'cosine':
            self.stop = get_stopwords(args)
        else:
            self.stop = None

        with open(filename, 'r') as f:
            coh_df = pd.read_csv(f, sep='|', names=['coh_idx', 'acts1', 'utts1', 'acts2', 'utts2'])

        for (idx, row) in tqdm(coh_df.iterrows(), desc="Data Loading"):
            if args.test and  idx >  test_amount:
                break

            acts1 = [int(x) for x in row['acts1'].split(' ')]
            self.acts1.append(acts1)
            acts2 = [int(x) for x in row['acts2'].split(' ')]
            self.acts2.append(acts2)

            utt1 = literal_eval(row['utts1'])
            if self.stop:
                utt1 = [[i for i in sent if i not in self.stop] for sent in utt1]
            utt1 = [sent+["<eos>"] for sent in utt1]
            if args.embedding == 'glove':
                utt1 = [[self.word2id[w] for w in sent] for sent in utt1]
            elif args.embedding == 'elmo':
                utt1 = [["<S>"] + sent + ["</S>"] for sent in utt1]

            utt2 = literal_eval(row['utts2'])
            if self.stop:
                utt2 = [[i for i in sent if i not in self.stop] for sent in utt2]
            utt2 = [sent+["<eos>"] for sent in utt2]
            if args.embedding == 'glove':
                utt2 = [[self.word2id[w] for w in sent] for sent in utt2]
            elif args.embedding == 'elmo':
                utt2 = [["<S>"] + sent + ["</S>"] for sent in utt2]

            self.utts1.append(utt1)
            self.utts2.append(utt2)

            self.coh_ixs.append(int(row['coh_idx']))

    def __len__(self):
        return len(self.acts1)

    def __getitem__(self, idx):
        utt1 = self.utts1[idx]
        utt2 = self.utts2[idx]
        return (utt1, utt2), (self.coh_ixs[idx], (self.acts1[idx], self.acts2[idx]))

    def get_dialog_len(self, idx):
        return len(self.acts[idx])

def get_dataloader(filename, args):
    dset = CoherencyPairDataSet(filename, args)
    batch_size = args.batch_size

    def _collate_glove(samples):
        # get max_seq_len and max_utt_len
        max_seq_len, max_utt_len = 0, 0
        for sample in samples:
            (utt1, utt2), (coh_ix, (acts1, acts2)) = sample
            max_utt_len = max(max_utt_len, len(acts1), len(acts2))
            for (u1,u2) in zip(utt1, utt2):
                max_seq_len = max(max_seq_len, len(u1), len(u2))

        # create padded batch
        utts_left, utts_right, coh_ixs, acts_left, acts_right = [], [], [], [], []
        sent_len_left, sent_len_right, dial_len_left, dial_len_right = [], [], [], []
        pad_id = dset.word2id["<pad>"]
        eos_id = dset.word2id["<eos>"]

        for sample in samples:
            (utt1, utt2), (coh_ix, (acts1, acts2)) = sample

            sent_len_left.append([len(u) for u in utt1] + [1]*(max_utt_len-len(utt1)))
            sent_len_right.append([len(u) for u in utt2] + [1]*(max_utt_len-len(utt2)))
            dial_len_left.append(len(utt1))
            dial_len_right.append(len(utt2))

            utt1 = [ u + [pad_id]*(max_seq_len-len(u)) for u in utt1]
            utt1 = utt1 + [[eos_id]+[pad_id]*(max_seq_len-1)]*(max_utt_len-len(utt1))
            utts_left.append(utt1)
            utt2 = [ u + [pad_id]*(max_seq_len-len(u)) for u in utt2]
            utt2 = utt2 + [[eos_id]+[pad_id]*(max_seq_len-1)]*(max_utt_len-len(utt2))
            utts_right.append(utt2)

            acts1 = acts1 + [0]*(max_utt_len-len(acts1))
            acts_left.append(acts1)
            acts2 = acts2 + [0]*(max_utt_len-len(acts2))
            acts_right.append(acts2)
            coh_ixs.append(coh_ix)
        return ((torch.tensor(utts_left, dtype=torch.long), torch.tensor(utts_right, dtype=torch.long)),
                (torch.tensor(coh_ixs, dtype=torch.float), (torch.tensor(acts_left, dtype=torch.long), torch.tensor(acts_right, dtype=torch.long))),
                (torch.tensor(sent_len_left, dtype=torch.long), torch.tensor(sent_len_right, dtype=torch.long),
                 torch.tensor(dial_len_left, dtype=torch.long), torch.tensor(dial_len_right, dtype=torch.long)))

    def _collate_elmo(samples):
        # get max_seq_len and max_utt_len
        max_seq_len, max_utt_len = 0, 0
        for sample in samples:
            (utt1, utt2), (coh_ix, (acts1, acts2)) = sample
            max_utt_len = max(max_utt_len, len(acts1), len(acts2))
            for (u1,u2) in zip(utt1, utt2):
                max_seq_len = max(max_seq_len, len(u1), len(u2))

        # create padded batch
        utts_left, utts_right, coh_ixs, acts_left, acts_right = [], [], [], [], []
        sent_len_left, sent_len_right, dial_len_left, dial_len_right = [], [], [], []
        pad = "<pad>"
        eos = "<eos>"

        for sample in samples:
            (utt1, utt2), (coh_ix, (acts1, acts2)) = sample

            sent_len_left.append([len(u) for u in utt1] + [1]*(max_utt_len-len(utt1)))
            sent_len_right.append([len(u) for u in utt2] + [1]*(max_utt_len-len(utt2)))
            dial_len_left.append(len(utt1))
            dial_len_right.append(len(utt2))

            acts1 = acts1 + [0]*(max_utt_len-len(acts1))
            acts_left.append(acts1)
            acts2 = acts2 + [0]*(max_utt_len-len(acts2))
            acts_right.append(acts2)
            coh_ixs.append(coh_ix)

            utt1 = [ u + [pad]*(max_seq_len-len(u)) for u in utt1]
            utt1 = utt1 + [[eos]+[pad]*(max_seq_len-1)]*(max_utt_len-len(utt1))
            utt1 = batch_to_ids(utt1)
            utts_left.append(utt1)
            utt2 = [ u + [pad]*(max_seq_len-len(u)) for u in utt2]
            utt2 = utt2 + [[eos]+[pad]*(max_seq_len-1)]*(max_utt_len-len(utt2))
            utt2 = batch_to_ids(utt2)
            utts_right.append(utt2)

            # print("size utt1: {} ; size utt2: {}".format(utt1.size(), utt2.size()))

        return ((torch.stack(utts_left, 0), torch.stack(utts_right, 0)),
                (torch.tensor(coh_ixs, dtype=torch.float), (torch.tensor(acts_left, dtype=torch.long), torch.tensor(acts_right, dtype=torch.long))),
                (torch.tensor(sent_len_left, dtype=torch.long), torch.tensor(sent_len_right, dtype=torch.long),
                 torch.tensor(dial_len_left, dtype=torch.long), torch.tensor(dial_len_right, dtype=torch.long)))

    def _collate_bert(samples):
        # get max_seq_len and max_utt_len
        max_seq_len, max_utt_len = 0, 0
        for sample in samples:
            (utt1, utt2), (coh_ix, (acts1, acts2)) = sample
            max_utt_len = max(max_utt_len, len(acts1), len(acts2))
            for (u1,u2) in zip(utt1, utt2):
                max_seq_len = max(max_seq_len, len(u1), len(u2))

        utts_left, utts_right, coh_ixs, acts_left, acts_right = [], [], [], [], []
        sent_len_left, sent_len_right, dial_len_left, dial_len_right = [], [], [], []

        for sample in samples:
            (utt1, utt2), (coh_ix, (acts1, acts2)) = sample

            sent_len_left.append([len(u) for u in utt1] + [1]*(max_utt_len-len(utt1)))
            sent_len_right.append([len(u) for u in utt2] + [1]*(max_utt_len-len(utt2)))
            dial_len_left.append(len(utt1))
            dial_len_right.append(len(utt2))

            acts1 = acts1 + [0]*(max_utt_len-len(acts1))
            acts_left.append(acts1)
            acts2 = acts2 + [0]*(max_utt_len-len(acts2))
            acts_right.append(acts2)
            coh_ixs.append(coh_ix)

            utt1 = [[" ".join(u) for sent in utt] for utt in utt1]
            utt1 = utt1 + [" "]*(max_utt_len-len(utt1))
            utts_left.append(utt1)
            utt2 = [[" ".join(u) for sent in utt] for utt in utt2]
            utt2 = utt2 + [" "]*(max_utt_len-len(utt2))
            utts_right.append(utt2)
        
        return ((utts_left, utts_right),
                (torch.tensor(coh_ixs, dtype=torch.float), (torch.tensor(acts_left, dtype=torch.long), torch.tensor(acts_right, dtype=torch.long))),
                (torch.tensor(sent_len_left, dtype=torch.long), torch.tensor(sent_len_right, dtype=torch.long),
                 torch.tensor(dial_len_left, dtype=torch.long), torch.tensor(dial_len_right, dtype=torch.long)))

    if args.embedding == 'glove':
        dload = DataLoader(dset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=_collate_glove)
    if args.embedding == 'elmo':
        dload = DataLoader(dset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=_collate_elmo)
    if args.embedding == 'bert':
        dload = DataLoader(dset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=_collate_bert)
    return dload

def load_vocab(args):
    f = open(os.path.join(args.datadir, "itos.txt"), "r")
    cnt = Counter()
    for i, word in enumerate(f):
        cnt[word[:-1].lower()] = 1

    return tt.vocab.Vocab(cnt, specials=['<pad>','<eos>','<unk>'])

def get_stopword_ids(args):
    words = set(stopwords.words('english'))
    vocab = load_vocab(args)
    return [vocab.stoi[w] for w in words]

def get_stopwords(args):
    exclude_f = os.path.join(args.datadir, "words2exclude.txt")
    words2exclude = []
    if os.path.isfile(exclude_f):
        print("loading excluded words")
        with open(exclude_f) as f:
            for line in f:
                words2exclude.append(line[:-1])

    return set(stopwords.words('english') + words2exclude)

# for testing purpose
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",
                        required=True,
                        type=str,
                        help="""The input directory where the files of daily
                        dialog are located. the folder should have
                        train/test/validation as subfolders""")
    parser.add_argument('--task',
                        required=True,
                        type=str,
                        default="up",
                        help="""for which task the dataset should be created.
                                alternatives: up (utterance permutation)
                                              us (utterance sampling)
                                              hup (half utterance petrurbation) """)
    parser.add_argument('--batch_size',
                        required=False,
                        type=int,
                        default=16)
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
    args = parser.parse_args()

    data_file = os.path.join(args.datadir, "coherency_dset_{}.txt".format(args.task))
    assert os.path.isfile(data_file), "could not find dataset file: {}".format(data_file)

    # dset = CoherencyPairDataSet(data_file, args)
    # batches = get_batches(data_file, args.batch_size)
    dload = get_dataloader(data_file, args)
    # emb = GloveEmbedding(args)


    for i, ((ul,ur),(c,(al,ar)),(l1,l2,l3,l4)) in enumerate(dload):
        print(ul.size())
        print(ur.size())

        if i > 1: break
