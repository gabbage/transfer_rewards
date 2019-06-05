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
        else:
            assert False, "wrong or not supported embedding"

        if args.model == 'cosine':
            self.stop = get_stopword_ids(args)
        else:
            self.stop = None

        with open(filename, 'r') as f:
            coh_df = pd.read_csv(f, sep='|', names=['coh_idx', 'acts1', 'utts1', 'acts2', 'utts2'])

        for (idx, row) in tqdm(coh_df.iterrows(), desc="Data Loading"):
            acts1 = [int(x) for x in row['acts1'].split(' ')]
            self.acts1.append(acts1)
            acts2 = [int(x) for x in row['acts2'].split(' ')]
            self.acts2.append(acts2)

            utt1 = literal_eval(row['utts1'])
            utt1 = [[self.word2id[w] for w in sent] for sent in utt1]
            if self.stop:
                utt1 = [[i for i in sent if i not in self.stop] for sent in utt1]
            utt2 = literal_eval(row['utts2'])
            utt2 = [[self.word2id[w] for w in sent] for sent in utt2]
            if self.stop:
                utt2 = [[i for i in sent if i not in self.stop] for sent in utt2]

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

    def _collate(samples):
        # get max_seq_len and max_utt_len
        max_seq_len, max_utt_len = 0, 0
        for sample in samples:
            (utt1, utt2), (coh_ix, (acts1, acts2)) = sample
            max_utt_len = max(max_utt_len, len(acts1), len(acts2))
            for (u1,u2) in zip(utt1, utt2):
                max_seq_len = max(max_seq_len, len(u1), len(u2))

        # create padded batch
        utts_left, utts_right, coh_ixs, acts_left, acts_right = [], [], [], [], []
        lengths_left, lengths_right = [], []
        pad_id = dset.word2id["<pad>"]

        for sample in samples:
            (utt1, utt2), (coh_ix, (acts1, acts2)) = sample

            lengths_left.append([len(u) for u in utt1] + [1]*(max_utt_len-len(utt1)))
            lengths_right.append([len(u) for u in utt2] + [1]*(max_utt_len-len(utt2)))

            utt1 = [ u + [pad_id]*(max_seq_len-len(u)) for u in utt1]
            utt1 = utt1 + [[pad_id]*max_seq_len]*(max_utt_len-len(utt1))
            utts_left.append(utt1)
            utt2 = [ u + [pad_id]*(max_seq_len-len(u)) for u in utt2]
            utt2 = utt2 + [[pad_id]*max_seq_len]*(max_utt_len-len(utt2))
            utts_right.append(utt2)
            acts1 = acts1 + [0]*(max_utt_len-len(acts1))
            acts_left.append(acts1)
            acts2 = acts2 + [0]*(max_utt_len-len(acts2))
            acts_right.append(acts2)
            coh_ixs.append(coh_ix)
        return ((torch.tensor(utts_left, dtype=torch.long), torch.tensor(utts_right, dtype=torch.long)),
                (torch.tensor(coh_ixs, dtype=torch.float), (torch.tensor(acts_left, dtype=torch.long), torch.tensor(acts_right, dtype=torch.long))),
                (torch.tensor(lengths_left, dtype=torch.long), torch.tensor(lengths_right, dtype=torch.long)))

    dload = DataLoader(dset, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=_collate)
    return dload

def load_vocab(args):
    f = open(os.path.join(args.datadir, "itos.txt"), "r")
    cnt = Counter()
    for i, word in enumerate(f):
        cnt[word[:-1].lower()] = i

    return tt.vocab.Vocab(cnt)

def get_stopword_ids(args):
    words = set(stopwords.words('english'))
    vocab = load_vocab(args)
    return [vocab.stoi[w] for w in words]

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
    args = parser.parse_args()

    data_file = os.path.join(args.datadir, "coherency_dset_{}.txt".format(args.task))
    assert os.path.isfile(data_file), "could not find dataset file: {}".format(data_file)

    # batches = get_batches(data_file, args.batch_size)
    dload = get_dataloader(data_file, args.batch_size)
    emb = GloveEmbedding(args)


    for i, ((ul,ur),(c,(al,ar))) in enumerate(dload):
        print(al[0])
        print(ar[0])

        if i > 1: break
