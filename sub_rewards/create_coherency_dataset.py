import os
from copy import deepcopy
import pandas as pd
from math import factorial
import random
import sys
from nltk import word_tokenize
from tqdm import tqdm, trange
import argparse
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer

act2word = {1:"inform",2:"question", 3:"directive", 4:"commissive"}
BERT_MODEL_NAME = "bert-large-uncased"
# how many sentence permutations and random inserts should be created per coherent dialog
PERMUTATIONS_PER_DIALOG = 2
RANDINSERTS_PER_DIALOG = 2

def permute(sents, sent_DAs, amount):
    """ return a list of different! permuted sentences and their respective dialog acts """
    """ if amount is greater than the possible amount of permutations, only the uniquely possible ones are returned """
    assert len(sents) == len(sent_DAs), "length of permuted sentences and list of DAs must be equal"

    permuted_sents, permuted_DAs = [], []
    previous = [list(range(len(sents)))]
    amount = min(amount, factorial(len(sents))-1)
    for i in range(amount):
        permutation = np.random.permutation(len(sents))
        while permutation.tolist() in previous:
            permutation = np.random.permutation(len(sents))

        previous.append(permutation.tolist())
        permuted_sents.append([sents[i] for i in permutation])
        permuted_DAs.append([sent_DAs[i] for i in permutation])
    return permuted_sents, permuted_DAs

def draw_rand_sent_from_df(df, tokenizer, word2id):
    """ df is supposed to be a pandas dataframe with colums 'act' and 'utt' (utterance), 
        with act being a number from 1 to 4 and utt being a sentence """

    pos = random.randint(0, len(df))
    return int(df['act'][pos]), word2id(tokenizer(df['utt'][pos]))


def random_insert(sents, sent_DAs, generator, amount):
    assert len(sents) == len(sent_DAs), "length of permuted sentences and list of DAs must be equal"

    random_sents, random_DAs = [], []
    for _ in range(amount):
        rDA, rsent = generator()
        pos = random.randint(0, len(sents)-1)
        random_sents.append(deepcopy(sents))
        random_DAs.append(deepcopy(sent_DAs))
        random_sents[-1][pos] = rsent
        random_DAs[-1][pos] = rDA
    return random_sents, random_DAs


class DailyDialogConverter:
    def __init__(self, data_dir, tokenizer, word2id):
        self.data_dir = data_dir
        self.act_utt_file = os.path.join(data_dir, 'act_utt.txt')

        self.tokenizer = tokenizer
        self.word2id = word2id

    def convert_dset(self):
        # data_dir is supposed to be the dir with the respective train/test/val-dataset files
        dial_file = os.path.join(self.data_dir, 'dialogues.txt')
        act_file = os.path.join(self.data_dir, 'dialogues_act.txt')
        output_file = os.path.join(self.data_dir, 'coherency_dset.txt')

        assert os.path.isfile(dial_file) and os.path.isfile(act_file), "could not find input files"
        assert os.path.isfile(self.act_utt_file), "missing act_utt.txt in data_dir"

        with open(self.act_utt_file, 'r') as f:
            act_utt_df = pd.read_csv(f, sep='|', names=['act','utt'])
        
        rand_generator = lambda: draw_rand_sent_from_df(act_utt_df, self.tokenizer, self.word2id)

        df = open(dial_file, 'r')
        af = open(act_file, 'r')
        of = open(output_file, 'w')

        for line_count, (dial, act) in tqdm(enumerate(zip(df, af)), total=11118):
            seqs = dial.split('__eou__')
            seqs = seqs[:-1]
            tok_seqs = [self.word2id(self.tokenizer(seq)) for seq in seqs]

            acts = act.split(' ')
            acts = acts[:-1]
            acts = [int(act) for act in acts]
            
            permuted_sents, permuted_DAs = permute(tok_seqs, acts, PERMUTATIONS_PER_DIALOG)
            random_sents, random_DAs = random_insert(tok_seqs, acts, rand_generator, RANDINSERTS_PER_DIALOG)

            sent_data = [tok_seqs]+permuted_sents+random_sents
            act_data = [acts]+permuted_DAs+random_DAs
            coh_data = [1.0] + [0.0]*(len(sent_data)-1)

            for i in np.random.permutation(len(sent_data)):
                of.write("{}|{}|{}\n".format(str(coh_data[i]), " ".join([str(a) for a in act_data[i]]), " ".join([str(s) for s in sent_data[i]])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",
                        # required=True,
                        type=str,
                        help="""The input directory where the files of daily
                        dialog are located. the folder should have
                        train/test/validation as subfolders""")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    bert_tok = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=True)
    word2id = lambda x: bert_tok.convert_tokens_to_ids(x)
    tokenizer = lambda x:bert_tok.tokenize(x)

    converter = DailyDialogConverter(args.datadir, tokenizer, word2id)
    converter.convert_dset()

    ### Test
    # act_utt_file = os.path.join(args.datadir, 'act_utt.txt')
    # with open(act_utt_file, 'r') as f:
        # act_utt_df = pd.read_csv(f, sep='|', names=['act','utt'])

    # sents = [["hello", "word"], ["whats", "up"]]
    # a = [[1,2,3],[4,5,6],[7,8,9]]
    # b = ['a', 'b', 'c']
    # print(permute(a,b,9))

    ###

if __name__ == "__main__":
    main()
