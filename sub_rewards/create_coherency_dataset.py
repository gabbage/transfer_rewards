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
import torchtext
from pytorch_pretrained_bert.tokenization import BertTokenizer
from allennlp.modules.elmo import batch_to_ids

act2word = {1:"inform",2:"question", 3:"directive", 4:"commissive"}
BERT_MODEL_NAME = "bert-base-uncased"
# how many sentence permutations and random inserts should be created per coherent dialog
PERMUTATIONS_PER_DIALOG = 1
RANDINSERTS_PER_DIALOG = 0
HALF_PERTURBATIONS_PER_DIALOG = 0
# how many original examples should be included in the dataset
PLAIN_COPIES_PER_DIALOG = 1

def permute(sents, sent_DAs, amount):
    """ return a list of different! permuted sentences and their respective dialog acts """
    """ if amount is greater than the possible amount of permutations, only the uniquely possible ones are returned """
    assert len(sents) == len(sent_DAs), "length of permuted sentences and list of DAs must be equal"

    if amount == 0:
        return []

    permutations = [list(range(len(sents)))]
    amount = min(amount, factorial(len(sents))-1)
    for i in range(amount):
        permutation = np.random.permutation(len(sents))
        while permutation.tolist() in permutations:
            permutation = np.random.permutation(len(sents))

        permutations.append(permutation.tolist())
    return permutations[1:] #the first one is the original, which was included s.t. won't be generated

def draw_rand_sent(dialogues_amount, dialogue_len, amount):
    """ df is supposed to be a pandas dataframe with colums 'act' and 'utt' (utterance), 
        with act being a number from 1 to 4 and utt being a sentence """

    permutations = []
    for _ in range(amount):
        permutations.append(
                [random.randint(0, dialogues_amount-1),
                 random.randint(0, dialogue_len)])
    return permutations

def draw_rand_sent_from_df(df):
    ix = random.randint(0, len(df['utt'])-1)
    return df['utt'][ix], df['act'][ix]

def random_insert(sents, sent_DAs, generator, amount):
    assert len(sents) == len(sent_DAs), "length of permuted sentences and list of DAs must be equal"
    
    if amount == 0:
        return [], []

    random_sents, random_DAs = [], []
    for _ in range(amount):
        rDA, rsent = generator()
        pos = random.randint(0, len(sents)-1)
        random_sents.append(deepcopy(sents))
        random_DAs.append(deepcopy(sent_DAs))
        random_sents[-1][pos] = rsent
        random_DAs[-1][pos] = rDA
    return random_sents, random_DAs

def half_perturb(sents, sent_DAs, amount):
    assert len(sents) == len(sent_DAs), "length of permuted sentences and list of DAs must be equal"
    
    if amount == 0 or len(sents) < 4:
        return []

    permutations = []
    for _ in range(amount):
        speaker = random.randint(0,1) # choose one of the speakers
        speaker_ix = list(filter(lambda x: (x-speaker) % 2 == 0, range(len(sents))))
        if len(speaker_ix) < 2:
            return []

        permuted_speaker_ix = np.random.permutation(speaker_ix)
        while speaker_ix == permuted_speaker_ix.tolist():
            permuted_speaker_ix = np.random.permutation(speaker_ix)
        new_sents = list(range(len(sents)))
        for (i_to, i_from) in zip(speaker_ix, permuted_speaker_ix):
            new_sents[i_to] = i_from

        if not new_sents in permutations:
            permutations.append(new_sents)

    return permutations

def utterance_insertions(length, amount):
    possible_permutations = []
    original = list(range(length))
    for ix in original:
        for y in range(length):
            if ix == y: continue

            ix_removed = original[0:ix] + ([] if ix == length-1 else original[ix+1:])
            ix_removed.insert(y, ix)
            possible_permutations.append(deepcopy(ix_removed))

    permutations = []
    for _ in range(amount):
        i = random.randint(0, len(possible_permutations)-1)
        permutations.append(possible_permutations[i])

    return permutations


#TODO: these functions are for later, when classification is relevant again
def generate_permutations(sents, sent_DAs, permutations):
    # use deepcopy !
    pass
# for a dialogue index and and utterance index, return the sentence from the df
def draw_sent_from_df(df, dialogue_ix, utt_ix): 
    pass

class DailyDialogConverter:
    def __init__(self, data_dir, tokenizer, word2id, task='', ranking_dataset = True):
        self.data_dir = data_dir
        self.act_utt_file = os.path.join(data_dir, 'act_utt.txt')

        self.tokenizer = tokenizer
        self.word2id = word2id
        self.output_file = None
        self.task = task
        self.ranking_dataset = ranking_dataset
        self.perturbation_statistics = 0


    def convert_dset(self, amounts):
        # data_dir is supposed to be the dir with the respective train/test/val-dataset files
        print("Creating {} perturbations for task {}".format(amounts, self.task))

        dial_file = os.path.join(self.data_dir, 'dialogues.txt')
        act_file = os.path.join(self.data_dir, 'dialogues_act.txt')
        self.output_file = os.path.join(self.data_dir, 'coherency_dset_{}.txt'.format(self.task))

        assert os.path.isfile(dial_file) and os.path.isfile(act_file), "could not find input files"
        assert os.path.isfile(self.act_utt_file), "missing act_utt.txt in data_dir"

        with open(self.act_utt_file, 'r') as f:
            act_utt_df = pd.read_csv(f, sep='|', names=['act','utt'])
        
        rand_generator = lambda: draw_rand_sent_from_df(act_utt_df)

        df = open(dial_file, 'r')
        af = open(act_file, 'r')
        of = open(self.output_file, 'w')

        
        for line_count, (dial, act) in tqdm(enumerate(zip(df, af)), total=11118):
            seqs = dial.split('__eou__')
            seqs = seqs[:-1]
            # if len(seqs) > 15:
                # continue # Values above create memory allocation errors with BERT

            tok_seqs = [self.tokenizer(seq) for seq in seqs]
            tok_seqs = [[w.lower() for w in utt] for utt in tok_seqs]
            tok_seqs = [self.word2id(seq) for seq in tok_seqs]

            acts = act.split(' ')
            acts = acts[:-1]
            acts = [int(act) for act in acts]

            if self.task == 'up':
                permuted_ixs = permute(tok_seqs, acts, amounts)
            elif self.task == 'us':
                l = 11118 if self.data_dir.endswith("train") else 1000
                permuted_ixs = draw_rand_sent(l, len(tok_seqs)-1, amounts)
            elif self.task == 'hup':
                permuted_ixs = half_perturb(tok_seqs, acts, amounts)
            elif self.task == 'ui':
                permuted_ixs = utterance_insertions(len(tok_seqs), amounts)

            self.perturbation_statistics += len(permuted_ixs)

            if self.task == 'us':
                for p in permuted_ixs:
                    a = " ".join([str(a) for a in acts])
                    u = str(tok_seqs)
                    insert_sent, insert_da = draw_rand_sent_from_df(act_utt_df)
                    insert_ix = p[1]
                    p_a = deepcopy(acts)
                    p_a[insert_ix] = insert_da
                    pa = " ".join([str(a) for a in p_a])
                    p_u = deepcopy(tok_seqs)
                    p_u[insert_ix] = self.word2id([w.lower() for w in self.tokenizer(insert_sent)])
                    of.write("{}|{}|{}|{}|{}\n".format("1",a,u,pa,p_u))
                    of.write("{}|{}|{}|{}|{}\n".format("0",pa,p_u,a,u))

            else:
                for p in permuted_ixs:
                    a = " ".join([str(a) for a in acts])
                    u = str(tok_seqs)
                    pa = [acts[i] for i in p]
                    p_a = " ".join([str(a) for a in pa])
                    pu = [tok_seqs[i] for i in p]
                    p_u = str(pu)
                    of.write("{}|{}|{}|{}|{}\n".format("1",a,u,p_a,p_u))
                    of.write("{}|{}|{}|{}|{}\n".format("0",p_a,p_u,a,u))

            """ write the original and created datapoints in random order to the file """
            # a = " ".join([str(a) for a in acts])
            # # u = " ".join([str(s) for s in sent_data[i]])
            # u = str(tok_seqs)
            # d = str(permuted_ixs)
            # of.write("{}|{}|{}\n".format(a, u, d))

    def call_shuf_on_output(self):
        """ randomly suffle/permute the output file, s.t. samples drawn from the same input are 
            not next to each other in the output """
        shuf_file = os.path.join(self.data_dir, 'coherency_dset_{}_shuf.txt'.format(self.task))
        cmd = "shuf {} > {}".format(self.output_file, shuf_file)
        os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",
                        required=True,
                        type=str,
                        help="""The input directory where the files of daily
                        dialog are located. """)
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--amount',
                        type=int,
                        default=20,
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
                                              hup (half utterance petrurbation)
                                              ui (utterance insertion, nothing directly added!)""")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    if args.embedding == 'bert':
        # Bert Settings
        bert_tok = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=True)
        word2id = lambda x: x # don't convert words to ids (yet). It gets done in the glove wrapper of mtl_coherence.py
        tokenizer = lambda x:bert_tok.tokenize(x)

    elif args.embedding == 'elmo':
        assert False, "elmo not yet supported"
        #Elmo Settings
        # word2id = lambda x: batch_to_ids([sent])

    elif args.embedding == 'glove':
        tokenizer = word_tokenize
        f = open(os.path.join(args.datadir, "itos.txt"), "r")
        word2id_dict = dict()
        for i, word in enumerate(f):
            word2id_dict[word[:-1].lower()] = i

        # word2id_dict = torchtext.vocab.GloVe(name="42B", dim=300).stoi
        word2id = lambda x: [word2id_dict[y] for y in x] # don't convert words to ids (yet). It gets done in the glove wrapper of mtl_coherence.py

    else:
        assert False, "the --embedding argument could not be detected. either bert, elmo or glove!"

    converter = DailyDialogConverter(args.datadir, tokenizer, word2id, task=args.task)
    converter.convert_dset(amounts=args.amount)
    converter.call_shuf_on_output()
    print("Amount of pertubations for task {} is: {}".format(args.task, converter.perturbation_statistics))

if __name__ == "__main__":
    main()
