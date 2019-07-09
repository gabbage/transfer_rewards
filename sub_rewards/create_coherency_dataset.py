import os
from copy import deepcopy
import pandas as pd
from math import factorial
import random
from collections import Counter, defaultdict
import sys
from nltk import word_tokenize
from tqdm import tqdm, trange
import argparse
import numpy as np
import re
from sklearn.model_selection import train_test_split

from swda.swda import CorpusReader, Transcript, Utterance
#from pytorch_pretrained_bert.tokenization import BertTokenizer
#from allennlp.modules.elmo import batch_to_ids

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


def half_perturb_switchboard(sents, sent_DAs, amount, speaker_ixs):
    assert len(sents) == len(sent_DAs), "length of permuted sentences and list of DAs must be equal"
    
    if amount == 0 or len(sents) < 4:
        return []

    permutations = []
    for _ in range(amount):
        speaker = random.randint(0,1) # choose one of the speakers
        speaker_ix = list(filter(lambda x: speaker_ixs[x] == speaker, range(len(sents))))
        #TODO: rename either speaker_ix or speaker_ixs, they are something different, but the names are too close
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
                    of.write("{}|{}|{}|{}|{}\n".format("0",a,u,pa,p_u))
                    of.write("{}|{}|{}|{}|{}\n".format("1",pa,p_u,a,u))

            else:
                for p in permuted_ixs:
                    a = " ".join([str(a) for a in acts])
                    u = str(tok_seqs)
                    pa = [acts[i] for i in p]
                    p_a = " ".join([str(a) for a in pa])
                    pu = [tok_seqs[i] for i in p]
                    p_u = str(pu)
                    of.write("{}|{}|{}|{}|{}\n".format("0",a,u,p_a,p_u))
                    of.write("{}|{}|{}|{}|{}\n".format("1",p_a,p_u,a,u))

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

class SwitchboardConverter:
    def __init__(self, data_dir, tokenizer, word2id, task='', seed=42):
        self.corpus = CorpusReader(data_dir)
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.word2id = word2id
        self.task = task

        self.utt_num = 0
        for utt in self.corpus.iter_utterances():
            self.utt_num += 1

        self.trans_num = 0
        for trans in self.corpus.iter_transcripts():
            self.trans_num += 1

        self.da2num = switchboard_da_mapping()
        
        # CAUTION: make sure that for each task the seed is the same s.t. the splits will be the same!
        train_ixs, val_ixs = train_test_split(range(self.trans_num), shuffle=True, train_size=0.8, random_state=seed)
        val_ixs, test_ixs = train_test_split(val_ixs, shuffle=True, train_size=0.5, random_state=seed)
        self.train_ixs, self.val_ixs, self.test_ixs = train_ixs, val_ixs, test_ixs

        self.utt_da_pairs = []
        prev_da = "%"
        for i, utt in enumerate(self.corpus.iter_utterances()):
            if i == r:
                sentence = re.sub(r"([+/\}\[\]]|\{\w)", "",
                                utt.text)
                act = utt.damsl_act_tag()
                if act == None: act = "%"
                if act == "+": act = prev_da

            self.utt_da_pair.append((sentence, act))

    def draw_rand_sent(self):
        #TODO: redo this function. this approach takes too long. maybe load sentences into a list once and then just random pick...
        r = random.randint(0, len(self.utt_da_pairs)-1)
        return self.utt_da_pairs[r]

    def create_vocab(self):
        print("Creating Vocab file for Switchboard")

        cnt = Counter()
        for utt in self.corpus.iter_utterances():
            sentence = re.sub(r"([+/\}\[\]]|\{\w)", "",
                            utt.text)
            sentence = self.tokenizer(sentence)
            for w in sentence:
                cnt[w] += 1

        itos_file = os.path.join(self.data_dir, "itos.txt")
        itosf = open(itos_file, "w")

        for (word, _) in cnt.most_common(25000):
            itosf.write("{}\n".format(word))

    def convert_dset(self, amounts):
        # create distinct train/validation/test files. they'll correspond to the created
        # splits from the constructor
        train_output_file = os.path.join(self.data_dir, 'train', 'coherency_dset_{}.txt'.format(self.task))
        val_output_file = os.path.join(self.data_dir, 'validation', 'coherency_dset_{}.txt'.format(self.task))
        test_output_file = os.path.join(self.data_dir, 'test', 'coherency_dset_{}.txt'.format(self.task))
        if not os.path.exists(os.path.join(self.data_dir, 'train')):
            os.makedirs(os.path.join(self.data_dir, 'train'))
        if not os.path.exists(os.path.join(self.data_dir, 'validation')):
            os.makedirs(os.path.join(self.data_dir, 'validation'))
        if not os.path.exists(os.path.join(self.data_dir, 'test')):
            os.makedirs(os.path.join(self.data_dir, 'test'))

        trainfile = open(train_output_file, 'w')
        valfile = open(val_output_file, 'w')
        testfile = open(test_output_file, 'w')

        for i,trans in enumerate(tqdm(self.corpus.iter_transcripts(display_progress=False))):
            if i > 20:
                break

            utterances = []
            acts = []
            speaker_ixs = []
            prev_act = "%"
            for utt in trans.utterances:
                sentence = re.sub(r"([+/\}\[\]]|\{\w)", "",
                                utt.text)
                # print(sentence, " ## DAs: ", utt.act_tag)
                sentence = self.word2id(self.tokenizer(sentence))
                utterances.append(sentence)
                act = utt.damsl_act_tag()
                if act == None: act = "%"
                if act == "+": act = prev_act
                acts.append(self.da2num[act])
                prev_act = act
                if "A" in utt.caller:
                    speaker_ixs.append(0)
                else:
                    speaker_ixs.append(1)

            if self.task == 'up':
                permuted_ixs = permute(utterances, acts, amounts)
            elif self.task == 'us':
                l = self.utt_num
                permuted_ixs = draw_rand_sent(l, len(utterances)-1, amounts) #TODO: write a Switchboard specific draw function
            elif self.task == 'hup':
                permuted_ixs = half_perturb_switchboard(utterances, acts, amounts, speaker_ixs)
            elif self.task == 'ui':
                permuted_ixs = utterance_insertions(len(utterances), amounts)

            if self.task == 'us':
                for p in permuted_ixs:
                    a = " ".join([str(a) for a in acts])
                    u = str(utterances)
                    insert_sent, insert_da = self.draw_rand_sent()
                    insert_ix = p[1]
                    p_a = deepcopy(acts)
                    p_a[insert_ix] = insert_da
                    pa = " ".join([str(a) for a in p_a])
                    p_u = deepcopy(utterances)
                    p_u[insert_ix] = self.word2id([w.lower() for w in self.tokenizer(insert_sent)])

                    if i in self.train_ixs:
                        trainfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,pa,p_u))
                        trainfile.write("{}|{}|{}|{}|{}\n".format("1",pa,p_u,a,u))
                    if i in self.val_ixs:
                        valfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,pa,p_u))
                        valfile.write("{}|{}|{}|{}|{}\n".format("1",pa,p_u,a,u))
                    if i in self.test_ixs:
                        testfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,pa,p_u))
                        testfile.write("{}|{}|{}|{}|{}\n".format("1",pa,p_u,a,u))

            else:
                for p in permuted_ixs:
                    a = " ".join([str(a) for a in acts])
                    u = str(utterances)
                    pa = [acts[i] for i in p]
                    p_a = " ".join([str(a) for a in pa])
                    pu = [utterances[i] for i in p]
                    p_u = str(pu)

                    if i in self.train_ixs:
                        trainfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,p_a,p_u))
                        trainfile.write("{}|{}|{}|{}|{}\n".format("1",p_a,p_u,a,u))
                    if i in self.val_ixs:
                        valfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,p_a,p_u))
                        valfile.write("{}|{}|{}|{}|{}\n".format("1",p_a,p_u,a,u))
                    if i in self.test_ixs:
                        testfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,p_a,p_u))
                        testfile.write("{}|{}|{}|{}|{}\n".format("1",p_a,p_u,a,u))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",
                        required=True,
                        type=str,
                        help="""The input directory where the files of the corpus
                        are located. """)
    parser.add_argument("--corpus",
                        required=True,
                        type=str,
                        help="""the name of the corpus to use, currently either 'DailyDialog' or 'Switchboard' """)
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--amount',
                        type=int,
                        default=20,
                        help="random seed for initialization")
    parser.add_argument('--word2id',
                        action='store_true',
                        help= "convert the words to ids")
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


    if args.word2id:
        f = open(os.path.join(args.datadir, "itos.txt"), "r")
        word2id_dict = dict()
        for i, word in enumerate(f):
            word2id_dict[word[:-1].lower()] = i

        word2id = lambda x: [word2id_dict[y] for y in x] # don't convert words to ids (yet). It gets done in the glove wrapper of mtl_coherence.py
    else:
        word2id = lambda x: x

    tokenizer = word_tokenize
    if args.corpus == 'DailyDialog':
        converter = DailyDialogConverter(args.datadir, tokenizer, word2id, task=args.task)
    elif args.corpus == 'Switchboard':
        converter = SwitchboardConverter(args.datadir, tokenizer, word2id, args.task, args.seed)
        converter.create_vocab()

    converter.convert_dset(amounts=args.amount)
    # converter.call_shuf_on_output()
    # print("Amount of pertubations for task {} is: {}".format(args.task, converter.perturbation_statistics))


def switchboard_da_mapping():
    mapping_dict = dict({
                "sd": 1,
                "b": 2,
                "sv": 3,
                "aa": 4,
                "%-": 5,
                "ba": 6,
                "qy": 7,
                "x": 8,
                "ny": 9,
                "fc": 10,
                "%": 11,
                "qw": 12,
                "nn": 13,
                "bk": 14,
                "h": 15,
                "qy^d": 16,
                "o": 17,
                "bh": 18,
                "^q": 19,
                "bf": 20,
                "na": 21,
                "ny^e": 22,
                "ad": 23,
                "^2": 24,
                "b^m": 25,
                "qo": 26,
                "qh": 27,
                "^h": 28,
                "ar": 29,
                "ng": 30,
                "nn^e": 31,
                "br": 32,
                "no": 33,
                "fp": 34,
                "qrr": 35,
                "arp": 36,
                "nd": 37,
                "t3": 38,
                "oo": 39,
                "co": 40,
                "cc": 41,
                "t1": 42,
                "bd": 43,
                "aap": 44,
                "am": 45,
                "^g": 46,
                "qw^d": 47,
                "fa": 48,
                "ft":49 
            })
    d = defaultdict(lambda: 11)
    for (k, v) in mapping_dict.items():
        d[k] = v
    return d

if __name__ == "__main__":
    main()
