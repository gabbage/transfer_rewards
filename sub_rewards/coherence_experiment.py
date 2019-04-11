from collections import Counter
import torch
import torch.nn as nn
import torchtext as tt
from nltk.corpus import stopwords
from nltk import word_tokenize
from torch.nn.modules.distance import CosineSimilarity

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

def sim(v1, v2):
    return v1.dot(v2).abs()

def sent_sim(sent1, sent2):
    max_indices = (0, 0)
    max_sim = 0.0
    for i, t1 in enumerate(sent1):
        for j, t2 in enumerate(sent2):
            s = sim(sent1[i], sent2[j])
            if s > max_sim:
                max_sim = s
                max_indices = (i, j)

    avg = (sent1[max_indices[0]] + sent2[max_indices[1]])*1/2

    return max_indices, avg

def elmo_rep(elmo, sent):
    char_ids = batch_to_ids([sent])
    return elmo(char_ids)['elmo_representations'][0][0]

def preprocess(sent):
    padlen = 50
    stop = set(stopwords.words('english'))
    sent_wo_stop = [w.lower() for w in sent if w.lower() not in stop]
    return sent_wo_stop + ["<pad>"]*(padlen-len(sent_wo_stop))

def main():
    sent1 = word_tokenize("let us use our forks that we just ate spaghetti with to kill tom")
    sent1 = preprocess(sent1)
    sent2 = word_tokenize("no i dont like eating spaghetti with forks")
    sent2 = preprocess(sent2)
    sent3 = word_tokenize("spaghetti was invented in the nineteenth century")
    sent3 = preprocess(sent3)
    ref1 = word_tokenize("Say Jim how about going for a few beers after dinner")
    ref1 = preprocess(ref1)
    ref2 = word_tokenize("You know that is tempting but is really not good for our fitness")
    ref2 = preprocess(ref2)
    ref3 = word_tokenize("What do you mean It will help us to relax")
    ref3 = preprocess(ref3)
    asap1 = word_tokenize("Computers a good because you can get infermation you can play games you can get pictures But when you on the computer you might find something or someone that is bad or is viris")
    asap1 = preprocess(asap1)
    asap2 = word_tokenize("If ther is a vris you might want shut off the computers so it does not get worse")
    asap2 = preprocess(asap2)
    asap3 = word_tokenize("The are websites for kids like games there are teen games there are adult games")
    asap3 = preprocess(asap3)

    # Always only have one of the following (Glove or Elmo) uncommented
    # Use GloVe
    cnt = Counter()
    for w in sent1+sent2+sent3+ref1+ref2+ref3:
        cnt[w.lower()] += 1

    vocab = tt.vocab.Vocab(cnt)
    vocab.load_vectors("glove.42B.300d")
    embed = nn.Embedding(len(vocab), 300)
    embed.weight.data.copy_(vocab.vectors)
    
    embed_fn = lambda x: embed(torch.tensor([vocab.stoi[w] for w in x], dtype=torch.long))

    # Use Elmo
    # elmo = Elmo(options_file, weight_file, 1, dropout=0)
    # embed_fn = lambda x: elmo_rep(elmo, x)

    print("------------ Test Sentences -------------------")
    es1 = embed_fn(sent1)
    es2 = embed_fn(sent2)
    es3 = embed_fn(sent3)
    f1_ix, f1 = sent_sim(es1, es2)
    f2_ix, f2 = sent_sim(es2, es3)
    print("similarity of sentences: ", sim(f1, f2).item() / 300)
    print("words: (", sent1[f1_ix[0]], ",", sent2[f1_ix[1]],") , (", sent2[f2_ix[0]],",",sent3[f2_ix[1]],")")
    print("------------ DailyDialog Sentences -------------")
    rs1 = embed_fn(ref1)
    rs2 = embed_fn(ref2)
    rs3 = embed_fn(ref3)
    f1_ix, f1 = sent_sim(rs1, rs2)
    f2_ix, f2 = sent_sim(rs2, rs3)
    print("similarity of sentences: ", sim(f1, f2).item()/ 300)
    print("words: (", ref1[f1_ix[0]], ",", ref2[f1_ix[1]],") , (", ref2[f2_ix[0]],",",ref3[f2_ix[1]],")")
    print("------------ ASAP Sentences --------------------")
    as1 = embed_fn(asap1)
    as2 = embed_fn(asap2)
    as3 = embed_fn(asap3)
    f1_ix, f1 = sent_sim(as1, as2)
    f2_ix, f2 = sent_sim(as2, as3)
    print("similarity of sentences: ", sim(f1, f2).item()/ 300)
    print("words: (", asap1[f1_ix[0]], ",", asap2[f1_ix[1]],") , (", asap2[f2_ix[0]],",",asap3[f2_ix[1]],")")

    print('--------------Coherence ------------------------')
    cos = CosineSimilarity(dim=0)
    print("test sentences: ", cos(es1.mean(1), es2.mean(1)).item(), ", ",cos(es2.mean(1), es3.mean(1)).item())
    print("daily sentenes: ", cos(rs1.mean(1), rs2.mean(1)).item(), ", ",cos(rs2.mean(1), rs3.mean(1)).item())
    print("asap sentenes: ", cos(as1.mean(1), as2.mean(1)).item(),  ", ",cos(as2.mean(1), as3.mean(1)).item())

if __name__ == '__main__':
    main()
