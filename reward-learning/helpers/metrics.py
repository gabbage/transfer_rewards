from collections import OrderedDict

import numpy as np
import scipy.stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from helpers.data_helpers import extract_ngrams2, sent2stokens_wostop


def jsd(p, q, base=np.e):
    # Convert to np.array
    p, q = np.asarray(p), np.asarray(q)

    # Normalize p, q to probabilities
    if p.sum() == 0 or q.sum() == 0:
        return -1.0

    p, q = p / p.sum(), q / q.sum()
    m = 1.0 / 2 * (p + q)

    return scipy.stats.entropy(p, m, base=base) / 2.0 + scipy.stats.entropy(q, m, base=base) / 2.0


def redundancy(text, n, stemmer, language='english'):
    summ_ngram = list(extract_ngrams2(text, stemmer, language, n))

    if len(summ_ngram) == 0:
        return 0.0
    else:
        return 1.0 - len(set(summ_ngram)) / float(len(summ_ngram))


def tfidf(article, summary, reductions, language='english'):
    corpus = [article, summary]
    vv = TfidfVectorizer(stop_words=language)
    vectors = vv.fit_transform(corpus)
    rewards = []

    if 'cos' in reductions:
        cos = cosine_similarity(vectors[0, :], vectors[1, :])[0][0]
        rewards.append(cos)

    if 'avg' in reductions:
        avg_value = np.mean(vectors[1])
        rewards.append(avg_value)

    return rewards


def word_distribution(text, vocab, n_list, stemmer, stopwords=None, language='english'):
    if vocab is None:
        vocab_list = []
        build_vocab = True
    else:
        vocab_list = vocab
        build_vocab = False

    word_dist = OrderedDict((el, 0) for el in vocab_list)
    n_grams = []

    for n in n_list:
        if n == 1:
            n_grams.extend(sent2stokens_wostop(text, stemmer, stopwords, language))
        else:
            n_grams.extend(extract_ngrams2([text], stemmer, language, n))

    for n_gram in n_grams:
        if n_gram in word_dist:
            word_dist[n_gram] = word_dist[n_gram] + 1
        elif build_vocab:
            word_dist[n_gram] = 1

    return list(word_dist.keys()), list(word_dist.values())


def js(article, summary, stemmer, n_list=None, stopwords=None, language='english'):
    if n_list is None:
        n_list = [1, 2]

    vocab, doc_word_dist = word_distribution(article, None, n_list, stemmer, stopwords, language)
    _, sum_word_dist = word_distribution(summary, vocab, n_list, stemmer, stopwords, language)

    return [jsd(sum_word_dist, doc_word_dist)]


def rouge(article, summary, n, stemmer, stopwords=None, language='english'):
    vocab, doc_word_dist = word_distribution(article, None, [n], stemmer, stopwords, language)
    _, sum_word_dist = word_distribution(summary, vocab, [n], stemmer, stopwords, language)
    rouge_score = np.sum([1 for cc in sum_word_dist if cc is not 0]) * 1.0 / len(sum_word_dist)

    return rouge_score

