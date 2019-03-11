import gzip
import logging
import os
import shutil
from collections import Counter

import requests
import torch
from gensim.models import KeyedVectors
from nltk import word_tokenize
from torchtext.vocab import Vectors
from tqdm import tqdm

from scorer.data_helper.json_reader import readArticleRefs


def download_file_from_google_drive(file_id, dest):
    base_url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(base_url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(base_url, params=params, stream=True)

    if response.status_code == 200:
        save_response_content(response, dest)
    else:
        print(response)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, dest):
    chunk_size = 32768

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


class GoogleNews(Vectors):
    def __init__(self, cache=None, unk_init=None, **kwargs):
        cache = '.vector_cache' if cache is None else cache
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = 300
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        self._id = '0B7XkCwpI5KDYNlNUTTlSS21pQmM'
        self._name = 'gnews'
        self._bin_file = os.path.join(cache, '{}.{}d.bin'.format(self._name, self.dim))
        self._dl_file = os.path.join(cache, '{}.{}d.bin.gz'.format(self._name, self.dim))

        if not os.path.isfile(self._dl_file):
            logging.info("Downloading GoogleNews vectors to {} might take a while!".format(self._dl_file))
            download_file_from_google_drive(self._id, self._dl_file)

        if not os.path.isfile(self._bin_file) and os.path.isfile(self._dl_file):
            logging.info("Extracting GoogleNews gz archive to {}".format(self._bin_file))
            with gzip.open(self._dl_file, 'rb') as f_in:
                with open(self._bin_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        if os.path.isfile(self._bin_file):
            logging.info("Loading vectors from {}".format(self._bin_file))
            gensim_vectors = KeyedVectors.load_word2vec_format(self._bin_file, binary=True)

            self.vectors = torch.Tensor(gensim_vectors.vectors).view(-1, self.dim)
            self.itos = gensim_vectors.index2word
            self.stoi = {}

            for k, v in gensim_vectors.vocab.items():
                self.stoi.update({k: v.index})

            print(len(self.itos))
            print(len(self.stoi.keys()))


def count_oovs(all_words, vectors):
    oov_words = Counter()

    for word in all_words:
        if word not in vectors.stoi:
            oov_words.update([word])

    return len(oov_words)


if __name__ == "__main__":
    vectors_dict = {#"glove_6b": GloVe('6B'),
                    #"glove_27B": GloVe('twitter.27B', 200),
                    #"glove_42B": GloVe('42B'),
                    #"glove_840B": GloVe('840B'),
                    #"fasttext": FastText(),
                    #"gnews": GoogleNews(),
                    }

    # Count distinct words like they are and as lower case
    word_counter = Counter()
    word_counter_lower = Counter()

    for entry in tqdm(readArticleRefs(), desc="Preparing counters for OOV statistics"):
        word_counter.update(word_tokenize(entry["article"]))
        word_counter_lower.update(word_tokenize(entry["article"].lower()))

    print("Distinct words: orig={}, lower={}".format(len(word_counter), len(word_counter_lower)))

    for vectors_name, vectors in vectors_dict.items():
        oov = count_oovs(word_counter, vectors)
        oov_lower = count_oovs(word_counter_lower, vectors)
        per_oov = oov / len(word_counter) * 100
        per_oov_lower = oov_lower / len(word_counter_lower) * 100

        print("OOV with {}: orig={} ({:.2f}%), lower={} ({:.2f}%)".format(vectors_name, oov, per_oov,
                                                                          oov_lower, per_oov_lower))
