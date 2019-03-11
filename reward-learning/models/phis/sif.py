import argparse
import logging
from collections import Counter
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD

from models.phis.encoders import Encoding, DimConstraint
from models.phis.phi_api import PhiAPI

"""
Smooth inverse frequency based on: https://openreview.net/pdf?id=SyK00v5xx
"""


class SIF(torch.nn.Module, PhiAPI):
    def __init__(self, input_size, a=1e-3, *args, **kwargs):
        super(SIF, self).__init__()
        super(nn.Module, self).__init__()
        self.input_size = input_size
        self.a = a
        self.dim_constraint = DimConstraint.BATCH_SEQ_CHANNELS(input_size)

    @property
    def output_size(self):
        return self.input_size

    def input_encoding(self):
        return Encoding.INDEX

    def output_encoding(self, input_encoding=None):
        return Encoding.SENTENCE

    @property
    def input_field_keys(self):
        return None

    @property
    def output_field_keys(self):
        return None

    def get_weighted_average(self, We, x, w):
        """
        Compute the weighted average vectors
        :param We: We[i,:] is the vector for word i
        :param x: x[i, :] are the indices of the words in sentence i
        :param w: w[i, :] are the weights for the words in sentence i
        :return: emb[i, :] are the weighted average vector for sentence i
        """
        n_samples = x.shape[0]
        emb = np.zeros((n_samples, We.shape[1]))
        for i in xrange(n_samples):
            emb[i, :] = w[i, :].dot(We[x[i, :], :]) / np.count_nonzero(w[i, :])
        return emb

    def compute_pc(self, X, npc=1):
        """
        Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: component_[i,:] is the i-th pc
        """
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(X)
        return svd.components_

    def remove_pc(self, X, npc=1):
        """
        Remove the projection on the principal components
        :param X: X[i,:] is a data point
        :param npc: number of principal components to remove
        :return: XX[i, :] is the data point after removing its projection
        """
        pc = self.compute_pc(X, npc)
        if npc == 1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)

        return XX

    def lookup_weight(self, ind, freq):
        if ind == 1:
            return 0.0
        elif ind not in freq:
            return 1.0
        else:
            prop = float(freq[ind]) / float(sum(freq.values()))
            return self.a / (self.a + prop)

    def forward(self, inputs):
        # Inputs is a dictionary containing:
        # {field_name: {'encoding': Encoding, 'value': str/FloatTensor, 'sent_len': LongTensor, 'ex_len': LongTensor}}
        weights_from_vocab = all(['field' in field for field in inputs.values()])

        if not weights_from_vocab:
            freq_counters = []

            for field_name, field in inputs.items():
                assert field['encoding'] in self.input_encoding(), \
                    'Encoder does not accept {}'.format(field['encoding'])

                ex = field['value']
                ex_len = field['ex_len']
                batch_size = ex_len.size(0)
                sections = list(ex_len.cpu().numpy())
                ex_splits = ex.split(sections) if batch_size > 1 else [ex]

                if len(freq_counters) == 0:
                    for i in range(batch_size):
                        freq_counters.append(Counter())

                for i, ex in enumerate(ex_splits):
                    freq_counters[i].update(list(ex.cpu().numpy().flatten()))

        for field_name, field in inputs.items():
            assert field['encoding'] in self.input_encoding(), 'Encoder does not accept {}'.format(field['encoding'])

            x = field.pop('value')
            sent_len = field.pop('sent_len')
            ex_len = field['ex_len']

            # shape of x: (sum(ex_len), max(sent_len))
            batch_size = ex_len.size(0)
            sections = list(ex_len.cpu().numpy())
            ex_splits = x.split(sections) if batch_size > 1 else [x]
            sent_len_splits = sent_len.split(sections) if batch_size > 1 else [sent_len]

            assert len(ex_splits) == len(freq_counters)

            sent_encs = []

            for ex, freq, s_len in zip(ex_splits, freq_counters, sent_len_splits):
                ex_np = ex.cpu().numpy()

                # Prepare the inputs with the prepare_inputs function if it is given
                if self.prep_inputs_fn and isinstance(self.prep_inputs_fn, dict):
                    ex = self.prep_inputs_fn[field_name](ex)
                elif self.prep_inputs_fn and callable(self.prep_inputs_fn):
                    ex = self.prep_inputs_fn(ex)

                self.dim_constraint.check(ex)

                # Calculate the word embeddings weights by using the inverse document frequency
                weight_fn = np.vectorize(partial(self.lookup_weight, freq=freq))
                weights = torch.tensor(weight_fn(ex_np), dtype=ex.dtype)
                weights = weights.cuda() if ex.is_cuda else weights
                emb_dim = ex.size(-1)
                weights = weights.view(weights.size(0), weights.size(1), 1).repeat(1, 1, emb_dim)
                assert weights.size() == ex.size()

                # s = s.view(-1, 1).repeat(1, emb_dim).float()
                weighted_sum = torch.sum(weights * ex, dim=(1, ))
                s_len = s_len.view(-1, 1).expand_as(weighted_sum).float()
                sent_encs.append((torch.div(weighted_sum, s_len)))

            field['value'] = torch.cat(tuple(sent_encs), dim=0).view((-1, 1, self.output_size))
            field['encoding'] = self.output_encoding(field['encoding'])

        return inputs

    @property
    def trainable(self):
        return False

    @staticmethod
    def phi_options(phi_options=None):
        phi_options = argparse.ArgumentParser() if phi_options is None else phi_options
        phi_options.add_argument('--a', action='store', default=1e-3, type=float,
                                 help='Whether or not the word embeddings should be weighted by their IDF.')
        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        phi_args, unparsed_args = SIF.phi_options().parse_known_args(unparsed_args)

        logging.info('{} phi arguments: {}'.format(SIF.__name__, phi_args))
        return SIF(input_size, phi_args.a, *args, **kwargs), phi_args, unparsed_args


init = SIF.init


if __name__ == '__main__':
    emb_dim = 300
    t = torch.randint(0, 1000, (10, 20))
    sent_lens = torch.randint(1, 21, (10,))
    ex_lens = torch.LongTensor([3, 7])
    sif_enc = SIF(emb_dim)
    sif_enc.prep_inputs_fn = torch.nn.Embedding.from_pretrained(torch.randn((1000, emb_dim)), freeze=True)
    enc = sif_enc({'test': {'encoding': Encoding.INDEX, 'value': t, 'sent_len': sent_lens, 'ex_len': ex_lens}})
    print(enc)
    print(enc['test']['value'].size())
