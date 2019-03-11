# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This file contains the definition of encoders used in https://arxiv.org/pdf/1705.02364.pdf
"""
import argparse
import logging
import os

import time
import numpy as np
import torch
import torch.nn as nn

from models.phis.encoders import Encoding, DimConstraint
from models.phis.phi_api import PhiAPI

"""
BLSTM (max/mean) encoder
"""


class InferSent(torch.nn.Module, PhiAPI):
    def __init__(self, input_size, reduction='max', keep_fixed=1, weight_file=None):
        super(InferSent, self).__init__()
        super(PhiAPI, self).__init__()
        super(torch.nn.Module, self).__init__()
        self.input_size = input_size
        self.enc_lstm_dim = 2048
        self.pool_type = reduction
        self.keep_fixed = bool(keep_fixed)
        self.enc_lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.enc_lstm_dim, bidirectional=True)
        self.dim_constraint = DimConstraint.BATCH_SEQ_CHANNELS(input_size)

        if weight_file and os.path.isfile(weight_file):
            logging.info("Restoring InferSent weights from {}".format(weight_file))
            self.load_state_dict(torch.load(weight_file))

    @property
    def output_size(self):
        return self.enc_lstm_dim * 2

    def input_encoding(self):
        return Encoding.WORD

    def output_encoding(self, input_encoding=None):
        return Encoding.SENTENCE

    @property
    def input_field_keys(self):
        return None

    @property
    def output_field_keys(self):
        return None

    @property
    def trainable(self):
        return True

    def is_cuda(self):
        # Either all weights are on CPU or they are on GPU
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def encode(self, field):
        assert field['encoding'] in self.input_encoding(), 'Encoder does not accept {}'.format(field['encoding'])
        x = field.pop('value')
        sent_len = field.pop('sent_len')
        self.dim_constraint.check(x)

        ctx = torch.no_grad() if self.keep_fixed else torch.enable_grad()

        with ctx:
            x = x.permute(1, 0, 2)

            # Sort by length (keep idx)
            sent_len = sent_len.cpu().numpy()
            sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
            sent_len_sorted = sent_len_sorted.copy()
            idx_unsort = np.argsort(idx_sort)

            idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() else torch.from_numpy(idx_sort)
            sent = x.index_select(1, idx_sort)

            # Handling padding in Recurrent Networks
            sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
            sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
            sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

            # Un-sort by length
            idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
                else torch.from_numpy(idx_unsort)
            sent_output = sent_output.index_select(1, idx_unsort)

            if self.pool_type == 'mean':
                sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).cuda()
                enc = torch.sum(sent_output, 0).squeeze(0)
                enc = enc / sent_len.expand_as(enc)
            elif self.pool_type == 'max':
                # if not self.max_pad:
                #     sent_output[sent_output == 0] = -1e9
                enc = torch.max(sent_output, 0)[0]
                if enc.ndimension() == 3:
                    enc = enc.squeeze(0)
                    assert enc.ndimension() == 2

        field['value'] = enc.view((-1, 1, self.output_size))
        field['encoding'] = self.output_encoding(field['encoding'])

    def forward(self, inputs):
        # Inputs is a dictionary containing:
        # {field_name: {'encoding': Encoding, 'value': str/FloatTensor, 'sent_len': LongTensor, 'ex_len': LongTensor}}
        for field_name, field in inputs.items():
            if isinstance(field, list):
                for f in field:
                    self.encode(f)
            else:
                self.encode(field)

        return inputs

    @staticmethod
    def phi_options(phi_options=None):
        phi_options = argparse.ArgumentParser() if phi_options is None else phi_options
        phi_options.add_argument('--infersent_reduction', action='store', default='max', type=str,
                                 help='Either use max or mean reduction over all hidden states.')
        phi_options.add_argument('--infersent_keep_fixed', action='store', default=1, type=int,
                                 help='Whether or not the weights of InferSent will be trained with backprop.')
        phi_options.add_argument('--infersent_weight_file', action='store', default=None, type=str,
                                 help='Relative or absolute path to InferSent weight file.')

        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        phi_args, unparsed_args = InferSent.phi_options().parse_known_args(unparsed_args)

        if phi_args.infersent_weight_file is None and os.path.isfile(os.path.join('.vector_cache', 'infersent1.pkl')):
            phi_args.infersent_weight_file = os.path.join('.vector_cache', 'infersent1.pkl')

        logging.info('{} phi arguments: {}'.format(InferSent.__name__, phi_args))
        return InferSent(input_size, phi_args.infersent_reduction, phi_args.infersent_keep_fixed,
                         phi_args.infersent_weight_file), phi_args, unparsed_args


init = InferSent.init

if __name__ == '__main__':
    phi = InferSent(300, 'max')
    phi.load_state_dict(torch.load('.vector_cache/infersent1.pkl'))

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    phi.to(dev)

    sent_lens = np.random.randint(1, 31, 10)
    tic = time.time()
    enc = phi({'test': {'encoding': Encoding.WORD,
                        'value': torch.rand((10, 30, 300)).to(dev),
                        'sent_len': torch.from_numpy(sent_lens).to(dev),
                        'ex_len': torch.tensor([2, 4, 4]).to(dev)}})

    print('Time needed: ', time.time() - tic)
    print(enc)
    print(enc['test']['value'].size())
