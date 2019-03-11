import argparse
import logging

import torch
import torch.nn

from models.phis.encoders import Encoding, DimConstraint
from models.phis.phi_api import PhiAPI

"""
Power mean sentence embeddings based on: https://arxiv.org/pdf/1803.01400.pdf
"""


class PowerMean(torch.nn.Module, PhiAPI):
    def __init__(self, input_size, p_values=None):
        super(PowerMean, self).__init__()
        super(PhiAPI, self).__init__()
        super(torch.nn.Module, self).__init__()
        self.input_size = input_size
        self.p_values = p_values if p_values is not None else ['1']
        self.dim_constraint = DimConstraint.BATCH_SEQ_CHANNELS(input_size)

    def encode(self, field):
        assert field['encoding'] in self.input_encoding(), 'Encoder does not accept {}'.format(field['encoding'])
        x = field.pop('value')
        sent_len = field.pop('sent_len')
        self.dim_constraint.check(x)
        sent_encs = []
        sentences = x.split(1, dim=0)

        for s, s_len in zip(sentences, sent_len):
            s_len = s_len.item()
            s, _ = s.split((s_len, s.size(1) - s_len), dim=1)
            sent_encs.append(torch.cat(tuple([PowerMean.calculate(s, p, dim=1) for p in self.p_values]), dim=1))

        sent_encs = torch.cat(tuple(sent_encs), dim=0)
        field['value'] = sent_encs.view((-1, 1, self.output_size))
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
    def calculate(x, p, dim=0):
        if p == '-inf':
            return torch.min(x, dim=dim)[0]
        elif p == 'inf':
            return torch.max(x, dim=dim)[0]
        else:
            p = float(p)

            if p == 0.0:
                # geometric mean
                n = x.size(dim)
                return torch.prod(x, dim=dim).pow(1 / n)
            elif p == -1.0:
                # harmonic mean
                n = x.size(dim)
                return n / torch.sum(1 / x, dim=tuple(dim))
            else:
                return (torch.mean(x.pow(p), dim=tuple(dim))).pow(1 / p)

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
        return False

    @property
    def output_size(self):
        return self.input_size * len(self.p_values)

    @staticmethod
    def phi_options(phi_options=None):
        phi_options = argparse.ArgumentParser() if phi_options is None else phi_options
        phi_options.add_argument('--p_values', action='append', default=None, type=str,
                                 help='List of p-values to produce different power means from word embeddings.')

        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        phi_args, unparsed_args = PowerMean.phi_options().parse_known_args(unparsed_args)

        if phi_args.p_values is None:
            phi_args.p_values = ['1']

        logging.info('{} phi arguments: {}'.format(PowerMean.__name__, phi_args))
        return PowerMean(input_size, phi_args.p_values), phi_args, unparsed_args


init = PowerMean.init

if __name__ == '__main__':
    t = torch.randn((10, 20, 100))
    sent_lens = torch.randint(1, 21, (10,))
    ex_lens = torch.LongTensor([3, 7])
    pmean_enc = PowerMean(100, p_values=['-1', '0', 'inf', '-inf', '1', '2'])
    enc = pmean_enc({'test': {'encoding': Encoding.WORD, 'value': t, 'sent_len': sent_lens, 'ex_len': ex_lens}})
    print(enc)
    print(enc['test']['value'].size())
