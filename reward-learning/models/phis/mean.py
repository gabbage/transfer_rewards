import logging

import torch
import torch.nn

from models.phis.encoders import Encoding, DimConstraint
from models.phis.phi_api import PhiAPI

"""
Power mean sentence embeddings based on: https://arxiv.org/pdf/1803.01400.pdf
"""


class Mean(torch.nn.Module, PhiAPI):
    def __init__(self, input_size):
        super(Mean, self).__init__()
        super(PhiAPI, self).__init__()
        super(torch.nn.Module, self).__init__()
        self.input_size = input_size
        self.dim_constraint = DimConstraint.BATCH_SEQ_CHANNELS(input_size)

    def encode_words(self, field):
        x = field.pop('value')
        sent_len = field.pop('sent_len')
        self.dim_constraint.check(x)
        sent_encs = []
        sentences = x.split(1, dim=0)

        for s, s_len in zip(sentences, sent_len):
            s_len = s_len.item()
            s, _ = s.split((s_len, s.size(1) - s_len), dim=1)
            sent_encs.append(torch.mean(s, dim=1))

        sent_encs = torch.cat(tuple(sent_encs), dim=0)
        field['value'] = sent_encs.view((-1, 1, self.output_size))
        field['encoding'] = self.output_encoding(field['encoding'])

    def encode_sentences(self, field):
        x = field.pop('value')
        ex_len = field.pop('ex_len').tolist()
        self.dim_constraint.check(x)
        docs = x.split(ex_len, dim=0)
        doc_encs = torch.cat(tuple([torch.mean(d, dim=0).view((1, -1)) for d in docs]), dim=0)
        field['value'] = doc_encs.view((-1, self.output_size))
        field['encoding'] = self.output_encoding(field['encoding'])

    def encode(self, field):
        assert field["encoding"] in self.input_encoding(), "Encoder does not accept {}".format(field["encoding"])

        if field["encoding"] == Encoding.WORD:
            return self.encode_words(field)
        elif field["encoding"] == Encoding.SENTENCE:
            return self.encode_sentences(field)

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

    def input_encoding(self):
        return Encoding.WORD | Encoding.SENTENCE

    def output_encoding(self, input_encoding=None):
        assert input_encoding is not None and input_encoding in self.input_encoding()
        return Encoding(int(input_encoding) << 1)

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
        return self.input_size

    @staticmethod
    def phi_options(phi_options=None):
        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        logging.info('{} phi arguments: {}'.format(Mean.__name__, None))
        return Mean(input_size), None, unparsed_args


init = Mean.init

if __name__ == '__main__':
    t = torch.randn((10, 20, 100))
    sent_lens = torch.randint(1, 21, (10,))
    ex_lens = torch.LongTensor([3, 7])
    mean_enc = Mean(100)
    enc = mean_enc({'test': {'encoding': Encoding.WORD, 'value': t, 'sent_len': sent_lens, 'ex_len': ex_lens}})

    assert enc['test']['value'].numpy().shape == (10, 1, 100), "Encoding has shape {}".format(enc['test']['value'].size())

    enc = mean_enc(enc)

    assert enc['test']['value'].numpy().shape == (2, 100), "Encoding has shape {}".format(enc['test']['value'].size())
