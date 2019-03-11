import argparse
import logging

import torch
import torch.nn.functional as F

from helpers.activations import PenalizedTanh
from models.phis.encoders import Encoding
from models.phis.phi_api import PhiAPI


class SimRedCNN(torch.nn.Module, PhiAPI):
    def __init__(self, input_size, filter_widths=None, filter_depths=None, cnn_actv='relu', cnn_dropout=0.0):
        super(SimRedCNN, self).__init__()
        super(torch.nn.Module, self).__init__()

        self.input_size = input_size
        self.filter_widths = filter_widths if filter_widths is not None else [1, 2, 3]
        self.filter_depths = filter_depths if filter_depths is not None else [20] * len(self.filter_widths)
        self.cnn_dropout = cnn_dropout

        if "relu" == cnn_actv.lower():
            self.cnn_actv = torch.nn.ReLU()
        elif "tanh" == cnn_actv.lower():
            self.cnn_actv = torch.nn.Tanh()
        elif "sig" == cnn_actv.lower():
            self.cnn_actv = torch.nn.Sigmoid()
        elif "ptanh" == cnn_actv.lower():
            self.cnn_actv = PenalizedTanh()
        else:
            self.cnn_actv = None

        # Initialize the convolutional layers
        for fw, fd in zip(self.filter_widths, self.filter_depths):
            conv_sa = torch.nn.utils.weight_norm(torch.nn.Conv2d(in_channels=1, out_channels=fd, kernel_size=fw))
            conv_ss = torch.nn.utils.weight_norm(torch.nn.Conv2d(in_channels=1, out_channels=fd, kernel_size=fw))
            setattr(self, "conv_{}sa".format(fw), conv_sa)
            setattr(self, "conv_{}ss".format(fw), conv_ss)

    @property
    def trainable(self):
        return True

    @property
    def output_size(self):
        return sum(self.filter_depths) * 2

    def input_encoding(self):
        return Encoding.WORD | Encoding.SENTENCE

    def output_encoding(self, input_encoding=None):
        return Encoding.DOCUMENT

    @property
    def input_field_keys(self):
        return ['summary', 'article']

    @property
    def output_field_keys(self):
        return ['summary']

    def normalize(self, t):
        t = t.view((-1, self.input_size))
        # Increase numerical stability by value clamping
        n = torch.clamp(torch.norm(t, dim=1), min=1e-10)
        return t / n.view((-1, 1))

    def similarity_matrix(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)

        return torch.matmul(x, y.transpose(0, 1))

    def pad_for_widest_convolution(self, x):
        pad_size = [0, max(0, max(self.filter_widths) - x.size(1)), 0, max(0, max(self.filter_widths) - x.size(0))]
        x = F.pad(x, pad_size, "constant")

        return x.view((1, 1, x.size(0), x.size(1)))

    def forward(self, inputs):
        # Inputs is a dictionary containing:
        # {field_name: {"encoding": Encoding, "value": str/FloatTensor, "sent_len": LongTensor, "ex_len": LongTensor}}
        assert 'summary' in inputs
        assert 'ex_len' in inputs['article']
        assert 'article' in inputs and inputs['article']['encoding'] in (Encoding.WORD | Encoding.SENTENCE)

        summaries = inputs['summary'] if isinstance(inputs['summary'], list) else [inputs['summary']]
        article_values = inputs['article']['value']
        article_ex_len = inputs['article']['ex_len'].tolist()

        for summary in summaries:
            assert summary['encoding'] in (Encoding.WORD | Encoding.SENTENCE)
            assert 'ex_len' in summary

            summary_values = summary['value']
            summary_ex_len = summary['ex_len'].tolist()

            encodings = []

            for s, a in zip(summary_values.split(summary_ex_len), article_values.split(article_ex_len)):
                sa_sim = self.pad_for_widest_convolution(self.similarity_matrix(s, a))
                ss_sim = self.similarity_matrix(s, s)
                ss_sim = self.pad_for_widest_convolution(ss_sim - torch.eye(ss_sim.size(0), device=ss_sim.device))
                sa_fm = [self.get_conv(fw, 'sa')(sa_sim) for fw in self.filter_widths]
                ss_fm = [self.get_conv(fw, 'ss')(ss_sim) for fw in self.filter_widths]

                if self.cnn_actv is not None:
                    sa_actv = [F.avg_pool2d(self.cnn_actv(fm), (fm.size(2), fm.size(3))) for fm in sa_fm]
                    ss_actv = [F.avg_pool2d(self.cnn_actv(fm), (fm.size(2), fm.size(3))) for fm in ss_fm]
                    sa_vec = torch.cat(tuple([F.dropout2d(a, self.cnn_dropout, self.training) for a in sa_actv]), dim=1)
                    ss_vec = torch.cat(tuple([F.dropout2d(a, self.cnn_dropout, self.training) for a in ss_actv]), dim=1)
                else:
                    sa_vec = torch.cat(tuple([F.dropout2d(F.avg_pool2d(fm, (fm.size(2), fm.size(3))), self.cnn_dropout,
                                                          self.training)
                                              for fm in sa_fm]), dim=1)
                    ss_vec = torch.cat(tuple([F.dropout2d(F.avg_pool2d(fm, (fm.size(2), fm.size(3))), self.cnn_dropout,
                                                          self.training)
                                              for fm in ss_fm]), dim=1)

                encodings.append(torch.cat((sa_vec, ss_vec), dim=1).view((1, -1)))

            summary['value'] = torch.cat(tuple(encodings))
            summary['encoding'] = Encoding.DOCUMENT

        inputs.pop('article')
        return inputs

    def get_conv(self, i, postfix):
        return getattr(self, "conv_{}{}".format(i, postfix))

    @staticmethod
    def phi_options(phi_options=None):
        phi_options = argparse.ArgumentParser() if phi_options is None else phi_options
        phi_options.add_argument("--cnn_filter_widths", action="append", default=None, type=int,
                                 help="The widths of applied convolution filters.")
        phi_options.add_argument("--cnn_filter_depths", action="append", default=None, type=int,
                                 help="The number of convolution filters per width.")
        phi_options.add_argument("--cnn_actv", action="store", default='relu', type=str,
                                 help="The activation function that is applied after the convolution.")
        phi_options.add_argument("--cnn_dropout", action="store", default=0.0, type=float,
                                 help="Percentage of dropout after the activation function and convolution.")

        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        phi_args, unparsed_args = SimRedCNN.phi_options().parse_known_args(unparsed_args)

        if phi_args.cnn_filter_widths is None:
            phi_args.cnn_filter_widths = [1, 2, 3]

        if phi_args.cnn_filter_depths is None:
            phi_args.cnn_filter_depths = [20] * len(phi_args.cnn_filter_widths)

        logging.info("{} phi arguments: {}".format(SimRedCNN.__name__, phi_args))
        return SimRedCNN(input_size, phi_args.cnn_filter_widths, phi_args.cnn_filter_depths,
                         phi_args.cnn_actv, phi_args.cnn_dropout), phi_args, unparsed_args


init = SimRedCNN.init
