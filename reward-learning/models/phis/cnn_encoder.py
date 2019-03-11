import argparse
import logging

import torch
import torch.nn.functional as F

from helpers.activations import PenalizedTanh
from models.phis.encoders import Encoding
from models.phis.phi_api import PhiAPI


class CNNEncoder(torch.nn.Module, PhiAPI):
    def __init__(self, input_size, filter_widths=None, filter_depths=None, actv='relu', dropout=0.0):
        super(CNNEncoder, self).__init__()
        super(torch.nn.Module, self).__init__()
        self.input_size = input_size
        self.filter_widths = filter_widths if filter_widths is not None else [3, 4, 5]
        self.filter_depths = filter_depths if filter_depths is not None else [100] * len(self.filter_widths)
        self.dropout = dropout

        if "relu" == actv.lower():
            self.actv = torch.nn.ReLU()
        elif "tanh" == actv.lower():
            self.actv = torch.nn.Tanh()
        elif "sig" == actv.lower():
            self.actv = torch.nn.Sigmoid()
        elif "ptanh" == actv.lower():
            self.actv = PenalizedTanh()
        else:
            self.actv = None

        # Initialize the convolutional layers
        for fw, fd in zip(self.filter_widths, self.filter_depths):
            conv = torch.nn.utils.weight_norm(torch.nn.Conv1d(in_channels=self.input_size, out_channels=fd,
                                                              kernel_size=fw))
            setattr(self, "conv_{}".format(fw), conv)

    def pad_for_widest_convolution(self, x, vocab=None):
        if x.size(-1) < max(self.filter_widths):
            pad_size = [0, max(self.filter_widths) - x.size(-1)]

            if vocab is not None:
                x = F.pad(x, pad_size, "constant", vocab.stoi["<pad>"])
            else:
                x = F.pad(x, pad_size, "constant")

        return x

    def encode_words(self, field):
        x = field.pop("value")
        x = x.permute(0, 2, 1)

        # Pad the sentences if they are too short for the widest convolution
        vocab = field["field"].vocab if "field" in field and hasattr(field["field"], "vocab") else None
        x = self.pad_for_widest_convolution(x, vocab)

        # Create the feature maps by applying the convolutions; each feature map has dimensions:
        # [number of sentences, filter_depth, sentence_length - (filter_width - 1)]
        feature_maps = [self.get_conv(fw)(F.dropout(x, self.dropout, self.training)) for fw in self.filter_widths]

        # Searching for the global max per feature over the words dimension in all feature maps and concatenating them
        # into one long vector will produce the sentence representation vector
        if self.actv is not None:
            x = torch.cat(tuple([F.max_pool1d(self.actv(fm), fm.size(2)) for fm in feature_maps]), dim=1)
        else:
            x = torch.cat(tuple([F.max_pool1d(fm, fm.size(2)) for fm in feature_maps]), dim=1)

        field["value"] = x.permute((0, 2, 1))
        field["encoding"] = self.output_encoding(field["encoding"])

    def encode_sentences(self, field):
        x = field.pop("value")
        ex_len = field.pop("ex_len")

        batch_size = ex_len.size(0)
        sections = list(ex_len.cpu().numpy())
        ex_splits = x.split(sections) if batch_size > 1 else [x]

        # Produce the encoding vectors for each example split
        enc_vecs = []

        for ex, seq_len in zip(ex_splits, ex_len):
            ex = ex.permute(1, 2, 0)

            # Pad the sentences if they are too short for the widest convolution
            vocab = field["field"].vocab if "field" in field and hasattr(field["field"], "vocab") else None
            ex = self.pad_for_widest_convolution(ex, vocab)

            # Create the feature maps by applying the convolutions; each feature map has dimensions:
            # [number of sentences, filter_depth, sentence_length - (filter_width - 1)]
            feature_maps = [self.get_conv(fw)(F.dropout(ex, self.dropout, self.training)) for fw in self.filter_widths]

            # Searching for the global max per feature over the words dimension in all feature maps and concatenating
            # them into one long vector will produce the sentence representation vector
            if self.actv is not None:
                ex = torch.cat(tuple([F.max_pool1d(self.actv(fm), fm.size(2)) for fm in feature_maps]), dim=1)
            else:
                ex = torch.cat(tuple([F.max_pool1d(fm, fm.size(2)) for fm in feature_maps]), dim=1)

            enc_vecs.append(ex.view((1, self.output_size)))

        field["value"] = torch.cat(tuple(enc_vecs), dim=0)
        field["encoding"] = self.output_encoding(field["encoding"])

    def encode(self, field):
        assert field["encoding"] in self.input_encoding(), "Encoder does not accept {}".format(field["encoding"])

        if field["encoding"] == Encoding.WORD:
            return self.encode_words(field)
        elif field["encoding"] == Encoding.SENTENCE:
            return self.encode_sentences(field)

    def forward(self, inputs):
        # Inputs is a dictionary containing:
        # {field_name: {"encoding": Encoding, "value": str/FloatTensor, "sent_len": LongTensor, "ex_len": LongTensor}}
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
        assert input_encoding is not None
        return Encoding(int(input_encoding) << 1)

    @property
    def input_field_keys(self):
        return None

    @property
    def output_field_keys(self):
        return None

    def get_conv(self, i):
        return getattr(self, "conv_{}".format(i))

    @property
    def trainable(self):
        return True

    @property
    def output_size(self):
        return sum(self.filter_depths)

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
        phi_args, unparsed_args = CNNEncoder.phi_options().parse_known_args(unparsed_args)

        if phi_args.cnn_filter_widths is None:
            phi_args.cnn_filter_widths = [3, 4, 5]

        if phi_args.cnn_filter_depths is None:
            phi_args.cnn_filter_depths = [100] * len(phi_args.cnn_filter_widths)

        logging.info("{} phi arguments: {}".format(CNNEncoder.__name__, phi_args))
        return CNNEncoder(input_size, phi_args.cnn_filter_widths, phi_args.cnn_filter_depths,
                          phi_args.cnn_actv, phi_args.cnn_dropout), phi_args, unparsed_args


init = CNNEncoder.init

if __name__ == "__main__":
    t = torch.randn((10, 20, 100))
    sent_lens = torch.randint(1, 21, (10,))
    ex_lens = torch.LongTensor([3, 7])
    cnn_enc = CNNEncoder(100)
    enc = cnn_enc({'test': {'encoding': Encoding.WORD, 'value': t, 'sent_len': sent_lens, 'ex_len': ex_lens}})
    print(enc)
    print(enc["test"]["value"].size())

    cnn_enc = CNNEncoder(300)
    enc = cnn_enc(enc)
    print(enc)
    print(enc["test"]["value"].size())
