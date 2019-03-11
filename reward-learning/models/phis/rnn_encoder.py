import argparse
import logging

import numpy as np
import torch
import torch.nn

from models.phis.encoders import Encoding, DimConstraint
from models.phis.phi_api import PhiAPI


class RNNEncoder(torch.nn.Module, PhiAPI):
    def __init__(self, input_size, cell_type="lstm", hidden_size=100, bidirectional=0, num_layers=1, dropout=0.0,
                 invert_seq=1, reduction='last'):
        super(RNNEncoder, self).__init__()
        super(PhiAPI, self).__init__()
        super(torch.nn.Module, self).__init__()
        self.input_size = input_size
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.bidirectional = bool(bidirectional)
        self.num_layers = num_layers
        self.dropout = dropout
        self.invert_seq = invert_seq
        self.reduction = reduction
        self.dim_constraint = DimConstraint.BATCH_SEQ_CHANNELS(input_size)

        if reduction.lower() not in ["mean", "max", "last"]:
            raise ValueError("Reduction must be one of last/mean/max!")

        if cell_type.lower() not in ["lstm", "gru"]:
            raise ValueError("Cell type must be one of last/mean/max!")

        if num_layers <= 0:
            raise ValueError("Number of layers must be greater-equal than one.")

        # Define the RNN
        if self.cell_type.lower() == "lstm":
            self.rnn = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True,
                                     bidirectional=self.bidirectional, num_layers=self.num_layers, dropout=self.dropout)
        elif self.cell_type.lower() == "gru":
            self.rnn = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True,
                                    bidirectional=self.bidirectional, num_layers=self.num_layers, dropout=self.dropout)

    def is_cuda(self):
        # Either all weights are on cpu or they are on gpu
        return self.rnn.bias_hh_l0.data.is_cuda

    def encode_words(self, field):
        x = field.pop("value")
        sent_len = field.pop("sent_len")

        # Check tensor for dimensionality, the run will stop if the dimensions do not match the expectation
        self.dim_constraint.check(x)

        if self.invert_seq:
            x = x.flip([1])

        # Sort by length (keep idx)
        sent_len_np = sent_len.cpu().numpy()
        sent_len_sorted, idx_sort = np.sort(sent_len_np)[::-1], np.argsort(-sent_len_np)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)
        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() else torch.from_numpy(idx_sort)
        sent = x.index_select(0, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = torch.nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted, batch_first=True)
        # sent_output, hidden = self.rnn(sent_packed)
        # sent_output = torch.nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() else torch.from_numpy(idx_unsort)
        # sent_output = sent_output.index_select(1, idx_unsort)

        if self.reduction.lower() == 'last':
            x = self.rnn(sent_packed)[1][0]
            x = x.index_select(1, idx_unsort)
        elif self.reduction.lower() == 'mean':
            x = self.rnn(sent_packed)[0]
            x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)[0]
            x = torch.sum(x, 1).squeeze(0)
            x = x / sent_len.float().view(-1, 1).expand_as(x)
        elif self.reduction.lower() == 'max':
            x = self.rnn(sent_packed)[0]
            x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)[0]
            x = torch.max(x, 1)[0]

        field["value"] = x.view((-1, 1, self.output_size))
        field["encoding"] = self.output_encoding(field["encoding"])

    def encode_sentences(self, field):
        x = field.pop("value")
        ex_len = field.pop("ex_len")

        # If the tensor has only two dimensions, add a new second dimension with size 1
        if len(x.size()) == 2:
            x = x.view((x.size(0), 1, x.size(1)))

        # Check tensor for dimensionality, the run will stop if the dimensions do not match the expectation
        self.dim_constraint.check(x)

        # Split the batch into its examples along the first dimension according to the ex_len tensor
        batch_size = ex_len.size(0)
        sections = list(ex_len.cpu().numpy())
        ex_splits = x.split(sections) if batch_size > 1 else [x]

        # Produce the encoding vectors for each example split
        enc_vecs = []

        for ex, seq_len in zip(ex_splits, ex_len):
            # The first dimension of each example is actually the sequence length
            ex = ex.permute((1, 0, 2))

            # Invert the sentence order, such that the RNN sees the last sentence first
            if self.invert_seq:
                ex = ex.flip([1])

            # Now push the split through the RNN and reduce the hidden vector respectively
            if self.reduction.lower() == 'last':
                _, hidden = self.rnn(ex)
                ex = hidden[0].view(1, self.output_size)
            elif self.reduction.lower() == 'mean':
                ex = self.rnn(ex)[0]
                ex = torch.sum(ex, 1)
                ex = torch.div(ex, seq_len.float().expand_as(ex))
                ex = ex.view((1, self.output_size))
            elif self.reduction.lower() == 'max':
                ex = self.rnn(ex)[0]
                ex = torch.max(ex, 1)[0].view((1, self.output_size))

            enc_vecs.append(ex)

        # Feed every example through the RNN and concatenate the hidden vectors along the batch dimension
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

    @property
    def trainable(self):
        return True

    @property
    def output_size(self):
        return self.hidden_size * 2 if self.bidirectional else self.hidden_size

    @staticmethod
    def phi_options(phi_options=None):
        phi_options = argparse.ArgumentParser() if phi_options is None else phi_options
        phi_options.add_argument("--rnn_cell_type", action="store", default="lstm", type=str,
                                 help="Whether the RNN cell should be LSTM or GRU.")
        phi_options.add_argument("--rnn_hidden_size", action="store", default=100, type=int,
                                 help="The number of hidden neurons in the RNN layer.")
        phi_options.add_argument("--rnn_bidirectional", action="store", default=0, type=int,
                                 help="Whether or not the RNN will be bidirectional.")
        phi_options.add_argument("--rnn_num_layers", action="store", default=1, type=int,
                                 help="The number of RNN layers in the RNN encoder.")
        phi_options.add_argument("--rnn_dropout", action="store", default=0.0, type=float,
                                 help="Whether or not the RNN will have dropout, only applied if rnn_layers > 1.")
        phi_options.add_argument("--rnn_invert_seq", action="store", default=1, type=int,
                                 help="Invert the input sequence order before passing it through the RNN.")
        phi_options.add_argument("--rnn_reduction", action="store", default="last", type=str,
                                 help="Either use the last hidden state or the max/mean over all hidden states.")

        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        phi_args, unparsed_args = RNNEncoder.phi_options().parse_known_args(unparsed_args)

        logging.info("{} phi arguments: {}".format(RNNEncoder.__name__, phi_args))
        return RNNEncoder(input_size, phi_args.rnn_cell_type, phi_args.rnn_hidden_size, phi_args.rnn_bidirectional,
                          phi_args.rnn_num_layers, phi_args.rnn_dropout, phi_args.rnn_invert_seq,
                          phi_args.rnn_reduction), phi_args, unparsed_args


init = RNNEncoder.init


if __name__ == "__main__":
    t = torch.randn((10, 20, 100))
    sent_lens = torch.randint(1, 21, (10,))
    ex_lens = torch.LongTensor([3, 7])
    rnn_enc = RNNEncoder(100, reduction='last', bidirectional=1)
    enc = rnn_enc({'test': {'encoding': Encoding.WORD, 'value': t, 'sent_len': sent_lens, 'ex_len': ex_lens}})
    print(enc)
    print(enc["test"]["value"].size())

    rnn_enc = RNNEncoder(200, hidden_size=200, reduction='max', bidirectional=1)
    enc = rnn_enc(enc)
    print(enc)
    print(enc["test"]["value"].size())
