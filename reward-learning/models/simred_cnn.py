import logging

import torch
import torch.nn.functional as F

from helpers.activations import PenalizedTanh
from models.mlp_base import MLPBase
from models.phis.encoders import Encoding


class SimRedCNN(MLPBase):
    def __init__(self, input_size, num_inputs, filter_widths=None, filter_depths=None, cnn_actv='relu', cnn_dropout=0.0,
                 hidden_dim=None, hidden_actv=None, hidden_dropout=None, numerical_output=0):
        self.input_size = input_size
        self.filter_widths = filter_widths if filter_widths is not None else [1, 2, 3]  # , 4, 5]
        self.filter_depths = filter_depths if filter_depths is not None else [20] * len(self.filter_widths)
        self.cnn_dropout = cnn_dropout

        super(SimRedCNN, self).__init__(input_size, num_inputs, hidden_dim, hidden_actv, hidden_dropout,
                                        numerical_output)

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

    def create_layers(self):
        for i in range(self.num_layers):
            input_size = sum(self.filter_depths) * self.num_inputs if i == 0 else self.hidden_dim[i - 1]
            self.add_layer(i, input_size, self.hidden_dim[i], actv=self.hidden_actv[i], dropout=self.hidden_dropout[i])

        self.add_layer('output', self.hidden_dim[i], self.num_outputs, actv='none', dropout=0.5)

    def predict(self, x):
        for i in range(self.num_layers):
            x = self.get_layer(i)(x)

        return self.get_layer('output')(x)

    def forward(self, inputs):
        # Inputs is a dictionary containing:
        # {field_name: {"encoding": Encoding, "value": str/FloatTensor, "sent_len": LongTensor, "ex_len": LongTensor}}
        assert 'summary' in inputs
        assert 'ex_len' in inputs['article']
        assert 'article' in inputs and inputs['article']['encoding'] in (Encoding.WORD | Encoding.SENTENCE)

        y_preds = []
        summaries = inputs['summary'] if isinstance(inputs['summary'], list) else [inputs['summary']]
        article_values = inputs['article']['value']
        article_ex_len = inputs['article']['ex_len'].tolist()

        for summary in summaries:
            assert summary['encoding'] in (Encoding.WORD | Encoding.SENTENCE)
            assert 'ex_len' in summary

            summary_values = summary['value']
            summary_ex_len = summary['ex_len'].tolist()

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

                x = torch.cat((sa_vec, ss_vec), dim=1)
                y_preds.append(self.predict(x.view((1, -1))))

        y_pred = torch.cat(tuple(y_preds)).view((len(summaries), len(article_ex_len))).permute((1, 0))

        return y_pred

    def get_conv(self, i, postfix):
        return getattr(self, "conv_{}{}".format(i, postfix))

    @staticmethod
    def model_options(model_options=None):
        model_options = MLPBase.model_options()
        model_options.set_defaults(numerical_output=1)
        model_options.add_argument("--cnn_filter_widths", action="append", default=None, type=int,
                                   help="The widths of applied convolution filters.")
        model_options.add_argument("--cnn_filter_depths", action="append", default=None, type=int,
                                   help="The number of convolution filters per width.")
        model_options.add_argument("--cnn_actv", action="store", default='relu', type=str,
                                   help="The activation function that is applied after the convolution.")
        model_options.add_argument("--cnn_dropout", action="store", default=0.0, type=float,
                                   help="Percentage of dropout after the activation function and convolution.")

        return model_options

    @classmethod
    def init(cls, input_size, num_inputs, unparsed_args=None, *args, **kwargs):
        model_args, unparsed_args = SimRedCNN.model_options().parse_known_args(unparsed_args)
        model_args = MLPBase.check_args(model_args)

        if model_args.cnn_filter_widths is None:
            model_args.cnn_filter_widths = [1, 2, 3]  # , 4, 5]

        if model_args.cnn_filter_depths is None:
            model_args.cnn_filter_depths = [20] * len(model_args.cnn_filter_widths)

        logging.info("{} model arguments: {}".format(SimRedCNN.__name__, model_args))
        return SimRedCNN(input_size, num_inputs, model_args.cnn_filter_widths, model_args.cnn_filter_depths,
                         model_args.cnn_actv, model_args.cnn_dropout, model_args.hidden_dim,
                         model_args.hidden_actv, model_args.hidden_dropout,
                         model_args.numerical_output), model_args, unparsed_args


init = SimRedCNN.init
