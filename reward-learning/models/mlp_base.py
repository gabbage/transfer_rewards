import abc
import argparse

import torch

from helpers.activations import PenalizedTanh
from models.model_api import ModelAPI


class MLPBase(ModelAPI):
    def __init__(self, input_size, num_inputs, hidden_dim, hidden_actv, hidden_dropout, numerical_output):
        super(MLPBase, self).__init__()
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.hidden_actv = hidden_actv
        self.hidden_dropout = hidden_dropout
        self.numerical_output = numerical_output
        self.num_outputs = 1 if numerical_output else 2
        self.num_layers = len(hidden_dim) if hidden_dim else 0

        if self.numerical_output:
            self.prep_outputs_fn = torch.squeeze

        self.create_layers()

    @abc.abstractmethod
    def forward(self, *x):
        pass

    @abc.abstractmethod
    def create_layers(self):
        pass

    def add_layer(self, i, input_dim, output_dim, bias=True, actv="relu", dropout=0.5):
        if "relu" == actv.lower():
            actv = torch.nn.ReLU()
        elif "tanh" == actv.lower():
            actv = torch.nn.Tanh()
        elif "sig" == actv.lower():
            actv = torch.nn.Sigmoid()
        elif "ptanh" == actv.lower():
            actv = PenalizedTanh()
        else:
            actv = None

        layer = torch.nn.Sequential()

        if dropout > 0.0:
            layer.add_module("layer_{}_dropout".format(i), torch.nn.Dropout(dropout))

        layer.add_module("layer_{}_fc".format(i), torch.nn.Linear(input_dim, output_dim, bias))

        if actv:
            layer.add_module("layer_{}_actv".format(i), actv)

        setattr(self, "layer_{}".format(i), layer)

    def get_layer(self, i):
        return getattr(self, "layer_{}".format(i))

    @staticmethod
    def model_options(model_options=None):
        model_options = argparse.ArgumentParser() if model_options is None else model_options
        model_options.add_argument("--hidden_dim", action="append", default=None, type=int,
                                   help="Number of nodes of the fully connected hidden layer before the output layer.")
        model_options.add_argument("--hidden_actv", action="append", default=None, type=str,
                                   help="The activation function of the hidden layer.")
        model_options.add_argument("--hidden_dropout", action="append", default=None, type=float,
                                   help="The percentage of nodes that are affected by dropout in the hidden layer.")
        model_options.add_argument("--numerical_output", action="store", default=0, type=int,
                                   help="Whether or not the model will produce a numerical output for regression.")
        return model_options

    @staticmethod
    def check_args(model_args):
        if model_args.hidden_dim:
            nb_hidden_layers = len(model_args.hidden_dim)

            if model_args.hidden_actv is None:
                model_args.hidden_actv = ["relu"] * nb_hidden_layers

            if model_args.hidden_dropout is None:
                model_args.hidden_dropout = [0.5] * nb_hidden_layers

            assert nb_hidden_layers == len(model_args.hidden_actv), "Number of hidden dimensions and activations has" \
                                                                    "to be equal!"
            assert nb_hidden_layers == len(model_args.hidden_dropout), "Number of hidden dimensions and dropout " \
                                                                       "values has to be equal!"

        return model_args

    @classmethod
    def init(cls, input_size, num_inputs, unparsed_args=None, *args, **kwargs):
        model_args, unparsed_args = MLPBase.model_options().parse_known_args(unparsed_args)
        model_args = MLPBase.check_args(model_args)
        return MLPBase(input_size, num_inputs, model_args.hidden_dim, model_args.hidden_actv, model_args.hidden_dropout,
                       model_args.numerical_output), model_args, unparsed_args


init = MLPBase.init
