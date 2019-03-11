import logging

import torch

from models.mlp_base import MLPBase
from models.phis.encoders import Encoding


class Regression(MLPBase):
    @staticmethod
    def model_options(model_options=None):
        model_options = MLPBase.model_options()
        model_options.set_defaults(numerical_output=1)
        return model_options

    def create_layers(self):
        for i in range(self.num_layers):
            input_dim = self.input_size * self.num_inputs if i == 0 else self.hidden_dim[i - 1]
            self.add_layer(i, input_dim, self.hidden_dim[i], actv=self.hidden_actv[i], dropout=self.hidden_dropout[i])

        self.add_layer('output', self.hidden_dim[i], self.num_outputs, actv='none', dropout=0.5)

    def predict(self, x):
        for i in range(self.num_layers):
            x = self.get_layer(i)(x)

        return self.get_layer('output')(x)

    def forward(self, inputs):
        # Inputs is a dictionary containing:
        # {field_name: feature-vector representation}
        assert 'summary' in inputs and inputs['summary']['encoding'] == Encoding.DOCUMENT

        if 'article' in inputs:
            assert inputs['article']['encoding'] == Encoding.DOCUMENT

            x = torch.cat(tuple([inputs['summary']['value'], inputs['article']['value']]), dim=1)
        else:
            x = inputs['summary']['value']

        y_pred = self.predict(x)

        # Preparing the output values if necessary
        if self.prep_outputs_fn:
            y_pred = self.prep_outputs_fn(y_pred)

        return y_pred

    @classmethod
    def init(cls, input_size, num_inputs, unparsed_args=None, *args, **kwargs):
        model_args, unparsed_args = Regression.model_options().parse_known_args(unparsed_args)
        model_args = MLPBase.check_args(model_args)

        logging.info('{} model arguments: {}'.format(Regression.__name__, model_args))
        return Regression(input_size, num_inputs, model_args.hidden_dim, model_args.hidden_actv, model_args.hidden_dropout,
                          model_args.numerical_output), model_args, unparsed_args


init = Regression.init
