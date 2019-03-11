import logging

import torch

from models.mlp_base import MLPBase
from models.phis.encoders import Encoding


class TesauroSymmetric(MLPBase):
    def __init__(self, input_size, num_inputs, hidden_dim, hidden_actv, hidden_dropout, numerical_output, invert=False):
        self.invert = bool(invert)
        super(TesauroSymmetric, self).__init__(input_size, num_inputs, hidden_dim, hidden_actv, hidden_dropout,
                                               numerical_output)

    @staticmethod
    def model_options(model_options=None):
        model_options = MLPBase.model_options()
        model_options.add_argument('--invert', action='store', default=0, type=int,
                                   help='Whether or not the second input should be inverted.')
        return model_options

    def create_layers(self):
        assert self.numerical_output == 0, 'TesauroSymmetric can only be used with classification problems!'

        for i in range(self.num_layers):
            input_dim = self.input_size * self.num_inputs if i == 0 else self.hidden_dim[i - 1]
            self.add_layer(i, input_dim, self.hidden_dim[i], actv=self.hidden_actv[i], dropout=self.hidden_dropout[i])

        self.add_layer('output', self.hidden_dim[i], 1, actv='none', dropout=0.5)

    def predict(self, x):
        for i in range(self.num_layers):
            x = self.get_layer(i)(x)

        return self.get_layer('output')(x)

    def forward(self, inputs):
        y_preds = []

        if 'summary' in inputs:
            if not isinstance(inputs['summary'], list):
                summaries = [inputs['summary']]
            else:
                summaries = inputs['summary']

            for i, summary in enumerate(summaries):
                assert summary['encoding'] == Encoding.DOCUMENT

                if 'article' in inputs:
                    assert inputs['article']['encoding'] == Encoding.DOCUMENT

                    feature_vector = torch.cat(tuple([summary['value'], inputs['article']['value']]), dim=1)
                else:
                    feature_vector = summary['value']

                if self.invert and i == 1:
                    y_preds.append(- self.predict(feature_vector))
                else:
                    y_preds.append(self.predict(feature_vector))

        y_pred = torch.cat(tuple(y_preds), dim=1)

        if self.invert:
            y_pred = torch.sum(y_pred, dim=1)

        return y_pred

    @classmethod
    def init(cls, input_size, num_inputs, unparsed_args=None, *args, **kwargs):
        model_args, unparsed_args = TesauroSymmetric.model_options().parse_known_args(unparsed_args)
        model_args = MLPBase.check_args(model_args)

        logging.info('{} model arguments: {}'.format(TesauroSymmetric.__name__, model_args))
        return TesauroSymmetric(input_size, num_inputs, model_args.hidden_dim, model_args.hidden_actv,
                                model_args.hidden_dropout, model_args.numerical_output,
                                bool(model_args.invert)), model_args, unparsed_args


init = TesauroSymmetric.init
