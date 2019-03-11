import logging

import torch

from models.mlp_base import MLPBase
from models.phis.encoders import Encoding


class SimplePref(MLPBase):
    def __init__(self, input_size, num_inputs, hidden_dim, hidden_actv, hidden_dropout, numerical_output,
                 reduction='concat'):
        self.reduction = reduction.lower()
        assert self.reduction in ['concat', 'diff'], "Reduction has to be one of 'diff' or 'concat'!"

        if self.reduction == 'diff':
            num_inputs = 2
        else:
            num_inputs = 3

        super(SimplePref, self).__init__(input_size, num_inputs, hidden_dim, hidden_actv, hidden_dropout,
                                         numerical_output)

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
        # Inputs is a dictionary that contains encoded representations that can be used directly
        # {field_name: encoded representation}
        assert 'article' in inputs and inputs['article']['encoding'] == Encoding.DOCUMENT
        assert 'summary' in inputs and len(inputs['summary']) == 2 and \
               all([summary['encoding'] == Encoding.DOCUMENT for summary in inputs['summary']])

        if self.reduction == 'concat':
            full_encoding_vectors = torch.cat([inputs['summary'][0]['value'],
                                               inputs['summary'][1]['value'],
                                               inputs['article']['value']], dim=1)
        elif self.reduction == 'diff':
            diff = inputs['summary'][0]['value'] - inputs['summary'][1]['value']
            full_encoding_vectors = torch.cat([diff, inputs['article']['value']], dim=1)

        return self.predict(full_encoding_vectors)

    @staticmethod
    def model_options(model_options=None):
        model_options = MLPBase.model_options()
        model_options.add_argument('--reduction', action='store', default='concat', type=str,
                                   help="Can either be 'concat' or 'diff' to reduce two summary encodings to one.")
        return model_options

    @classmethod
    def init(cls, input_size, num_inputs, unparsed_args=None, *args, **kwargs):
        model_args, unparsed_args = SimplePref.model_options().parse_known_args(unparsed_args)
        model_args = MLPBase.check_args(model_args)

        logging.info('{} model arguments: {}'.format(SimplePref.__name__, model_args))
        return SimplePref(input_size, num_inputs, model_args.hidden_dim, model_args.hidden_actv,
                          model_args.hidden_dropout, model_args.numerical_output, model_args.reduction), \
            model_args, unparsed_args


init = SimplePref.init
