import logging

import torch

from models.mlp_base import MLPBase
from models.phis.encoders import Encoding


class CmpNN(MLPBase):
    def __init__(self, input_size, num_inputs, hidden_dim, hidden_actv, hidden_dropout, numerical_output):
        super(CmpNN, self).__init__(input_size, num_inputs, hidden_dim, hidden_actv, hidden_dropout, numerical_output)

    @staticmethod
    def model_options(model_options=None):
        model_options = MLPBase.model_options()
        return model_options

    def create_layers(self):
        assert self.numerical_output == 0, '{} can only be used with classification problems!'.format(CmpNN.__name__)
        i = -1

        for i in range(self.num_layers):
            input_dim = self.input_size * self.num_inputs if i == 0 else self.hidden_dim[i - 1]
            self.add_layer('{}a'.format(i), input_dim, self.hidden_dim[i], actv=self.hidden_actv[i],
                           dropout=self.hidden_dropout[i])
            self.add_layer('{}b'.format(i), input_dim, self.hidden_dim[i], actv=self.hidden_actv[i],
                           dropout=self.hidden_dropout[i])

        self.add_layer('{}a'.format(i+1), self.hidden_dim[i], 1, actv='none', dropout=0.5)
        self.add_layer('{}b'.format(i+1), self.hidden_dim[i], 1, actv='none', dropout=0.5)

    def predict(self, x1, x2):
        i = -1

        for i in range(self.num_layers):
            tmp_x1 = self.get_layer('{}a'.format(i))(x1) + self.get_layer('{}b'.format(i))(x2)
            tmp_x2 = self.get_layer('{}b'.format(i))(x1) + self.get_layer('{}a'.format(i))(x2)
            x1 = tmp_x1 / 2.0
            x2 = tmp_x2 / 2.0

        tmp_x1 = self.get_layer('{}a'.format(i+1))(x1) + self.get_layer('{}b'.format(i+1))(x2)
        tmp_x2 = self.get_layer('{}b'.format(i+1))(x1) + self.get_layer('{}a'.format(i+1))(x2)
        x1 = tmp_x1 / 2.0
        x2 = tmp_x2 / 2.0

        return torch.cat(tuple([x1, x2]), dim=-1)

    def forward(self, inputs):
        assert 'summary' in inputs and isinstance(inputs['summary'], list) and len(inputs['summary']) == 2

        summary_1 = inputs['summary'][0]['value']
        summary_2 = inputs['summary'][1]['value']

        if 'article' in inputs and inputs['article']['encoding'] == Encoding.DOCUMENT:
            article = inputs['article']['value']

            # Summary conditioned on the article
            summary_1 = torch.cat((summary_1, article), dim=1)
            summary_2 = torch.cat((summary_2, article), dim=1)

        return self.predict(summary_1, summary_2)

    @classmethod
    def init(cls, input_size, num_inputs, unparsed_args=None, *args, **kwargs):
        model_args, unparsed_args = CmpNN.model_options().parse_known_args(unparsed_args)
        model_args = MLPBase.check_args(model_args)

        logging.info('{} model arguments: {}'.format(CmpNN.__name__, model_args))
        return CmpNN(input_size, num_inputs, model_args.hidden_dim, model_args.hidden_actv,
                     model_args.hidden_dropout, model_args.numerical_output), model_args, unparsed_args


init = CmpNN.init


if __name__ == '__main__':
    a = torch.randn(1, 100)
    s1 = torch.randn(1, 100)
    s2 = torch.randn(1, 100)
    model = CmpNN(100, 2, [50], ['relu'], [0.5], 0)
    print(model)

    batch = {'article': {'encoding': Encoding.DOCUMENT, 'value': a},
             'summary': [{'encoding': Encoding.DOCUMENT, 'value': s1},
                         {'encoding': Encoding.DOCUMENT, 'value': s2}]}
    pred = model(batch)
    print(pred, torch.argmax(pred))
    
    batch = {'article': {'encoding': Encoding.DOCUMENT, 'value': a},
             'summary': [{'encoding': Encoding.DOCUMENT, 'value': s2},
                         {'encoding': Encoding.DOCUMENT, 'value': s1}]}
    pred = model(batch)
    print(pred, torch.argmax(pred))
