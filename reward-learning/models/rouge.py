import logging

import torch

from models.mlp_base import MLPBase


class Rouge(MLPBase):
    def create_layers(self):
        for i in range(self.num_layers):
            input_dim = self.input_size * self.num_inputs if i == 0 else self.hidden_dim[i - 1]
            self.add_layer(i, input_dim, self.hidden_dim[i], actv=self.hidden_actv[i], dropout=self.hidden_dropout[i])

        self.add_layer("output", self.hidden_dim[i], 9, actv="none", dropout=0.5)

    def predict(self, x):
        for i in range(self.num_layers):
            x = self.get_layer(i)(x)

        return self.get_layer("output")(x)

    def forward(self, inputs):
        # Inputs is a dictionary that contains encoded representations that can be used directly
        # {field_name: encoded representation}
        assert "summary_1" in inputs
        assert "summary_2" in inputs
        assert "article" in inputs

        full_encoding_vectors = torch.cat([inputs["summary_1"], inputs["summary_2"], inputs["article"]], dim=1)

        return self.predict(full_encoding_vectors)

    @classmethod
    def init(cls, input_size, num_inputs, unparsed_args=None, *args, **kwargs):
        model_args, unparsed_args = Rouge.model_options().parse_known_args(unparsed_args)
        model_args = MLPBase.check_args(model_args)

        logging.info("Model arguments: {}".format(model_args))
        return Rouge(input_size, num_inputs, model_args.hidden_dim, model_args.hidden_actv,
                     model_args.hidden_dropout, model_args.numerical_output), model_args, unparsed_args


init = Rouge.init
