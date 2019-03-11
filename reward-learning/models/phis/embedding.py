import argparse

import torch
import torch.nn

from models.phis.encoders import Encoding, DimConstraint
from models.phis.phi_api import PhiAPI


class EmbeddingEncoder(torch.nn.Module, PhiAPI):
    def __init__(self, embedding_dim, embedding_layer=None):
        super(EmbeddingEncoder, self).__init__()
        super(PhiAPI, self).__init__()
        super(torch.nn.Module, self).__init__()

        if embedding_layer:
            self.prep_inputs_fn = embedding_layer
            self.embedding_dim = embedding_layer.embedding_dim
        else:
            self.embedding_dim = embedding_dim

        self.dim_constraint = DimConstraint([(0, -1), (1, -1)])

    def encode(self, field_name, field):
        assert field["encoding"] in self.input_encoding(), "Encoder does not accept {}".format(field["encoding"])
        x = field.pop("value")
        self.dim_constraint.check(x)

        if self.prep_inputs_fn and isinstance(self.prep_inputs_fn, dict):
            field["value"] = self.prep_inputs_fn[field_name](x)
            field["encoding"] = self.output_encoding()
        elif self.prep_inputs_fn and callable(self.prep_inputs_fn):
            field["value"] = self.prep_inputs_fn(x)
            field["encoding"] = self.output_encoding()

    def forward(self, inputs):
        for field_name, field in inputs.items():
            if isinstance(field, list):
                for f in field:
                    self.encode(field_name, f)
            else:
                self.encode(field_name, field)

        return inputs

    @property
    def trainable(self):
        if self.prep_inputs_fn is None:
            return False
        else:
            return self.prep_inputs_fn.weight.requires_grad

    @property
    def output_size(self):
        return self.embedding_dim

    @staticmethod
    def phi_options(phi_options=None):
        phi_options = argparse.ArgumentParser() if phi_options is None else phi_options
        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        phi_args, unparsed_args = EmbeddingEncoder.phi_options().parse_known_args(unparsed_args)
        return EmbeddingEncoder(input_size), phi_args, unparsed_args

    def input_encoding(self):
        return Encoding.INDEX

    def output_encoding(self, input_encoding=None):
        return Encoding.WORD

    @property
    def input_field_keys(self):
        return None

    @property
    def output_field_keys(self):
        return None


init = EmbeddingEncoder.init
