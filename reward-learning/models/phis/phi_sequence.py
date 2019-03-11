import argparse
import importlib
import logging

import torch
import torch.nn

from models.phis.embedding import EmbeddingEncoder
from models.phis.encoders import EncoderSequence, Encoding
from models.phis.phi_api import PhiAPI


class PhiSequence(torch.nn.Module, PhiAPI):
    def __init__(self, enc_seq, input_field_keys, output_field_keys):
        super(PhiSequence, self).__init__()
        super(torch.nn.Module, self).__init__()
        self.enc_seq = enc_seq
        self._input_field_keys = input_field_keys
        self._output_field_keys = output_field_keys

        assert len(enc_seq) > 0, 'The EncoderSequence must have at least one Encoder!'

        for i, enc in enumerate(enc_seq):
            if isinstance(enc, torch.nn.Module):
                self.add_module('module_{}'.format(i), enc)

    def forward(self, inputs):
        for i, enc in enumerate(self.enc_seq):
            outputs = getattr(self, 'module_{}'.format(i))(inputs) if hasattr(self, 'module_{}'.format(i)) else enc(inputs)
            inputs = outputs

        return outputs

    def input_encoding(self):
        return self.enc_seq.input_encoding()

    def output_encoding(self, input_encoding=None):
        return self.enc_seq.output_encoding(input_encoding)

    @property
    def input_field_keys(self):
        return self._input_field_keys

    @property
    def output_field_keys(self):
        return self._output_field_keys

    @property
    def trainable(self):
        return any([enc.trainable for enc in self.enc_seq])

    @property
    def output_size(self):
        return self.enc_seq.last().output_size

    @property
    def num_inputs(self):
        return len(self._input_field_keys)

    @property
    def num_outputs(self):
        return len(self._output_field_keys)

    @staticmethod
    def phi_options(phi_options=None):
        phi_options = argparse.ArgumentParser() if phi_options is None else phi_options
        phi_options.add_argument('--phis', action='append', default=None, type=str,
                                 help='Ordered list of phi names that get initialized as well.')

        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args, phis, input_field_keys, device, fields=None):
        output_field_keys = input_field_keys

        if phis is None or len(phis) == 0:
            return None, None, unparsed_args, input_field_keys
        else:
            enc_seq = EncoderSequence()
            phi_args = None

            for sub_phi in phis:
                sub_phi_module = importlib.import_module('models.phis.{}'.format(sub_phi))
                sub_phi, sub_phi_args, unparsed_args = sub_phi_module.init(input_size, unparsed_args,
                                                                           device=device, fields=fields)
                input_size = sub_phi.output_size

                # Check if phi receives all necessary input fields
                if sub_phi.input_field_keys is not None:
                    assert all([(f in input_field_keys) for f in sub_phi.input_field_keys])

                if sub_phi.output_field_keys is not None:
                    output_field_keys = sub_phi.output_field_keys

                if phi_args is None:
                    phi_args = sub_phi_args
                elif sub_phi_args is not None:
                    vars(phi_args).update(vars(sub_phi_args))

                enc_seq.append(sub_phi)

            if Encoding.INDEX not in enc_seq.input_encoding() and Encoding.WORD in enc_seq.input_encoding():
                logging.debug("Add embedding layer as first encoder in sequence!")
                enc_seq.prepend(EmbeddingEncoder(input_size))

            return cls(enc_seq, input_field_keys, output_field_keys), phi_args, unparsed_args


init = PhiSequence.init


if __name__ == '__main__':
    # phi_seq, phi_args, unparsed_args = init(300, ['--phis=embedding', '--phis=pmean', '--phis=rnn_encoder'])
    phi_seq, phi_args, unparsed_args = init(300, ['--phis=embedding', '--phis=cnn_encoder', '--phis=rnn_encoder'])
    print(phi_seq.output_size)
    print(phi_seq.trainable)
    print(phi_seq.input_encoding())
    print(phi_seq.output_encoding())
    print(phi_args)

    if Encoding.INDEX not in phi_seq.input_encoding():
        phi_seq.enc_seq.prepend(EmbeddingEncoder(300))

    phi_seq.enc_seq[0].prep_inputs_fn = torch.nn.Embedding.from_pretrained(torch.randn((1000, 300)), freeze=True)
    t = torch.randint(0, 1000, (10, 20))
    sent_lens = torch.randint(1, 21, (10,))
    ex_lens = torch.LongTensor([3, 7])
    inputs = {'test': {'encoding': Encoding.INDEX, 'value': t, 'sent_len': sent_lens, 'ex_len': ex_lens}}

    output = phi_seq(inputs)
    print(output)
