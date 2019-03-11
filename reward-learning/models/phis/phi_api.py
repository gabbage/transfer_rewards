import abc

from models.phis.encoders import Encoder


class PhiAPI(Encoder):
    def __init__(self):
        super(PhiAPI, self).__init__()
        super(Encoder, self).__init__()
        self.prep_inputs_fn = None

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def trainable(self):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def input_field_keys(self):
        pass

    @property
    @abc.abstractmethod
    def output_field_keys(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def phi_options(phi_options=None):
        pass

    @classmethod
    @abc.abstractmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        pass
