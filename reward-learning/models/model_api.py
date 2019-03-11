import abc
from torch.nn import Module


class ModelAPI(Module):
    def __init__(self):
        super(ModelAPI, self).__init__()
        super(Module, self).__init__()
        self.prep_inputs_fn = None
        self.prep_targets_fn = None
        self.prep_outputs_fn = None

    @abc.abstractmethod
    def forward(self, *x):
        pass

    @staticmethod
    @abc.abstractmethod
    def model_options(model_options=None):
        pass

    @classmethod
    @abc.abstractmethod
    def init(cls, input_size, num_inputs, unparsed_args=None, *args, **kwargs):
        pass
