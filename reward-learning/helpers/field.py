import re

import six
import torch
from nltk import word_tokenize, sent_tokenize
from torchtext.data import Field, Example, RawField

from models.phis.encoders import Encoding


class RawTextField(RawField):
    def __init__(self, is_target=False, lower=False, actual_input=True):
        super(RawTextField, self).__init__()
        self.is_target = is_target
        self.lower = lower
        self.use_vocab = False
        self.actual_input = actual_input

        if self.lower:
            self.preprocessing = six.text_type.lower

    def process(self, batch, *args, **kwargs):
        """ Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
            postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)

        return {"value": batch, "encoding": Encoding.RAW, "field": self}


class TextField(Field):
    def __init__(self, **kwargs):
        kwargs["sequential"] = True
        kwargs["use_vocab"] = True
        kwargs["include_lengths"] = True
        kwargs["batch_first"] = True
        kwargs["tokenize"] = word_tokenize
        super(TextField, self).__init__(**kwargs)

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        val, sentence_length = super(Field, self).process(batch, device)
        return {"value": val, "sent_len": sentence_length, "encoding": Encoding.INDEX, "field": self}


class FloatField(Field):
    def __init__(self, **kwargs):
        if 'actual_input' in kwargs:
            self.actual_input = kwargs.pop('actual_input')
        else:
            if 'is_target' in kwargs and kwargs['is_target']:
                self.actual_input = False
            else:
                self.actual_input = True

        kwargs["sequential"] = False
        kwargs["use_vocab"] = False
        kwargs["batch_first"] = True

        if 'fp16' in kwargs and kwargs.pop('fp16'):
            kwargs["dtype"] = torch.float16
        else:
            kwargs["dtype"] = torch.float

        super(FloatField, self).__init__(**kwargs)


class FloatTargetField(FloatField):
    def __init__(self, **kwargs):
        kwargs["actual_input"] = False
        kwargs["is_target"] = True
        super(FloatTargetField, self).__init__(**kwargs)


class ClassIndexTargetField(Field):
    def __init__(self, **kwargs):
        self.actual_input = False
        kwargs["sequential"] = False
        kwargs["use_vocab"] = False
        kwargs["include_lengths"] = False
        kwargs["batch_first"] = True
        kwargs["is_target"] = True
        kwargs["unk_token"] = None
        kwargs["pad_token"] = None
        kwargs["preprocessing"] = self.convert

        if 'lookup' in kwargs:
            self.lookup = kwargs.pop('lookup')
        else:
            self.lookup = None

        super(ClassIndexTargetField, self).__init__(**kwargs)

    def convert(self, x):
        if self.lookup is None:
            return x
        elif isinstance(x, list):
            return [self.lookup[i] for i in x]
        else:
            return self.lookup[x]

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
            device (torch.device): The device to create the tensor on
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        return torch.tensor(batch, dtype=self.dtype, device=device)


class SentenceWrappedField(Field):
    def __init__(self, **kwargs):
        if 'vocab' in kwargs:
            self.vocab = kwargs.pop('vocab')

        if 'actual_input' in kwargs:
            self.actual_input = kwargs.pop('actual_input')
        else:
            if 'is_target' in kwargs and kwargs['is_target']:
                self.actual_input = False
            else:
                self.actual_input = True

        if 'fp16' in kwargs and kwargs.pop('fp16'):
            kwargs["dtype"] = torch.float16

        kwargs["sequential"] = True
        kwargs["use_vocab"] = True
        kwargs["include_lengths"] = True
        kwargs["batch_first"] = True
        # DO NOT USE THIS:
        # self.tokenize_fn = kwargs["tokenize"] if "tokenize" in kwargs else lambda x: \
        #   word_tokenize(re.sub("'", " ' ", x))
        # USE ONE OF THOSE:
        self.tokenize_fn = kwargs["tokenize"] if "tokenize" in kwargs else lambda x: \
            word_tokenize(re.sub("( '|' )", " ' ", x))
        # self.tokenize_fn = kwargs["tokenize"] if "tokenize" in kwargs else word_tokenize
        kwargs["preprocessing"] = self.tokenize_sentences
        kwargs["tokenize"] = sent_tokenize

        if "eos_token" in kwargs:
            self.eos = [kwargs.pop("eos_token")]
        else:
            self.eos = []

        if "bos_token" in kwargs:
            self.bos = [kwargs.pop("bos_token")]
        else:
            self.bos = []

        super(SentenceWrappedField, self).__init__(**kwargs)

    def tokenize_sentences(self, sentences):
        # TODO Wrapping the tokenized sentence in <s> and </s> is necessary for InferSent
        return [self.bos + self.tokenize_fn(s) + self.eos for s in sentences]

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        example_lengths = torch.tensor([len(ex) for ex in batch], dtype=self.dtype, device=device)
        padded = self.pad([sent for ex in batch for sent in ex])
        val, sentence_length = self.numericalize(padded, device=device)
        return {"value": val, "sent_len": sentence_length, "ex_len": example_lengths, "encoding": Encoding.INDEX,
                "field": self}


class LazyExample(object):

    @classmethod
    def fromlist(cls, data, fields, lazy=None):
        if lazy is None:
            lazy = [False] * len(data)

        ex = Example()
        for (name, field), val, is_lazy in zip(fields, data, lazy):
            if field is not None:
                if isinstance(val, six.string_types):
                    val = val.rstrip('\n')
                # Handle field tuples
                if isinstance(name, tuple):
                    for n, f in zip(name, field):
                        if is_lazy:
                            setattr(ex, n + "_lazyfn", f.preprocess)
                            setattr(ex, n, val)
                        else:
                            setattr(ex, n, f.preprocess(val))
                else:
                    if is_lazy:
                        setattr(ex, name + "_lazyfn", field.preprocess)
                        setattr(ex, name, val)
                    else:
                        setattr(ex, name, field.preprocess(val))
        return ex

    def __getattribute__(self, name):
        if name.endswith("_lazyfn"):
            value = super(LazyExample, self).__getattribute__(name)
        else:
            value = super(LazyExample, self).__getattribute__(name)

            if hasattr(self, name + "_lazyfn"):
                # Fetch the result of the lazy function only once
                prep_fn = getattr(self, name + "_lazyfn")
                value = prep_fn(value)
                setattr(self, name, value)
                delattr(self, name + "_lazyfn")

        return value
