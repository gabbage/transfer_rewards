import abc
from flags import Flags


class DimConstraint(object):
    def __init__(self, dim_size_list):
        self.dim_size_list = dim_size_list

    def check(self, tensor):
        nb_dims = len(set([d for d, s in self.dim_size_list]))
        assert len(tensor.size()) == nb_dims, 'Number of dimension has to match'

        fixed_dims = [(d, s) for d, s in self.dim_size_list if s != -1]
        for d, s in fixed_dims:
            assert tensor.size(d) == s, 'Dimension {} should have size {} but actually has size {}'.format(
                d, s, tensor.size(d))

    def is_valid(self, tensor):
        nb_dims = len(set([d for d, s in self.dim_size_list]))

        if len(tensor.size()) == nb_dims:
            fixed_dims = [(d, s) for d, s in self.dim_size_list if s != -1]
            for d, s in fixed_dims:
                if tensor.size(d) != s:
                    return False

            return True
        else:
            return False

    @staticmethod
    def var_size():
        return -1

    @staticmethod
    def BATCH_SEQ_CHANNELS(channels):
        return DimConstraint([(0, -1), (1, -1), (2, channels)])


class Encoding(Flags):
    UNDEFINED = ()
    RAW = ()
    INDEX = ()
    WORD = ()
    SENTENCE = ()
    DOCUMENT = ()


class Encoder(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def input_encoding(self):
        """
        :return: Encoding types this encoder can take as input
        :rtype: Encoding
        """
        return Encoding.UNDEFINED

    @abc.abstractmethod
    def output_encoding(self, input_encoding=None):
        """
        :return: Encoding types this encoder can produce
        :rtype: Encoding
        """
        return Encoding(int(input_encoding) << 1) if input_encoding else Encoding.UNDEFINED


class Index2WordEncoder(Encoder):
    def input_encoding(self):
        return Encoding.INDEX

    def output_encoding(self, input_encoding=None):
        return Encoding.WORD


class Sent2DocEncoder(Encoder):
    def input_encoding(self):
        return Encoding.SENTENCE

    def output_encoding(self, input_encoding=None):
        return Encoding.DOCUMENT


class Word2SentEncoder(Encoder):
    def input_encoding(self):
        return Encoding.WORD

    def output_encoding(self, input_encoding=None):
        return Encoding.SENTENCE


class Word2DocEncoder(Encoder):
    def input_encoding(self):
        return Encoding.WORD

    def output_encoding(self, input_encoding=None):
        return Encoding.DOCUMENT


class EncoderSequence(Encoder, list):
    def __init__(self):
        super(EncoderSequence, self).__init__()
        super(Encoder, self).__init__()

    def input_encoding(self):
        return self[0].input_encoding()

    def output_encoding(self, input_encoding=None):
        if input_encoding is None:
            return self[-1].output_encoding(self[-1].input_encoding())
        else:
            for enc in self:
                assert input_encoding in enc.input_encoding(), \
                    'Input encoding must be one of {}'.format(self.input_encoding())
                output_encoding = enc.output_encoding(input_encoding)
                input_encoding = output_encoding

            return output_encoding

    def append(self, encoder):
        if len(self) == 0:
            super(EncoderSequence, self).append(encoder)
        else:
            assert any([enc in self.output_encoding() for enc in encoder.input_encoding()]), \
                'Last encoder does not provide the encoding that the new encoder requires as input!'
            super(EncoderSequence, self).append(encoder)

    def prepend(self, encoder):
        assert any([enc in self.input_encoding() for enc in encoder.output_encoding()])
        super(EncoderSequence, self).insert(0, encoder)

    def last(self):
        return self[-1]


if __name__ == '__main__':
    enc_seq = EncoderSequence()
    enc_seq.append(Word2SentEncoder())
    enc_seq.append(Sent2DocEncoder())
    print('Encoders:', len(enc_seq))
    print('Input encodings: ', enc_seq.input_encoding())
    print('Output encodings: ', enc_seq.output_encoding())
    print('From ', Encoding.WORD, 'to', enc_seq.output_encoding(Encoding.WORD))
    enc_seq.prepend(Index2WordEncoder())
    print('Encoders:', len(enc_seq))
    print('Input encodings: ', enc_seq.input_encoding())
    print('Output encodings: ', enc_seq.output_encoding())
    print('From ', Encoding.INDEX, 'to', enc_seq.output_encoding(Encoding.INDEX))
