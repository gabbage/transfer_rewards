import argparse
import logging
import torch
import torch.nn

from nltk import sent_tokenize
from pytorch_pretrained_bert import BertTokenizer, BertModel

from models.phis.encoders import Encoding
from models.phis.phi_api import PhiAPI


class Bert(PhiAPI):
    def __init__(self, input_size, bert_model, bert_cased, device):
        super(Bert, self).__init__()
        self.device = device
        self.input_size = input_size
        self.bert_model = bert_model
        self.bert_cased = bool(bert_cased)
        self.max_sequence_length = 512
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, cache_dir='.vector_cache',
                                                       do_lower_case=not self.bert_cased)
        self.model = BertModel.from_pretrained(bert_model, cache_dir='.vector_cache')
        self.model.to(device)

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def __call__(self, inputs):
        # Inputs is a dictionary containing:
        # {field_name: {"encoding": Encoding, "value": str/FloatTensor, "sent_len": LongTensor, "ex_len": LongTensor}}
        for field_name, field in inputs.items():
            if isinstance(field, list):
                for f in field:
                    self.encode(f)
            else:
                self.encode(field)

        return inputs

    @property
    def trainable(self):
        return False

    @property
    def output_size(self):
        return self.model.config.hidden_size

    @property
    def input_field_keys(self):
        return None

    @property
    def output_field_keys(self):
        return None

    @staticmethod
    def phi_options(phi_options=None):
        phi_options = argparse.ArgumentParser() if phi_options is None else phi_options
        phi_options.add_argument('--bert_model', action='store', default='bert-base-cased', type=str,
                                 help='Bert pre-trained model selected in the list: bert-base-uncased, '
                                      'bert-large-uncased, bert-base-cased, bert-large-cased, '
                                      'bert-base-multilingual-uncased, '
                                      'bert-base-multilingual-cased, bert-base-chinese.')
        phi_options.add_argument('--bert_cased', action='store', default=1, type=int,
                                 help='Whether or not the tokenizer is using original case words or lower cased ones.')
        phi_options.add_argument('--bert_sentence_wise', action='store', default=1, type=int,
                                 help='Whether or not the BERT tokenizer will be applied to each sentence or the '
                                      'whole text.')

        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        phi_args, unparsed_args = Bert.phi_options().parse_known_args(unparsed_args)

        # Add the bert model to unparsed arguments again to forward this argument to the model
        unparsed_args.append("--bert_model={}".format(phi_args.bert_model))

        logging.info("{} phi arguments: {}".format(Bert.__name__, phi_args))
        return Bert(input_size, phi_args.bert_model, phi_args.bert_cased, kwargs['device']), phi_args, unparsed_args

    def input_encoding(self):
        return Encoding.RAW

    def output_encoding(self, input_encoding=None):
        return Encoding.WORD

    def encode(self, field):
        assert field['encoding'] in self.input_encoding(), 'Encoder does not accept {}'.format(field['encoding'])

        raw_texts = field.pop('value')
        encs = []
        ex_len = []

        with torch.no_grad():
            for raw_text in raw_texts:
                sentences = sent_tokenize(raw_text)

                ex_len.append(len(sentences))
                ex_input_ids = []
                ex_segment_ids = []
                ex_input_mask = []

                for sentence in sentences:
                    tokens = self.tokenizer.tokenize(sentence)

                    if len(tokens) > 512 - 2:
                        logging.warning('One sentence was longer than 512 tokens that can be passed to BERT!')
                        tokens = tokens[:(self.max_sequence_length - 2)]

                    tokens = ['[CLS]'] + tokens + ['[SEP]']
                    segment_ids = [0] * len(tokens)

                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                    # The mask has 1 for real tokens and 0 for padding tokens. Only real
                    # tokens are attended to.
                    input_mask = [1] * len(input_ids)

                    # Zero-pad up to the sequence length.
                    padding = [0] * (self.max_sequence_length - len(input_ids))
                    input_ids += padding
                    input_mask += padding
                    segment_ids += padding

                    assert len(input_ids) == self.max_sequence_length
                    assert len(input_mask) == self.max_sequence_length
                    assert len(segment_ids) == self.max_sequence_length

                    ex_input_ids.append(torch.tensor(input_ids, dtype=torch.long, device=self.device).view((1, -1)))
                    ex_segment_ids.append(torch.tensor(segment_ids, dtype=torch.long, device=self.device).view((1, -1)))
                    ex_input_mask.append(torch.tensor(input_mask, dtype=torch.long, device=self.device).view((1, -1)))

                    # _, pooled_output = self.model(
                    #     torch.tensor(input_ids, dtype=torch.long, device=self.device).view((1, -1)),
                    #     torch.tensor(segment_ids, dtype=torch.long, device=self.device).view((1, -1)),
                    #     torch.tensor(input_mask, dtype=torch.long, device=self.device).view((1, -1)),
                    #     output_all_encoded_layers=False)
                    # encs.append(pooled_output)

                _, batch_encodings = self.model(torch.cat(tuple(ex_input_ids)),
                                                torch.cat(tuple(ex_segment_ids)),
                                                torch.cat(tuple(ex_input_mask)), output_all_encoded_layers=False)

                encs.append(batch_encodings.view((-1, 1, self.output_size)))

        field['value'] = torch.cat(tuple(encs), dim=0).view((-1, 1, self.output_size))
        field['ex_len'] = torch.tensor(ex_len, dtype=torch.long, device=self.device)
        field['encoding'] = Encoding.SENTENCE


init = Bert.init
