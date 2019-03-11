import argparse
import itertools
import logging
import os
import pickle
from functools import partial

import xxhash

import torch
import torch.nn
from nltk import sent_tokenize
from pytorch_pretrained_bert import BertTokenizer, BertModel

from models.phis.encoders import Encoding
from models.phis.phi_api import PhiAPI
from multiprocessing import Pool


class BertSimRed(torch.nn.Module, PhiAPI):
    def __init__(self, input_size, model_name, cased, sim_pooling, max_samples, cache_dir=None, workers=4, keep_fix=1,
                 device=torch.device('cpu')):
        super(BertSimRed, self).__init__()
        super(torch.nn.Module, self).__init__()

        self.device = device
        self.input_size = input_size
        self.model_name = model_name
        self.cased = bool(cased)
        self.sim_pooling = sim_pooling
        self.max_samples = max_samples
        self.cache_dir = cache_dir

        if cache_dir is not None:
            self.keep_fix = bool(keep_fix)
        else:
            self.keep_fix = True

        self.max_seq_len = 512
        self.pool = Pool(workers)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir='.vector_cache',
                                                       do_lower_case=not self.cased)
        self.add_module('bert', BertModel.from_pretrained(model_name, cache_dir='.vector_cache'))

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

    @staticmethod
    def prepare_summary_article_pairs(summary, article, tokenizer, max_seq_len=512):
        summary_sentences = sent_tokenize(summary)
        article_sentences = sent_tokenize(article)
        summary_sent_tokens = [tokenizer.tokenize(s) for s in summary_sentences]
        article_sent_tokens = [tokenizer.tokenize(a) for a in article_sentences]
        pairs = []

        for i in range(len(summary_sentences)):
            for j in range(len(article_sentences)):
                s_tokens = summary_sent_tokens[i]
                a_tokens = article_sent_tokens[j]

                if len(s_tokens) + len(a_tokens) > 512 - 3:
                    logging.warning('Got more than 512 tokens with summary and article sentence!')

                BertSimRed._truncate_seq_pair(s_tokens, a_tokens, max_seq_len - 3)
                tokens = ['[CLS]'] + s_tokens + ['[SEP]'] + a_tokens + ['[SEP]']

                # Compute the three-typed embedding
                segment_ids = [0] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_len - len(input_ids))
                segment_ids += padding
                input_ids += padding
                input_mask += padding

                pairs.append((input_ids, segment_ids, input_mask))

        return pairs

    @staticmethod
    def prepare_summary_pairs(summary, tokenizer, max_seq_len=512):
        summary_sentences = sent_tokenize(summary)
        summary_sent_tokens = [tokenizer.tokenize(s) for s in summary_sentences]
        pairs = []

        for i, j in itertools.combinations(list(range(len(summary_sentences))), r=2):
            s1_tokens = summary_sent_tokens[i]
            s2_tokens = summary_sent_tokens[j]

            if len(s1_tokens) + len(s2_tokens) > 512 - 3:
                logging.warning('Got more than 512 tokens with summary sentence!')

            BertSimRed._truncate_seq_pair(s1_tokens, s2_tokens, max_seq_len - 3)
            tokens = ['[CLS]'] + s1_tokens + ['[SEP]'] + s2_tokens + ['[SEP]']

            # Compute the three-typed embedding
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_len - len(input_ids))
            input_ids += padding
            segment_ids += padding
            input_mask += padding

            pairs.append((input_ids, segment_ids, input_mask))

        return pairs

    @staticmethod
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def forward(self, inputs):
        # Inputs is a dictionary containing:
        # {field_name: {"encoding": Encoding, "value": str/FloatTensor, "sent_len": LongTensor, "ex_len": LongTensor}}
        assert 'article' in inputs
        assert 'summary' in inputs

        summaries = inputs['summary'] if isinstance(inputs['summary'], list) else [inputs['summary']]
        features = []

        for summary in summaries:
            features.append(self.encode(summary, inputs['article']))

        if len(features) == 1:
            return {'summary': {'value': features[0], 'encoding': Encoding.DOCUMENT}}
        else:
            return {'summary': [{'value': f, 'encoding': Encoding.DOCUMENT} for f in features]}

    def to_tensor(self, values):
        return torch.tensor(values, dtype=torch.long, device=self.device).view((1, -1))

    def encode(self, batch_summary, batch_article):
        assert batch_summary['encoding'] == Encoding.RAW
        assert batch_article['encoding'] == Encoding.RAW

        # Prepare summary - article pairs
        _prep_sa_pairs = partial(BertSimRed.prepare_summary_article_pairs, tokenizer=self.tokenizer,
                                 max_seq_len=self.max_seq_len)
        batch_sa_pairs = self.pool.starmap(_prep_sa_pairs, zip(batch_summary['value'], batch_article['value']))

        # Prepare summary sentence pairs
        _prep_ss_pairs = partial(BertSimRed.prepare_summary_pairs, tokenizer=self.tokenizer,
                                 max_seq_len=self.max_seq_len)
        batch_ss_pairs = self.pool.map(_prep_ss_pairs, batch_summary['value'])

        batch_sa_vecs = self.process_pairs(batch_sa_pairs, batch_summary, batch_article, 'sa')
        batch_ss_vecs = self.process_pairs(batch_ss_pairs, batch_summary, batch_article, 'ss')

        return torch.cat((torch.cat(tuple(batch_sa_vecs)), torch.cat(tuple(batch_ss_vecs))), dim=1)

    def process_pairs(self, batch_pairs, batch_summary, batch_article, cache_subdir):
        batch_vecs = []
        ctx = torch.no_grad() if self.keep_fix else torch.enable_grad()

        for pairs, summary, article in zip(batch_pairs, batch_summary['value'], batch_article['value']):
            if len(pairs) > 0:
                sim_vec = None
                cache_file = None

                if self.cache_dir:
                    cache_key = xxhash.xxh64(summary + article).hexdigest()
                    cache_file = os.path.join(self.cache_dir, cache_subdir, '{}.p'.format(cache_key))

                    if os.path.isfile(cache_file):
                        sim_vec = pickle.load(open(cache_file, 'rb'))

                if sim_vec is None:
                    with ctx:
                        sim_mat = []

                        for chunk in BertSimRed.chunks(pairs, self.max_samples):
                            input_ids = torch.cat(tuple([self.to_tensor(i) for i, _, _ in chunk]))
                            segment_ids = torch.cat(tuple([self.to_tensor(s) for _, s, _ in chunk]))
                            input_mask = torch.cat(tuple([self.to_tensor(m) for _, _, m in chunk]))

                            _, sim_mat_chunk = self.bert(input_ids, segment_ids, input_mask,
                                                         output_all_encoded_layers=False)

                            sim_mat.append(sim_mat_chunk)

                        sim_mat = torch.cat(tuple(sim_mat))

                        # Apply pooling
                        if self.sim_pooling.lower() == 'mean':
                            sim_vec = torch.mean(sim_mat, dim=(0,))
                        elif self.sim_pooling.lower() == 'max':
                            sim_vec = torch.max(sim_mat, dim=0)
                        else:
                            raise ValueError(
                                "Wrong BERT similarity matrix pooling specified: {}".format(self.sim_pooling))

                if cache_file is not None and not os.path.isfile(cache_file):
                    if not os.path.exists(os.path.dirname(cache_file)):
                        os.makedirs(os.path.dirname(cache_file))

                    pickle.dump(sim_vec, open(cache_file, 'wb'))

                batch_vecs.append(sim_vec.view((1, self.bert.config.hidden_size)))
            else:
                batch_vecs.append(torch.zeros((1, self.bert.config.hidden_size), device=self.device))

        return batch_vecs

    @property
    def trainable(self):
        return True

    @property
    def output_size(self):
        return self.bert.config.hidden_size * 2

    def input_encoding(self):
        return Encoding.RAW

    def output_encoding(self, input_encoding=None):
        return Encoding.DOCUMENT

    @property
    def input_field_keys(self):
        return ['summary', 'article']

    @property
    def output_field_keys(self):
        return ['summary']

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
        phi_options.add_argument('--bert_sim_pooling', action='store', default='mean', type=str,
                                 help='The type of pooling that is applied.')
        phi_options.add_argument('--bert_max_samples', action='store', default=32, type=int,
                                 help='The number of samples that are passed through the BERT model at one time.')
        phi_options.add_argument('--bert_cache_dir', action='store', default=None, type=str,
                                 help='Cache directory where precomputed simred matrices can be stored.')
        phi_options.add_argument('--bert_workers', action='store', default=4, type=int,
                                 help='The number of workers used to tokenize and prepare inputs.')
        phi_options.add_argument('--bert_keep_fix', action='store', default=1, type=int,
                                 help='Whether or not to train the bert model.')

        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        phi_args, unparsed_args = BertSimRed.phi_options().parse_known_args(unparsed_args)

        # Add the bert model to unparsed arguments again to forward this argument to the model
        # unparsed_args.append("--bert_model={}".format(phi_args.bert_model))

        logging.info("{} phi arguments: {}".format(BertSimRed.__name__, phi_args))
        return BertSimRed(input_size, phi_args.bert_model, phi_args.bert_cased, phi_args.bert_sim_pooling,
                          phi_args.bert_max_samples, phi_args.bert_cache_dir, phi_args.bert_workers,
                          phi_args.bert_keep_fix, kwargs['device']), phi_args, unparsed_args


init = BertSimRed.init
