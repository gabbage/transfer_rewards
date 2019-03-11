import argparse
import logging
from functools import partial
from multiprocessing import Pool

import torch
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from helpers.metrics import rouge, js, redundancy, tfidf
from models.phis.encoders import Encoding
from models.phis.phi_api import PhiAPI


class Metrics(PhiAPI):
    def __init__(self, input_size, metrics=None, language='english', device=torch.device('cpu'), num_workers=4):
        super(Metrics, self).__init__()
        self.device = device
        self.language = language
        self.input_size = input_size
        self.metrics = metrics
        self.stopwords = set(stopwords.words(self.language))
        self.stemmer = PorterStemmer()
        self.workers = Pool(num_workers)
        self.computable_metrics = ["rouge", "js", "redundancy", "tfidf"]

    def __call__(self, inputs):
        assert 'summary' in inputs
        assert 'article' in inputs

        articles = inputs['article']
        batch_size = len(articles['value'])
        summaries = inputs['summary'] if isinstance(inputs['summary'], list) else [inputs['summary']]
        features = []

        for j, summary in enumerate(summaries):
            summary_features = torch.zeros(batch_size, len(self.metrics))

            for i, metric in enumerate(self.metrics):
                if metric in inputs:
                    values = inputs[metric][j] if isinstance(inputs[metric], list) else inputs[metric]

                    summary_features[:, i] = torch.squeeze(values)
                else:
                    assert metric.split('-')[0] in self.computable_metrics
                    values = None

                    if 'rouge' in metric:
                        _, n = metric.split('-')
                        _rouge = partial(rouge, n=int(n), stemmer=self.stemmer, stopwords=self.stopwords,
                                         language=self.language)
                        values = self.workers.starmap(_rouge, zip(articles['value'], summary['value']))
                    elif 'js' == metric:
                        _js = partial(js, stemmer=self.stemmer, stopwords=self.stopwords, language=self.language)
                        values = self.workers.starmap(_js, zip(articles['value'], summary['value']))
                    elif 'redundancy' in metric:
                        _, n = metric.split('-')
                        _redundancy = partial(redundancy, n=int(n), stemmer=self.stemmer, language=self.language)
                        values = self.workers.map(_redundancy, summary['value'])
                    elif 'tfidf' in metric:
                        _, n = metric.split('-')
                        _tfidf = partial(tfidf, reductions=n, language=self.language)
                        values = self.workers.starmap(_tfidf, zip(articles['value'], summary['value']))

                    if values is None:
                        values = torch.zeros((batch_size, 1))
                    else:
                        values = torch.tensor(values, dtype=torch.float)

                    assert torch.squeeze(values).size(dim=0) == batch_size

                    summary_features[:, i] = torch.squeeze(values)

            features.append(summary_features.to(self.device))

        if len(features) == 1:
            return {'summary': {'value': features[0], 'encoding': Encoding.DOCUMENT}}
        else:
            return {'summary': [{'value': f, 'encoding': Encoding.DOCUMENT} for f in features]}

    @property
    def trainable(self):
        return False

    @property
    def output_size(self):
        return len(self.metrics)

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
        phi_options.add_argument('--metric_features', action='append', default=None, type=str,
                                 help='A list of metrics that should be picked from precomputed values or computed '
                                      'on the fly.')
        phi_options.add_argument('--metric_language', action='store', default='english', type=str,
                                 help='String representation of the language that the summary/article are in.')

        return phi_options

    @classmethod
    def init(cls, input_size, unparsed_args=None, *args, **kwargs):
        # The parameter input_dim is the vector length of the embeddings, in case of tf*idf no embeddings are used
        phi_args, unparsed_args = Metrics.phi_options().parse_known_args(unparsed_args)

        if phi_args.metric_features is None:
            phi_args.metric_features = ["rouge-1", "rouge-2", "js", "redundancy_1_2", "tfidf_cos_avg"]

        logging.info("{} phi arguments: {}".format(Metrics.__name__, phi_args))
        return Metrics(input_size, phi_args.metric_features, phi_args.metric_language, kwargs['device']), \
               phi_args, unparsed_args


init = Metrics.init
