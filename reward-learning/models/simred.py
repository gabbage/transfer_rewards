import argparse
import logging

import numpy as np
import torch

from models.model_api import ModelAPI
from models.phis.encoders import Encoding, DimConstraint


class SimRed(ModelAPI):
    def __init__(self, input_size, num_inputs, numerical_output, gain, alpha):
        super(SimRed, self).__init__()
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.numerical_output = bool(numerical_output)
        self.gain = gain
        self.alpha = alpha
        self.dim_constraint = DimConstraint.BATCH_SEQ_CHANNELS(input_size)

    @staticmethod
    def model_options(model_options=None):
        model_options = argparse.ArgumentParser() if model_options is None else model_options
        model_options.add_argument("--numerical_output", action="store", default=0, type=int,
                                   help="Whether or not the model will produce a numerical output for regression.")
        model_options.add_argument("--gain", action="store", default=1.0, type=float,
                                   help="The gain of the simred score, see 'normalized_simred_score' function.")
        model_options.add_argument("--alpha", action="store", default=0.5, type=float,
                                   help="The similarity is multiplied with alpha and redundancy with 1 - alpha.")
        return model_options

    def normalize(self, t):
        t = t.view(-1, self.input_size)
        n = torch.clamp(torch.norm(t, dim=1), min=1e-10)  # Increase numerical stability by value clamping
        return t / n.view(-1, 1)

    def normalized_sentence_similarity(self, s, a):
        s = self.normalize(s)
        a = self.normalize(a)

        return torch.mean(torch.matmul(s, a.transpose(0, 1)))

    def normalized_sentence_redundancy(self, s):
        s = self.normalize(s)

        if s.size(0) == 1:
            return torch.sum(torch.zeros_like(s) * s)

        sim_mat = torch.matmul(s, s.transpose(0, 1))
        eye = torch.eye(sim_mat.size(0), device=sim_mat.device)
        red = torch.sum((sim_mat - eye) ** 2) / (s.size(0) * s.size(0) - s.size(0))
        return red * 2.0 - 1.0

    def normalized_simred_score(self, s, a):
        sim = self.normalized_sentence_similarity(s, a)
        red = self.normalized_sentence_redundancy(s)
        return self.gain * (self.alpha * sim - (1.0 - self.alpha) * red)

    def forward(self, inputs):
        assert 'article' in inputs and inputs['article']['encoding'] == Encoding.SENTENCE
        assert 'ex_len' in inputs['article']
        assert 'summary' in inputs

        y_preds = []
        summaries = inputs['summary'] if isinstance(inputs['summary'], list) else [inputs['summary']]

        article_values = inputs['article']['value']
        article_ex_len = inputs['article']['ex_len'].tolist()

        for summary in summaries:
            assert summary['encoding'] == Encoding.SENTENCE
            assert 'ex_len' in summary

            summary_values = summary['value']
            summary_ex_len = summary['ex_len'].tolist()

            for s, a in zip(summary_values.split(summary_ex_len), article_values.split(article_ex_len)):
                y_preds.append(self.normalized_simred_score(s, a).view(1))

        y_pred = torch.cat(y_preds).view(len(summaries), len(article_ex_len)).permute(1, 0)

        if self.numerical_output and y_pred.size(1) == 2:
            y_pred = y_pred[:, 0] - y_pred[:, 1]

        return y_pred

    @classmethod
    def init(cls, input_size, num_inputs, unparsed_args=None, *args, **kwargs):
        model_args, unparsed_args = SimRed.model_options().parse_known_args(unparsed_args)

        logging.info('{} model arguments: {}'.format(SimRed.__name__, model_args))
        return SimRed(input_size, num_inputs, model_args.numerical_output, model_args.gain, model_args.alpha), \
            model_args, unparsed_args


init = SimRed.init


def proof_of_concept():
    # Proof of concept
    embedding_size = 10
    # The first summary contains 4 sentence embeddings
    summary_1 = np.random.randn(4, embedding_size)
    # The second summary contains the sentence embeddings from the first summary twice
    summary_2 = np.concatenate([summary_1, summary_1], axis=0)
    # The article contains the sentence embeddings from the first summary and 10 additional sentence embeddings
    article = np.concatenate([summary_1, np.random.randn(10, embedding_size)], axis=0)

    # The similarity with the article should be high, but the redundancy in a summary should be low
    def normalize(t):
        t = np.squeeze(t)
        n = np.linalg.norm(t, axis=1)
        return t / n.reshape(-1, 1)

    def normalized_sentence_similarity(s, a):
        s = normalize(s)
        a = normalize(a)

        sim_mat = s.dot(a.transpose())
        return np.mean(sim_mat)

    # Those scores are equal even if the second summary contains the sentences twice
    sim_1 = normalized_sentence_similarity(summary_1, article)
    sim_2 = normalized_sentence_similarity(summary_2, article)
    print("Sentence similarity of first summary:", sim_1)
    print("Sentence similarity of second summary:", sim_2)

    def normalized_sentence_redundancy(s):
        s = normalize(s)

        if s.shape[0] == 1:
            return 0.0

        sim_mat = s.dot(s.transpose())
        # print("old: ", ((np.sum(sim_mat) - sim_mat.shape[0]) / (sim_mat.size - sim_mat.shape[0])) * 2.0 - 1.0)
        return (np.sum(np.square(sim_mat - np.eye(sim_mat.shape[0]))) / (sim_mat.size - sim_mat.shape[0])) * 2.0 - 1.0

    def normalized_simred_score(s, a, gain=1.0, alpha=0.5):
        sim = normalized_sentence_similarity(s, a)
        red = normalized_sentence_redundancy(s)
        return gain * (alpha * sim + (1 - alpha) * red)

    red_1 = normalized_sentence_redundancy(summary_1)
    red_2 = normalized_sentence_redundancy(summary_2)
    print("Sentence redundancy of first summary:", red_1)
    print("Sentence redundancy of second summary:", red_2)

    # Goal: high similarity with low redundancy
    print("First summary score:", normalized_simred_score(summary_1, article))
    print("Second summary score:", normalized_simred_score(summary_2, article))

    v = np.random.randn(1, embedding_size)
    vm = np.concatenate([v, v, v, v], axis=0)
    print(normalized_sentence_similarity(vm, vm), "-", normalized_sentence_redundancy(vm), "=", normalized_simred_score(vm, -vm, gain=1))


if __name__ == '__main__':
    a = {'encoding': Encoding.SENTENCE, 'value': torch.randn(10, 1, 100), 'ex_len': torch.LongTensor([5, 5])}
    s1 = {'encoding': Encoding.SENTENCE, 'value': torch.randn(5, 1, 100), 'ex_len': torch.LongTensor([2, 3])}
    s2 = {'encoding': Encoding.SENTENCE, 'value': torch.randn(7, 1, 100), 'ex_len': torch.LongTensor([4, 3])}
    model = SimRed(100, 2, 0, 1.0, 0.85)

    batch = {'article': a, 'summary': [s1, s2]}
    print(model(batch))

    batch = {'article': a, 'summary': [s2, s1]}
    print(model(batch))
