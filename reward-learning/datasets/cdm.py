import glob
import itertools
import logging
import math
import os
import pickle
import time
from collections import Counter, OrderedDict
from enum import Enum

import numpy as np
import torch
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab, GloVe
from tqdm import tqdm

from helpers.field import ClassIndexTargetField, FloatField, RawTextField, SentenceWrappedField, FloatTargetField
from scorer.data_helper.json_reader import readScores, readArticleRefs


class CDMRankingType(Enum):
    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3


class Pointwise(object):
    def __init__(self, input_fields, target_fields, label_key, device=torch.device("cpu")):
        self.input_fields = input_fields
        self.target_fields = target_fields
        self.label_key = label_key
        self.device = device

    def __call__(self, *args, **kwargs):
        sample_ids = kwargs.pop('sample_ids')
        dataset = kwargs.pop('dataset')
        ranking_type = kwargs.pop('ranking_type')

        assert ranking_type == CDMRankingType.POINTWISE, "The ranking type has to be 'POINTWISE'!"

        article_ids = []
        samples = []

        for sid in sample_ids:
            k_split = sid.split("-")
            article_id = k_split[0]

            if len(k_split) == 1:
                samples.append(dataset.ref_summaries[article_id])
            if len(k_split) > 1:
                samples.append(dataset.samples[article_id][sid])

            article_ids.append(article_id)

        articles = [dataset.articles[a] for a in article_ids]
        batch = {}

        for field_name, field in self.input_fields.items():
            if field_name == 'article':
                batch[field_name] = field.process([field.preprocess(a) for a in articles], self.device)
            else:
                batch[field_name] = field.process([field.preprocess(s[field_name]) for s in samples], self.device)

        for field_name, field in self.target_fields.items():
            if field_name == 'label':
                batch[field_name] = field.process([field.preprocess(s[self.label_key]) for s in samples], self.device)
            else:
                batch[field_name] = field.process([field.preprocess(s[field_name]) for s in samples], self.device)

        batch['input_fields'] = self.input_fields
        batch['target_fields'] = self.target_fields

        return batch


class PairwisePreference(object):
    def __init__(self, input_fields, target_fields, preference_key, device=torch.device("cpu")):
        self.input_fields = input_fields
        self.target_fields = target_fields
        self.preference_key = preference_key
        self.device = device

    def __call__(self, *args, **kwargs):
        sample_ids = kwargs.pop('sample_ids')
        dataset = kwargs.pop('dataset')
        ranking_type = kwargs.pop('ranking_type')

        assert ranking_type == CDMRankingType.PAIRWISE, "The ranking type has to be 'PAIRWISE'!"

        article_ids = []
        sample_columns = OrderedDict()

        for row in sample_ids:
            assert len(row) == 2

            for col, _id in enumerate(row):
                if col not in sample_columns:
                    sample_columns.update({col: []})

                k_split = _id.split("-")
                article_id = k_split[0]

                if len(k_split) == 1:
                    sample_columns[col].append(dataset.ref_summaries[article_id])
                if len(k_split) > 1:
                    sample_columns[col].append(dataset.samples[article_id][_id])

            article_ids.append(article_id)

        article = [dataset.articles[a] for a in article_ids]
        batch = {}

        if len(sample_ids) > 0:
            for field_name, field in self.input_fields.items():
                if field_name == 'article':
                    # Article is provided only once
                    batch[field_name] = field.process([field.preprocess(s) for s in article], self.device)
                else:
                    values = []
                    for col, samples in sample_columns.items():
                        values.append(field.process([field.preprocess(s[field_name]) for s in samples], self.device))

                    batch[field_name] = values

            for field_name, field in self.target_fields.items():
                if field_name == 'label':
                    values = []
                    for col, samples in sample_columns.items():
                        values.append(np.asarray([s[self.preference_key] for s in samples]))

                    if self.preference_key == 'swap_count':
                        labels = np.where(values[0] < values[1], 1, -1)
                    else:
                        labels = np.where(values[0] > values[1], 1, -1)

                    batch[field_name] = field.process([field.preprocess(l) for l in labels], self.device)
                else:
                    values = []
                    for col, samples in sample_columns.items():
                        values.append(field.process([field.preprocess(s[field_name]) for s in samples], self.device))

                    batch[field_name] = values

            batch['input_fields'] = self.input_fields
            batch['target_fields'] = self.target_fields
        else:
            batch = None

        return batch


class ClassicIterator(object):
    def __init__(self, dataset, article_ids, batch_size, example_fn, shuffle=True,
                 ranking_type=CDMRankingType.PAIRWISE):
        self.dataset = dataset
        self.ranking_type = ranking_type
        self.ids = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.example_fn = example_fn

        for article_id in article_ids:
            if ranking_type == CDMRankingType.POINTWISE:
                if article_id in dataset.samples:
                    self.ids.extend(list(dataset.samples[article_id].keys()))

                if article_id in dataset.ref_summaries:
                    self.ids.append(article_id)
            elif ranking_type == CDMRankingType.PAIRWISE:
                if article_id in dataset.samples:
                    nb_samples = len(dataset.samples[article_id])
                else:
                    nb_samples = 0

                if nb_samples >= 1 and article_id in dataset.ref_summaries:
                    self.ids.extend(list(dataset.samples[article_id].keys()))
            elif ranking_type == CDMRankingType.LISTWISE:
                if article_id in dataset.samples:
                    nb_samples = len(dataset.samples[article_id])
                else:
                    nb_samples = 0

                if article_id in dataset.ref_summaries:
                    nb_samples += 1

                if nb_samples > 0:
                    self.ids.append(article_id)

        self.ids = sorted(self.ids)
        self.total_samples = len(self.ids)

    @property
    def article_ids(self):
        if self.ranking_type == CDMRankingType.POINTWISE or self.ranking_type == CDMRankingType.PAIRWISE:
            return list(set([i.split('-')[0] for i in self.ids]))
        elif self.ranking_type == CDMRankingType.LISTWISE:
            return self.ids

    def __len__(self):
        return math.ceil(self.total_samples / self.batch_size)

    def __iter__(self):
        for i in np.arange(0, self.total_samples, self.batch_size):
            if i == 0 and self.shuffle:
                self.ids = sorted(self.ids)
                np.random.shuffle(self.ids)

            batch_ids = self.ids[i:i + self.batch_size]
            sample_ids = []

            if self.ranking_type == CDMRankingType.POINTWISE:
                sample_ids = batch_ids
            elif self.ranking_type == CDMRankingType.PAIRWISE:
                pos_of_ref = np.random.randint(0, 2, len(batch_ids))
                sample_ids = [[i.split("-")[0], i] if p == 0 else [i, i.split("-")[0]]
                              for i, p in zip(batch_ids, pos_of_ref)]
            elif self.ranking_type == CDMRankingType.LISTWISE:
                for article_id in batch_ids:
                    sample_list = []
                    if article_id in self.dataset.samples:
                        sample_list.extend(list(self.dataset.samples[article_id].keys()))

                    if article_id in self.dataset.ref_summaries:
                        sample_list.append(article_id)

                    sample_ids.append(sample_list)

            batch = self.example_fn(sample_ids=sample_ids, dataset=self.dataset, ranking_type=self.ranking_type)

            if batch is not None:
                yield batch


class PermutationsIterator(object):
    def __init__(self, dataset, article_ids, batch_size, example_fn, shuffle=True):
        self.dataset = dataset
        self.ids = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.example_fn = example_fn
        self.total_samples = 0
        self.ranking_type = CDMRankingType.PAIRWISE

        for article_id in article_ids:
            samples_list = []

            if article_id in dataset.samples:
                samples_list.extend(list(dataset.samples[article_id].keys()))

            if article_id in dataset.ref_summaries:
                samples_list.append(article_id)

            permutations = list(itertools.permutations(samples_list, r=2))
            self.ids.extend(permutations)

        self.ids = sorted(self.ids)
        self.total_samples += len(self.ids)

    @property
    def article_ids(self):
        article_ids = []

        for i, j in self.ids:
            article_ids.append(i.split('-')[0])
            article_ids.append(j.split('-')[0])

        return list(set(article_ids))

    def __len__(self):
        return math.ceil(self.total_samples / self.batch_size)

    def __iter__(self):
        for i in np.arange(0, self.total_samples, self.batch_size):
            if i == 0 and self.shuffle:
                self.ids = sorted(self.ids)
                np.random.shuffle(self.ids)

            batch_ids = self.ids[i:i + self.batch_size]
            sample_ids = [list(t) for t in batch_ids]

            yield self.example_fn(sample_ids=sample_ids, dataset=self.dataset, ranking_type=self.ranking_type)


class SwapCombinationsIterator(object):
    def __init__(self, dataset, article_ids, batch_size, example_fn, shuffle=True, min_gap=1):
        self.dataset = dataset
        self.ids = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.example_fn = example_fn
        self.total_samples = 0
        self.ranking_type = CDMRankingType.PAIRWISE

        for article_id in article_ids:
            samples_list = []

            if article_id in dataset.samples:
                samples_list.extend(list(dataset.samples[article_id].keys()))

            if article_id in dataset.ref_summaries:
                samples_list.append(article_id)

            combinations = []

            for s1, s2 in itertools.combinations(samples_list, r=2):
                s1_split = s1.split('-')
                s2_split = s2.split('-')

                if len(s1_split) == 3 and len(s2_split) == 3:
                    if np.abs(int(s1_split[1]) - int(s2_split[1])) >= min_gap:
                        combinations.append((s1, s2))
                else:
                    swap_count = int(s1_split[1]) if len(s1_split) == 3 else int(s2_split[1])

                    if swap_count >= min_gap:
                        combinations.append((s1, s2))

            self.ids.extend(combinations)

        self.ids = sorted(self.ids)
        self.total_samples += len(self.ids)

    @property
    def article_ids(self):
        article_ids = []

        for i, j in self.ids:
            article_ids.append(i.split('-')[0])
            article_ids.append(j.split('-')[0])

        return list(set(article_ids))

    def __len__(self):
        return math.ceil(self.total_samples / self.batch_size)

    def __iter__(self):
        for i in np.arange(0, self.total_samples, self.batch_size):
            if i == 0 and self.shuffle:
                self.ids = sorted(self.ids)
                np.random.shuffle(self.ids)

            batch_ids = self.ids[i:i + self.batch_size]
            pos_of_ref = np.random.randint(0, 2, len(batch_ids))
            sample_ids = [list(t) if p == 0 else list(reversed(t)) for t, p in zip(batch_ids, pos_of_ref)]

            yield self.example_fn(sample_ids=sample_ids, dataset=self.dataset, ranking_type=self.ranking_type)


class CombinationsIterator(object):
    def __init__(self, dataset, article_ids, batch_size, example_fn, shuffle=True):
        self.dataset = dataset
        self.ids = []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.example_fn = example_fn
        self.total_samples = 0
        self.ranking_type = CDMRankingType.PAIRWISE

        for article_id in article_ids:
            samples_list = []

            if article_id in dataset.samples:
                samples_list.extend(list(dataset.samples[article_id].keys()))

            if article_id in dataset.ref_summaries:
                samples_list.append(article_id)

            combinations = list(itertools.combinations(samples_list, r=2))
            self.ids.extend(combinations)

        self.ids = sorted(self.ids)
        self.total_samples += len(self.ids)

    @property
    def article_ids(self):
        article_ids = []

        for i, j in self.ids:
            article_ids.append(i.split('-')[0])
            article_ids.append(j.split('-')[0])

        return list(set(article_ids))

    def __len__(self):
        return math.ceil(self.total_samples / self.batch_size)

    def __iter__(self):
        for i in np.arange(0, self.total_samples, self.batch_size):
            if i == 0 and self.shuffle:
                self.ids = sorted(self.ids)
                np.random.shuffle(self.ids)

            batch_ids = self.ids[i:i + self.batch_size]
            pos_of_ref = np.random.randint(0, 2, len(batch_ids))
            sample_ids = [list(t) if p == 0 else list(reversed(t)) for t, p in zip(batch_ids, pos_of_ref)]

            yield self.example_fn(sample_ids=sample_ids, dataset=self.dataset, ranking_type=self.ranking_type)


class RandomIterator(object):
    def __init__(self, dataset, article_ids, batch_size, example_fn, ranking_type=CDMRankingType.PAIRWISE,
                 all_articles_once=True, min_gap=1):
        self.dataset = dataset
        self.ranking_type = ranking_type
        self.ids = []
        self.batch_size = batch_size
        self.example_fn = example_fn
        self.all_articles_once = all_articles_once
        self.min_gap = min_gap

        for article_id in article_ids:
            if ranking_type == CDMRankingType.POINTWISE:
                if article_id in dataset.samples:
                    self.ids.extend(list(dataset.samples[article_id].keys()))

                if article_id in dataset.ref_summaries:
                    self.ids.append(article_id)
            elif ranking_type == CDMRankingType.PAIRWISE or ranking_type == CDMRankingType.LISTWISE:
                if article_id in dataset.samples:
                    nb_samples = len(dataset.samples[article_id])
                else:
                    nb_samples = 0

                if nb_samples >= 1 and article_id in dataset.ref_summaries:
                    self.ids.append(article_id)

        self.ids = sorted(self.ids)
        self.total_samples = len(self.ids)

    @property
    def article_ids(self):
        if self.ranking_type == CDMRankingType.POINTWISE:
            return list(set([i.split('-')[0] for i in self.ids]))
        elif self.ranking_type == CDMRankingType.PAIRWISE or self.ranking_type == CDMRankingType.LISTWISE:
            return self.ids

    def __len__(self):
        return math.ceil(self.total_samples / self.batch_size)

    def __iter__(self):
        for i in np.arange(0, self.total_samples, self.batch_size):
            if i == 0:
                self.ids = sorted(self.ids)
                np.random.shuffle(self.ids)

            batch_ids = self.ids[i:i + self.batch_size]
            sample_ids = []

            if self.ranking_type == CDMRankingType.POINTWISE:
                sample_ids = batch_ids
            elif self.ranking_type == CDMRankingType.PAIRWISE:
                for article_id in batch_ids:
                    swaps_set = {0, 1, 2, 3, 4, 5}
                    swaps = list(swaps_set)
                    np.random.shuffle(swaps)
                    k1 = None

                    for swap_count_1 in swaps:
                        if swap_count_1 == 0:
                            k1 = article_id
                            break

                        possible_keys = [k for k, s in self.dataset.samples[article_id].items()
                                         if s['swap_count'] == swap_count_1]

                        if len(possible_keys) > 0:
                            k1 = np.random.choice(possible_keys)
                            break

                    assert k1 is not None

                    invalid_swaps = list(range(swap_count_1 - self.min_gap + 1, swap_count_1 + self.min_gap))
                    swaps = list(swaps_set - set(invalid_swaps))
                    np.random.shuffle(swaps)
                    k2 = None

                    for swap_count_2 in swaps:
                        if swap_count_2 == 0:
                            k2 = article_id
                            break

                        possible_keys = [k for k, s in self.dataset.samples[article_id].items()
                                         if s['swap_count'] == swap_count_2]

                        if len(possible_keys) > 0:
                            k2 = np.random.choice(possible_keys)
                            break

                    if k2 is not None:
                        # assert k2 is not None
                        sample_ids.append([k1, k2])
            elif self.ranking_type == CDMRankingType.LISTWISE:
                for article_id in batch_ids:
                    sample_list = []
                    if article_id in self.dataset.samples:
                        sample_list.extend(list(self.dataset.samples[article_id].keys()))

                    if article_id in self.dataset.ref_summaries:
                        sample_list.append(article_id)

                    sample_ids.append(sample_ids)

            batch = self.example_fn(sample_ids=sample_ids, dataset=self.dataset, ranking_type=self.ranking_type)

            if batch is not None:
                yield batch


class Splitter(object):
    @staticmethod
    def k_fold(k, all_ids):
        all_ids = sorted(all_ids)
        np.random.shuffle(all_ids)

        nb_ids = len(all_ids)
        split_size = nb_ids // k

        result = {}

        for i in range(1, k + 1):
            test_ids = all_ids[(i - 1) * split_size:i * split_size]
            train_ids = set(all_ids) - set(test_ids)
            result.update({i: {'train': list(train_ids), 'test': test_ids}})

        return result

    @staticmethod
    def split(splits, all_ids):
        remaining_ids = set(all_ids)

        for _, _, split_ids in splits:
            remaining_ids = remaining_ids - set(split_ids)

        remaining_ids = sorted(list(remaining_ids))
        np.random.shuffle(remaining_ids)
        result = {}

        for split_key, split_size, split_ids in splits:
            split_ids = [_id for _id in split_ids if _id in all_ids]
            remaining_percentage = split_size - (float(len(split_ids)) / float(len(all_ids)))

            if remaining_percentage > 0.0:
                split_point = int(np.floor(remaining_percentage * len(all_ids)))
                new_split_ids = remaining_ids[:split_point]
                remaining_ids = remaining_ids[split_point:]
            else:
                new_split_ids = []

            result.update({split_key: split_ids + new_split_ids})

        return result

    @staticmethod
    def check_splits(splits, directory):
        split_file = os.path.join(directory, '{}_ids.p')

        for split_key, article_ids in splits.items():
            if os.path.isfile(split_file.format(split_key)):
                previous_ids = pickle.load(open(split_file.format(split_key), "rb"))

                if set(previous_ids) != set(article_ids):
                    logging.warning('The {} set contains {}/{} new (previously unused) articles!'.format(
                        split_key, len(set(article_ids) - set(previous_ids)), len(article_ids)))
                    time.sleep(1)
                    answer = input("The current {} article ids differ from the previous run! "
                                   "Continue anyways (y/n)? ".format(split_key))

                    if answer.lower() not in ["y", "yes"]:
                        exit(0)
            else:
                pickle.dump(article_ids, open(split_file.format(split_key), "wb"))


class CDM(Dataset):
    string_fields = ["summary", "article"]
    metric_fields = ["rouge-1", "rouge-2", "js", "redundancy_1_2", "tfidf_cos_avg", "swap_count"]
    human_fields = ["overall", "grammar", "redundancy", "focus", "hter", "clarity"]
    available_fields = string_fields + metric_fields + human_fields

    def __init__(self, use_swap_samples=True, fields=None, start=0, pick_first=None):
        assert fields is not None and len(fields) >= 1, "A list of targets must be provided!"
        assert all([(f in self.available_fields) for f in fields]), "At least one field is invalid!"

        if use_swap_samples:
            assert all([(f not in self.human_fields) for f in fields]), "The swap summaries don't have human scores!"
        if not use_swap_samples:
            assert all([(f not in self.metric_fields) for f in fields]), "The scored summaries don't have auto metrics!"

        # Initialize the class properties
        self.fields = fields
        self.start = start
        self.pick_first = pick_first
        self.use_swap_samples = use_swap_samples

        # Properties containing the data
        self.articles = {}
        self.ref_summaries = {}
        self.samples = {}

        # Load the data according to the input type and training signal
        self.load_articles_and_refs()
        self.load_sample_summaries()
        self.check_integrity()

    @property
    def article_ids(self):
        return list(self.articles.keys())

    def load_sample_summaries(self):
        if self.use_swap_samples:
            for sample_file in glob.glob(os.path.join("data", "cdm", "samples", "*.p")):
                samples = pickle.load(open(sample_file, "rb"))
                logging.info("Loaded swap samples from {}".format(sample_file))

                for article_id, swap_dict in tqdm(samples.items(),
                                                  desc="Fetch swap samples from {}".format(
                                                      os.path.basename(sample_file))):
                    if article_id not in self.article_ids:
                        continue

                    if article_id not in self.samples:
                        self.samples.update({article_id: {}})

                    for nb_swaps_key, swap_samples in swap_dict.items():
                        # Pick less swap summaries if this has been specified
                        if self.pick_first is not None and self.pick_first > 0:
                            # If the start and end are greater than len(swap_samples) the resulting list will be empty
                            end = self.start + self.pick_first
                            swap_samples = swap_samples[self.start:end]

                        for swap_id, swap_sample in enumerate(swap_samples):
                            swap_summary_key = "{}-{}-{}".format(article_id, nb_swaps_key, swap_id)

                            if swap_summary_key not in self.samples[article_id]:
                                self.samples[article_id].update({swap_summary_key: {}})

                            if 'swap_count' in self.fields:
                                self.samples[article_id][swap_summary_key]['swap_count'] = nb_swaps_key

                            if 'summary' in self.fields:
                                self.samples[article_id][swap_summary_key]['summary'] = swap_sample['summary']

            for s in [f for f in ['js', 'redundancy_1_2', 'tfidf_cos_avg', 'rouge-1', 'rouge-2'] if f in self.fields]:
                swap_features_file = os.path.join("data", "cdm", "swap_features", s.replace("-", "_"), '*.p')

                for sample_file in glob.glob(swap_features_file):
                    samples = pickle.load(open(sample_file, "rb"))
                    logging.info("Loaded swap samples from {}".format(sample_file))

                    for article_id, swap_dict in tqdm(samples.items(), desc="Fetch auto metrics for swap samples"):
                        if article_id not in self.article_ids:
                            continue

                        if article_id not in self.samples:
                            self.samples.update({article_id: {}})

                        for nb_swaps_key, swap_samples in swap_dict.items():
                            if isinstance(nb_swaps_key, int):
                                for swap_id, swap_sample in enumerate(swap_samples):
                                    swap_summary_key = "{}-{}-{}".format(article_id, nb_swaps_key, swap_id)

                                    if swap_summary_key not in self.samples[article_id]:
                                        self.samples[article_id].update({swap_summary_key: {}})

                                    if 'swap_count' in self.fields:
                                        self.samples[article_id][swap_summary_key]['swap_count'] = nb_swaps_key

                                    if s == 'tfidf_cos_avg':
                                        self.samples[article_id][swap_summary_key]['tfidf-cos'] = swap_sample[0]
                                        self.samples[article_id][swap_summary_key]['tfidf-avg'] = swap_sample[1]
                                    if s == 'redundancy_1_2':
                                        self.samples[article_id][swap_summary_key]['redundancy-1'] = swap_sample[0]
                                        self.samples[article_id][swap_summary_key]['redundancy-2'] = swap_sample[1]
                                    else:
                                        self.samples[article_id][swap_summary_key][s] = swap_sample

                        if 'ref' in swap_dict:
                            if s == 'tfidf_cos_avg':
                                self.ref_summaries[article_id]['tfidf-cos'] = swap_dict['ref'][0]
                                self.ref_summaries[article_id]['tfidf-avg'] = swap_dict['ref'][1]
                            if s == 'redundancy_1_2':
                                self.ref_summaries[article_id]['redundancy-1'] = swap_dict['ref'][0]
                                self.ref_summaries[article_id]['redundancy-2'] = swap_dict['ref'][1]
                            else:
                                self.ref_summaries[article_id][s] = swap_dict['ref']

            if 'swap_count' in self.fields:
                for ref_summary in self.ref_summaries.values():
                    ref_summary['swap_count'] = 0
        else:
            scores, _ = readScores()

            for score in tqdm(scores, desc="Fetch samples from scored summaries"):
                article_id = score['id']

                # if score['scores']['overall'] is None:
                #     pos = self.article_ids.index(article_id)
                #     self.article_ids.pop(pos)
                #     continue

                if article_id not in self.article_ids:
                    continue

                # In case it is a reference summary the human scores get copied to article_refs dict
                if score['sys_name'] == 'reference':
                    for f in self.fields:
                        # Check if the score key exists for the current reference summary
                        if f not in self.string_fields and f in score['scores']:
                            self.ref_summaries[article_id][f] = score['scores'][f]
                else:
                    # If it is a system summary a new sample will be generated for each
                    if article_id not in self.samples:
                        self.samples.update({article_id: {}})

                    summary_key = "{}-{}".format(article_id, score['sys_name'])

                    if summary_key not in self.samples[article_id]:
                        self.samples[article_id].update({summary_key: {}})

                    if 'summary' in self.fields:
                        self.samples[article_id][summary_key]['summary'] = score['sys_summ']

                    for f in self.fields:
                        if f in score['scores']:
                            self.samples[article_id][summary_key][f] = score['scores'][f]

    def load_articles_and_refs(self):
        # Either load all article and reference summaries if swap summaries are used ...
        if self.use_swap_samples:
            article_refs = readArticleRefs(as_dict=True)
        else:
            # ... or load only the article and reference summaries for which human scores do exist!
            _, article_ids = readScores()
            article_refs = readArticleRefs(article_ids, as_dict=True)

        # Copy articles and reference summaries to the specific dictionaries
        for i, (article_id, article_ref) in enumerate(article_refs.items()):
            self.articles[article_id] = article_ref['article']
            self.ref_summaries[article_id] = {'summary': article_ref['ref']}

    def check_integrity(self):
        # Rewrite double value fields as single value fields
        tmp_fields = self.fields[:]
        self.fields = []

        for f in tmp_fields:
            if f == "redundancy_1_2":
                self.fields.append("redundancy-1")
                self.fields.append("redundancy-2")
            elif f == "tfidf_cos_avg":
                self.fields.append("tfidf-cos")
                self.fields.append("tfidf-avg")
            else:
                self.fields.append(f)

        fields = [f for f in self.fields if f != 'article']
        drop_ids = []

        for article_id, samples_per_article in self.samples.items():
            for sample_id, sample in samples_per_article.items():
                if not all([f in sample for f in fields]) or any([sample[f] is None for f in fields]):
                    # print("Missing fields in sample summary", article_id, sample_id)
                    drop_ids.append((article_id, sample_id))

        for article_id, sample_id in drop_ids:
            del self.samples[article_id][sample_id]

        drop_ids = []

        for article_id, ref_summary in self.ref_summaries.items():
            if not all([f in ref_summary for f in fields]) or any([ref_summary[f] is None for f in fields]):
                # print("Missing fields in reference summary", article_id)
                drop_ids.append(article_id)

        for article_id in drop_ids:
            del self.ref_summaries[article_id]

        drop_ids = []

        for article_id in self.article_ids:
            if (article_id not in self.samples or len(self.samples[article_id]) == 0) \
                    and article_id not in self.ref_summaries:
                # print("No samples and reference summaries for article", article_id)
                drop_ids.append(article_id)

        for article_id in drop_ids:
            del self.articles[article_id]

    def count_tokens(self, text_field):
        fields = [f for f in self.fields if f in self.string_fields]
        counter = Counter()

        def safe_counter_update(token_list):
            for token in token_list:
                if isinstance(token, str):
                    counter.update([token])
                else:
                    counter.update(token)

        # If the article is one of the requested fields we will count the word frequencies there
        if 'article' in fields:
            for article in tqdm(self.articles.values(), desc="Counting words in articles"):
                safe_counter_update(text_field.preprocess(article))

        # Do that as well for reference and sample summaries if the 'summary' is one of the requested fields
        if 'summary' in fields:
            for ref_summary in tqdm(self.ref_summaries.values(), desc="Counting words in reference summaries"):
                safe_counter_update(text_field.preprocess(ref_summary['summary']))

            for samples_per_article in tqdm(self.samples.values(), desc="Counting words in sample summaries"):
                for sample in samples_per_article.values():
                    safe_counter_update(text_field.preprocess(sample['summary']))

        return counter

    @classmethod
    def swap_based(cls, session_dir, vectors, device, batch_size, start=0, pick_first=None, raw=False,
                   lower_case=False, regression=False, debug=False, min_gap=None, fp16=False):
        cdm = cls(fields=['summary', 'article', 'swap_count'], start=start, pick_first=pick_first)
        cdm_test = cls(use_swap_samples=False, fields=['summary', 'article', 'overall'])
        text_field = RawTextField(lower=lower_case) if raw else SentenceWrappedField(lower=lower_case, fp16=fp16)

        if not raw:
            counter_file = os.path.join('data', 'cdm',
                                        '{}{}_counter.p'.format('cdm_swap', '_lower' if lower_case else ''))

            if os.path.isfile(counter_file):
                counter = pickle.load(open(counter_file, "rb"))
            else:
                counter = cdm.count_tokens(text_field)
                counter += cdm_test.count_tokens(text_field)
                pickle.dump(counter, open(counter_file, "wb"))

            vocab = Vocab(counter, specials=["<unk>", "<pad>"], vectors=vectors, unk_init=torch.randn_like)
            logging.info('Vocabulary contains {} entries'.format(len(vocab)))
            text_field.vocab = vocab

        # Split
        test_ids = pickle.load(open(os.path.join('data', 'cdm', 'test_ids.p'), 'rb'))
        splits = Splitter.split(splits=(('train', 0.8, []), ('val', 0.2, test_ids)), all_ids=cdm.article_ids)
        Splitter.check_splits(splits, session_dir)
        Splitter.check_splits({'test': cdm_test.article_ids}, session_dir)

        input_fields = {'summary': text_field, 'article': text_field,
                        'swap_count': FloatField(actual_input=False, fp16=fp16)}
        target_fields = {
            'label': FloatTargetField(fp16=fp16) if regression else ClassIndexTargetField(lookup={1: 0, -1: 1})}
        pairwise_pref = PairwisePreference(input_fields=input_fields, target_fields=target_fields,
                                           preference_key='swap_count', device=device)
        iterators = {}

        for split_key, article_ids in splits.items():
            if debug:
                article_ids = article_ids[:2]

            if min_gap is not None:
                iterators[split_key] = RandomIterator(cdm, article_ids, batch_size, pairwise_pref, min_gap=min_gap)
            else:
                iterators[split_key] = ClassicIterator(cdm, article_ids, batch_size, pairwise_pref)

            logging.info("{} set contains {} samples in total".format(split_key.capitalize(),
                                                                      iterators[split_key].total_samples))

        iterators['test'] = CombinationsIterator(cdm_test, cdm_test.article_ids, batch_size,
                                                 PairwisePreference(input_fields={'summary': text_field,
                                                                                  'article': text_field},
                                                                    target_fields=target_fields,
                                                                    preference_key='overall',
                                                                    device=device), shuffle=False)
        logging.info("{} set contains {} samples in total".format("Test", iterators['test'].total_samples))

        fields = {}
        fields.update(input_fields)
        fields.update(target_fields)

        return fields, iterators

    @classmethod
    def handcraft_swap_based(cls, session_dir, vectors, device, batch_size, start=0, pick_first=None, raw=False,
                             lower_case=False, regression=False, debug=False, min_gap=None, fp16=False):
        cdm = cls(fields=['summary', 'article', 'swap_count', 'rouge-1', 'rouge-2', 'js', 'redundancy_1_2',
                          'tfidf_cos_avg'], start=start, pick_first=pick_first)
        cdm_test = cls(use_swap_samples=False, fields=['summary', 'article', 'overall'])
        text_field = RawTextField(lower=lower_case) if raw else SentenceWrappedField(lower=lower_case, fp16=fp16)

        if not raw:
            counter_file = os.path.join('data', 'cdm',
                                        '{}{}_counter.p'.format('cdm_swap', '_lower' if lower_case else ''))

            if os.path.isfile(counter_file):
                counter = pickle.load(open(counter_file, "rb"))
            else:
                counter = cdm.count_tokens(text_field)
                counter += cdm_test.count_tokens(text_field)
                pickle.dump(counter, open(counter_file, "wb"))

            vocab = Vocab(counter, specials=["<unk>", "<pad>"], vectors=vectors, unk_init=torch.randn_like)
            logging.info('Vocabulary contains {} entries'.format(len(vocab)))
            text_field.vocab = vocab

        # Split
        test_ids = pickle.load(open(os.path.join('data', 'cdm', 'test_ids.p'), 'rb'))
        splits = Splitter.split(splits=(('train', 0.8, []), ('val', 0.2, test_ids)), all_ids=cdm.article_ids)
        Splitter.check_splits(splits, session_dir)
        Splitter.check_splits({'test': cdm_test.article_ids}, session_dir)

        input_fields = {'summary': text_field, 'article': text_field,
                        'swap_count': FloatField(actual_input=False, fp16=fp16),
                        'rouge-1': FloatField(fp16=fp16), 'rouge-2': FloatField(fp16=fp16),
                        'js': FloatField(fp16=fp16), 'redundancy-1': FloatField(fp16=fp16),
                        'redundancy-2': FloatField(fp16=fp16), 'tfidf-cos': FloatField(fp16=fp16),
                        'tfidf-avg': FloatField(fp16=fp16)}
        target_fields = {
            'label': FloatTargetField(fp16=fp16) if regression else ClassIndexTargetField(lookup={1: 0, -1: 1})}
        pairwise_pref = PairwisePreference(input_fields=input_fields, target_fields=target_fields,
                                           preference_key='swap_count', device=device)
        iterators = {}

        for split_key, article_ids in splits.items():
            if debug:
                article_ids = article_ids[:2]

            if min_gap is not None:
                iterators[split_key] = RandomIterator(cdm, article_ids, batch_size, pairwise_pref, min_gap=min_gap)
            else:
                iterators[split_key] = ClassicIterator(cdm, article_ids, batch_size, pairwise_pref)

            logging.info("{} set contains {} samples in total".format(split_key.capitalize(),
                                                                      iterators[split_key].total_samples))

        iterators['test'] = CombinationsIterator(cdm_test, cdm_test.article_ids, batch_size,
                                                 PairwisePreference(input_fields={'summary': text_field,
                                                                                  'article': text_field},
                                                                    target_fields=target_fields,
                                                                    preference_key='overall',
                                                                    device=device), shuffle=False)
        logging.info("{} set contains {} samples in total".format("Test", iterators['test'].total_samples))

        fields = {}
        fields.update(input_fields)
        fields.update(target_fields)

        return fields, iterators

    @classmethod
    def human_based(cls, session_dir, vectors, device, batch_size, k_fold=None, raw=False, lower_case=False,
                    regression=True, debug=False, combinations=False, fp16=False):
        cdm = cls(use_swap_samples=False, fields=['summary', 'article', 'overall'])
        text_field = RawTextField(lower=lower_case) if raw else SentenceWrappedField(lower=lower_case, fp16=fp16)

        if k_fold is not None:
            splits = Splitter.k_fold(k_fold[1], all_ids=cdm.article_ids)[k_fold[0]]
        else:
            splits = Splitter.split(splits=(('train', 0.7, []), ('test', 0.3, [])), all_ids=cdm.article_ids)

        Splitter.check_splits(splits, session_dir)

        if not raw:
            # Creating the vocabulary for the cdm-human dataset is fast enough
            counter = cdm.count_tokens(text_field)
            vocab = Vocab(counter, specials=["<unk>", "<pad>"], vectors=vectors, unk_init=torch.randn_like)
            logging.info('Vocabulary contains {} entries'.format(len(vocab)))
            text_field.vocab = vocab

        input_fields = {'summary': text_field, 'article': text_field,
                        'overall': FloatField(actual_input=False, fp16=fp16)}
        target_fields = {
            'label': FloatTargetField(fp16=fp16) if regression else ClassIndexTargetField(lookup={1: 0, -1: 1})}
        iterators = {}

        for split_key, article_ids in splits.items():
            if debug:
                article_ids = article_ids[:2]

            if combinations:
                iterators.update({split_key: CombinationsIterator(cdm, article_ids, batch_size,
                                                                  PairwisePreference(input_fields,
                                                                                     target_fields,
                                                                                     preference_key='overall',
                                                                                     device=device))})
            # elif 'prm' in self.run_args.dataset_name:
            #     iterators.update({split_key: PermutationsIterator(cdm, article_ids, batch_size,
            #                                                       PairwisePreference(input_fields,
            #                                                                          target_fields,
            #                                                                          preference_key='overall',
            #                                                                          device=device))})
            else:
                iterators.update({split_key: ClassicIterator(cdm, article_ids, batch_size,
                                                             Pointwise(input_fields, target_fields,
                                                                       label_key='overall',
                                                                       device=device),
                                                             ranking_type=CDMRankingType.POINTWISE)})

            logging.info("{} set contains {} samples in total".format(split_key.capitalize(),
                                                                      iterators[split_key].total_samples))

        fields = {}
        fields.update(input_fields)
        fields.update(target_fields)

        return fields, iterators


def test_cdm():
    # This code shows how to use the the CDM dataset class
    # TODO Download the pregenerated swap samples from here:
    # https://drive.google.com/uc?id=1hgRZF5882q0JVa_2oUPRS_nwlCRlSTI6&export=download
    # TODO Extract the files "1_100.p", "2_100.p" ... into data/samples/ directory

    # If you need the data as raw strings
    # TODO Set debug to False to get all examples (debug limits each split to 10 examples)
    # TODO Adopt lower_case and pick_first parameters to your needs
    # pick_first=2 means only the first two swap summaries for each topic are used
    batch_size = 8
    start = time.time()
    debug = True
    # cdm = CDM(use_swap_samples=False, fields=['summary', 'article', 'overall'])
    cdm = CDM(
        fields=["summary", "article", "swap_count", "rouge-1", "rouge-2", "js", "redundancy_1_2", "tfidf_cos_avg"],
        start=0, pick_first=5)
    print("Loading took {:.2f}".format(time.time() - start))
    # kfold = Splitter().k_fold(5, all_ids=cdm.article_ids)
    # print(kfold)
    # exit(0)

    splits = Splitter().split(splits=(('train', 0.8, []), ('val', 0.2, pickle.load(open(
        os.path.join('data', 'cdm', 'test_ids.p'), "rb")))), all_ids=cdm.article_ids)

    lower_case = False
    swf = SentenceWrappedField(lower=lower_case)  # RawTextField(lower=lower_case)  #
    start = time.time()

    counter_file = os.path.join('data', 'cdm',
                                '{}{}_counter.p'.format('cdm_swap', '_lower' if lower_case else ''))

    assert os.path.isfile(counter_file)
    counter = pickle.load(open(counter_file, "rb"))

    # counter = cdm.count_tokens(swf)
    vectors = GloVe(name='840B', dim=300)
    vocab = Vocab(counter, specials=[Field().unk_token, Field().pad_token], vectors=vectors)
    swf.vocab = vocab
    print("Vocab build took {:.2f}".format(time.time() - start))
    label_field = ClassIndexTargetField(lookup={1: 0, -1: 1})  # StringTargetField()  # FloatTargetField()

    input_fields = {'summary': swf, 'article': swf, 'swap_count': FloatField()}
    target_fields = {'label': label_field}
    train_iter = SwapCombinationsIterator(cdm, splits['train'][:10], batch_size,
                                          PairwisePreference(input_fields=input_fields,
                                                             target_fields=target_fields,
                                                             preference_key='swap_count'), min_gap=3)
    val_iter = ClassicIterator(cdm, splits['val'], batch_size, PairwisePreference(input_fields=input_fields,
                                                                                  target_fields=target_fields,
                                                                                  preference_key='swap_count'))
    print("{} train samples with batch_size {} gives {} iterations".format(train_iter.total_samples, batch_size,
                                                                           len(train_iter)))
    print("{} val samples with batch_size {} gives {} iterations".format(val_iter.total_samples, batch_size,
                                                                         len(val_iter)))

    i = 0
    for batch in tqdm(train_iter, desc="Running through all training iterations"):
        i += 1

    print(batch)

    assert i == len(train_iter)

    i = 0
    for batch in tqdm(val_iter, desc="Running through all validation iterations"):
        i += 1

    assert i == len(val_iter)
    print(batch)

    cdm_test = CDM(use_swap_samples=False, fields=["summary", "article", "overall"], debug=debug)
    test_iter = PermutationsIterator(cdm_test, cdm_test.article_ids, batch_size,
                                     PairwisePreference(
                                         input_fields={'summary': swf, 'article': swf, 'overall': FloatField()},
                                         target_fields={'label': label_field},
                                         preference_key='overall'),
                                     shuffle=False)

    print("{} test samples with batch_size {} gives {} iterations".format(test_iter.total_samples, batch_size,
                                                                          len(test_iter)))

    i = 0
    for batch in tqdm(test_iter, desc="Running through all test iterations"):
        i += 1

    assert i == len(test_iter)
    print(batch)


if __name__ == "__main__":
    test_cdm()
