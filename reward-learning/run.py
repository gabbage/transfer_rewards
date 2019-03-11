import argparse
import glob
import importlib
import logging
import os
import random
import re
from copy import deepcopy
from datetime import datetime
from logging.handlers import SocketHandler

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.optim
#from apex.optimizers import FusedAdam, FP16_Optimizer
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.handlers import EarlyStopping
from ignite.metrics import Loss, Accuracy
#from pytorch_pretrained_bert import BertAdam
#from tabulate import tabulates
from torchtext.vocab import GloVe, FastText

from datasets.cdm import CDM, RandomIterator
from helpers.checkpoint import save_weights, restore_weights
from helpers.evaluation import run_absolute_rank_evaluation, run_pairwise_rank_evaluation
from helpers.loss import SimpleMarginRankingLoss, SaferAccuracy, PairwiseMSE
from helpers.metric_history import MetricHistory
from helpers.nlp_engine import create_evaluator, create_trainer
from helpers.vectors import GoogleNews
from helpers.visdom_plot import VisdomPlot
from models.model_api import ModelAPI
from models.phis.encoders import Encoding
from models.phis.phi_api import PhiAPI
from models.phis.phi_sequence import PhiSequence


class Runner(object):
    def __init__(self, run_args, unparsed_args):
        # Properties of the Runner object
        self.run_args = run_args
        self.unparsed_args = unparsed_args
        self.phi_args = None
        self.model_args = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.initial_epoch = 0
        self.phi = None
        self.trainer = None
        self.validator = None
        self.tester = None

        # Prepare and configure the logging environment
        self.logger = Runner.prepare_loggers(run_args.session_dir, run_args.debug)

        # Log the run arguments (model arguments will be logged later)
        self.logger.info('Run arguments: {}'.format(run_args))
        Runner.save_arguments(os.path.join(run_args.session_dir, 'run_args.txt'), run_args)

        # Set torch specific details
        if run_args.force_cpu:
            self.device = torch.device('cpu')
            self.logger.info('Device: {} (enforced)'.format(str(self.device)))
        else:
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    gpu_id = input("More than one GPU is available! Please specify the GPU id: ")
                else:
                    gpu_id = 0

                torch.cuda.set_device(int(gpu_id))
                self.device = torch.device("cuda:{:d}".format(int(gpu_id)))
            else:
                self.device = torch.device('cpu')
            self.logger.info('Device: {}'.format(str(self.device)))

        if run_args.fp16:
            self.logger.info('Use 16-bits float with apex to improve performance!')
            run_args.optimizer = 'fused_adam'

        # Set the number of threads to use
        if str(self.device) == 'cpu':
            torch.set_num_threads(run_args.num_threads)

        # Fix the random generator seed
        self.logger.debug('Set manual seed for random generators to {}'.format(run_args.random_seed))
        random.seed(run_args.random_seed)
        np.random.seed(run_args.random_seed)
        torch.manual_seed(run_args.random_seed)

        if 'cuda' in str(self.device):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(run_args.random_seed)

    @staticmethod
    def parse_run_args(args=None, custom_run_options_fn=None):
        run_options = argparse.ArgumentParser()
        run_options.add_argument('-a', '--learning_rate', action='store', default=1e-5, type=float,
                                 help='Change the learning rate of the optimizer.')
        run_options.add_argument('-e', '--epochs', action='store', default=3, type=int,
                                 help='The number of epochs for which the model is trained.')
        run_options.add_argument('-r', '--restore_epoch', action='store', default=None, type=int,
                                 help='If weights should be restored from another epoch than the last one. '
                                      'Set to -1 to restore the epoch specified in file "restore.txt" in session dir.')
        run_options.add_argument('-b', '--batch_size', action='store', default=8, type=int,
                                 help='The number of samples in one mini-batch.')
        run_options.add_argument('-s', '--session_name', action='store', default=str(datetime.now()).
                                 replace(' ', '_').replace(':', '-')[:-7], type=str,
                                 help='Session name of this training and evaluation run.')
        run_options.add_argument('-m', '--model_file', action='store', type=str, required=True,
                                 help='Relative path to either a python module in models folder or a saved model file.')
        run_options.add_argument('-v', '--vectors_name', action='store', default='glove840b', type=str,
                                 help='The word vectors that get used to embed the words/tokens of the sentence.')
        run_options.add_argument('--vectors_dim', action='store', default=300, type=int,
                                 help='The dimensionality of the word vectors (must be supported by the used vectors).')
        run_options.add_argument('-d', '--dataset_name', action='store', type=str, required=True,
                                 help='The name of the dataset that will be processed.')
        run_options.add_argument('--num_threads', action='store', default=4, type=int,
                                 help='Set the number of threads to use for parallelism.')
        run_options.add_argument('--merge_vocabs', action='store', default=1, type=int,
                                 help='Merge the vocabularies in case multiple input fields should use the same')
        run_options.add_argument('-c', '--force_cpu', action='store', default=0, type=int,
                                 help='Force the runner to work on cpu')
        run_options.add_argument('-l', '--loss', action='store', type=str, required=True,
                                 help='The loss function to use for training')
        run_options.add_argument('-o', '--optimizer', action='store', default='adam', type=str,
                                 help='The optimizer to use for training')
        run_options.add_argument('-w', '--weight_decay', action='store', default=0, type=float,
                                 help='The weight of the L2 weight regularizer')
        run_options.add_argument('--pick_first', action='store', default=None, type=int,
                                 help='Pick only the first n samples (concrete impact depends also on the dataset).')
        run_options.add_argument('--note', action='store', default='', type=str,
                                 help='Add a small note here to explain what the training run is supposed to do.')
        run_options.add_argument('--rank_evaluation', action='store', default='absolute', type=str,
                                 help="The type of the rank evaluation can be 'absolute' if the model produces a rank "
                                      "for a single summary or 'pairwise' if model outputs pairwise preferences.")
        run_options.add_argument('--early_stopping_patience', action='store', default=0, type=int,
                                 help='The patience of early stopping handler. No early stopping if set to zero.')
        run_options.add_argument('--early_stopping_metric', action='store', default=None, type=str,
                                 help='The metric name used for early stopping.')
        run_options.add_argument('--lower_case', action='store', default=0, type=int,
                                 help='Whether or not the text will be converted to lower case.')
        run_options.add_argument('--debug', action='store', default=0, type=int,
                                 help='The debug flag limits the dataset size to 10 examples to speed up computations.')
        run_options.add_argument('-p', '--phis', action='append', default=None, type=str,
                                 help='Use this argument to specify multiple encoding functions in order.')
        run_options.add_argument('--train_phi', action='store', default=1, type=int,
                                 help='Whether or not the phi weights should be trained along with the model.')
        run_options.add_argument('--train_embeddings', action='store', default=0, type=int,
                                 help='Whether or not to train the embedding layer along with the model.')
        run_options.add_argument('--drop_embeddings', action='store', default=1, type=int,
                                 help='Whether or not the embeddings get dropped. Use in case of different vocab size.')
        run_options.add_argument('--random_seed', action='store', default=1337, type=int,
                                 help='The seed for random generators.')
        run_options.add_argument('--lower_gap', action='store', default=None, type=int,
                                 help='Number of epochs after which the min_gap will be decreased by one.')
        run_options.add_argument('-k', '--k_fold', action='store', default=None, type=str,
                                 help='If k-fold cross validation should be used. If only a specific split should be '
                                      'executed then set this argument to "3/5" for the third split of 5-fold CV. Or '
                                      'use "1,3,5/5" to run training with the first, third and fifth split.')
        run_options.add_argument('--warmup_proportion', action="store", default=-1.0, type=float,
                                 help='Proportion of training to perform linear learning rate warmup for. '
                                      'E.g., 0.1 = 10%% of training.')
        run_options.add_argument('--fp16', action='store_true',
                                 help='Whether to use 16-bit float precision instead of 32-bit')

        if custom_run_options_fn:
            custom_run_options_fn(run_options)

        run_args, unparsed_args = run_options.parse_known_args(args)

        # Add the path to the session directory to the run_args dictionary
        run_args_dict = vars(run_args)

        if run_args.phis is not None and len(run_args.phis) > 0:
            phi_name = '_'.join([p.replace('_', '-') for p in run_args.phis])
            run_args_dict['session_dir'] = os.path.join('runs', run_args.model_file, phi_name, run_args.session_name)
        else:
            run_args_dict['session_dir'] = os.path.join('runs', run_args.model_file, run_args.session_name)

        if run_args.phis is not None and len(run_args.phis) > 0:
            phi_name = '_'.join([p.replace('_', '-') for p in run_args.phis])
            run_args_dict['env_name'] = '{}_{}'.format(run_args.model_file.replace('_', '-'), phi_name)
        else:
            run_args_dict['env_name'] = run_args.model_file.replace('_', '-')

        return run_args, unparsed_args

    def _build_model(self, input_size, num_inputs, fields):
        assert self.run_args.model_file is not None, 'A model file has to be specified!'
        module = importlib.import_module('models.{}'.format(self.run_args.model_file))
        self.model, self.model_args, self.unparsed_args = module.init(input_size, num_inputs, self.unparsed_args,
                                                                      device=self.device)

        if self.phi is None:
            self.logger.debug('The model will have its own embedding layer because no phi is used!')
            self._add_embedding_layer(fields)

    @staticmethod
    def add_embedding_layer(model, train_embeddings, merge_vocabs, fields):
        assert isinstance(model, ModelAPI) or isinstance(model, PhiAPI), 'The model has to be a subclass of the ' \
                                                                         'ModelAPI or PhiAPI!'

        model.prep_inputs_fn = {}
        freeze = not bool(train_embeddings)

        for field_name, field in fields.items():
            if field.use_vocab and not field.is_target:
                if merge_vocabs:
                    model.prep_inputs_fn = torch.nn.Embedding.from_pretrained(field.vocab.vectors, freeze=freeze)
                    break
                else:
                    model.prep_inputs_fn.update(
                        {field_name: torch.nn.Embedding.from_pretrained(field.vocab.vectors, freeze=freeze)})

    def _add_embedding_layer(self, fields):
        Runner.add_embedding_layer(self.model, self.run_args.train_embeddings, self.run_args.merge_vocabs, fields)

    def _build_phi(self, input_size, fields, input_field_keys):
        if self.run_args.phis is None or len(self.run_args.phis) == 0:
            return input_size, len(input_field_keys)
        else:
            self.phi, self.phi_args, self.unparsed_args = PhiSequence.init(input_size, self.unparsed_args,
                                                                           self.run_args.phis, input_field_keys,
                                                                           self.device, fields)

            if Encoding.INDEX in self.phi.enc_seq.input_encoding():
                Runner.add_embedding_layer(self.phi.enc_seq[0], self.run_args.train_embeddings,
                                           self.run_args.merge_vocabs, fields)

            return self.phi.output_size, self.phi.num_outputs

    def restore_model_weights(self):
        self.initial_epoch = restore_weights(self.model, self.run_args.session_dir, self.run_args.restore_epoch,
                                             component_key='model', device=self.device,
                                             drop_embeddings=self.run_args.drop_embeddings)

    def restore_phi_weights(self):
        if self.phi and self.phi.trainable:
            last_phi_epoch = restore_weights(self.phi, self.run_args.session_dir, self.run_args.restore_epoch,
                                             component_key='phi', device=self.device,
                                             drop_embeddings=self.run_args.drop_embeddings)

            if last_phi_epoch != self.initial_epoch:
                self.logger.warning('Phi epoch ({}) differs from last model epoch ({})!'.format(last_phi_epoch,
                                                                                                self.initial_epoch))

    def restore_optimizer_state(self):
        if self.optimizer:
            restore_weights(self.optimizer, self.run_args.session_dir, self.run_args.restore_epoch,
                            component_key='optimizer', device=self.device)

    def set_initial_epoch_handler(self, engine):
        engine.state.epoch = self.initial_epoch
        self.logger.info('{} training of {} from epoch {}.'.format('Continue' if self.initial_epoch > 0 else 'Start',
                                                                   self.run_args.model_file, self.initial_epoch + 1))

    @staticmethod
    def save_arguments(argument_file, args):
        """ Save all arguments of the argument parser to a file. """
        with open(argument_file, 'w') as fh:
            for arg in vars(args):
                fh.write('{}={}\n'.format(arg, getattr(args, arg)))

    def _check_and_save_arguments(self):
        phi_args_file = os.path.join(self.run_args.session_dir, 'phi_args.txt')
        model_args_file = os.path.join(self.run_args.session_dir, 'model_args.txt')

        def check(args_file, args, name):
            diff_dict = {}

            if os.path.isfile(args_file):
                for line in open(args_file, 'r'):
                    key, value = line.split('=')
                    key = key.strip()
                    value = value.strip()

                    if not hasattr(args, key):
                        diff_dict[key] = {'old': value, 'new': None}
                    elif "{}".format(getattr(args, key)) != value:
                        diff_dict[key] = {'old': value, 'new': "{}".format(getattr(args, key))}
                        delattr(args, key)
                    else:
                        delattr(args, key)
                for key in vars(args):
                    diff_dict[key] = {'old': None, 'new': "{}".format(getattr(args, key))}

                if len(diff_dict) > 0:
                    self.logger.info('The old and new {} arguments differ by:\n{}'.format(name, diff_dict))
                    answer = input('Do you want to start the run anyways? (y/n) ')

                    if answer.lower() in ['y', 'yes']:
                        Runner.save_arguments(args_file, args)
                    else:
                        self.logger.info('Stopping the run because the user decided to!')
                        exit(0)
            else:
                Runner.save_arguments(args_file, args)

        check(model_args_file, deepcopy(self.model_args), 'model')

        if self.phi is not None:
            check(phi_args_file, deepcopy(self.phi_args), 'phi')

    def save_model_weights(self, epoch):
        save_weights(self.model, self.run_args.session_dir, epoch, component_key='model')

    def save_phi_weights(self, epoch):
        if self.phi:
            save_weights(self.phi, self.run_args.session_dir, epoch, component_key='phi')

    def save_optimizer_state(self, epoch):
        if self.optimizer:
            save_weights(self.optimizer, self.run_args.session_dir, epoch, component_key='optimizer')

    def save_weights_handler(self, engine):
        self.save_model_weights(engine.state.epoch)
        self.save_phi_weights(engine.state.epoch)
        self.save_optimizer_state(engine.state.epoch)

    def decrease_min_gap(self, engine, train_iter):
        if (engine.state.epoch % self.run_args.lower_gap) == 0:
            new_min_gap = max(1, train_iter.min_gap - 1)

            if train_iter.min_gap != new_min_gap:
                self.logger.info("Decrease random iterator min_gap to {}".format(new_min_gap))
                train_iter.min_gap = new_min_gap

    def score_function(self, engine):
        value = engine.state.metrics[self.run_args.early_stopping_metric]

        return -value if 'loss' in self.run_args.early_stopping_metric else value

    @staticmethod
    def prepare_loggers(session_dir, debug):
        # Create session directory if necessary
        if not os.path.isdir(session_dir):
            os.makedirs(session_dir)

        # Create logging handlers with special formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
        file_handler = logging.FileHandler(os.path.join(session_dir, 'output.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))

        # Create a logger and add the handlers to it
        logger = logging.getLogger()
        logger.handlers = []
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        if debug:
            logger.addHandler(SocketHandler('127.0.0.1', 19996))
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        return logger

    @staticmethod
    def load_vectors(vectors_name, vectors_dim=300):
        if vectors_name.lower() == 'glove' or vectors_name.lower() == 'glove6b':
            vectors = GloVe(name='6B', dim=vectors_dim)
        elif vectors_name.lower() == 'glove27b':
            vectors = GloVe(name='twitter.27B', dim=vectors_dim)
        elif vectors_name.lower() == 'glove42b':
            vectors = GloVe(name='42B', dim=vectors_dim)
        elif vectors_name.lower() == 'glove840b':
            vectors = GloVe(name='840B', dim=vectors_dim)
        elif vectors_name.lower() == 'gnews':
            vectors = GoogleNews()
        elif vectors_name.lower() == 'fasttext':
            vectors = FastText()
        elif vectors_name.lower() == 'none':
            vectors = None
        else:
            raise NotImplementedError

        if vectors:
            logging.info('Loaded {} with {} vectors of size {}'.format(vectors_name, vectors.vectors.size(0),
                                                                       vectors.dim))
        return vectors

    def _load_vectors(self):
        return Runner.load_vectors(self.run_args.vectors_name, self.run_args.vectors_dim)

    def load_data(self, vectors, start=0):
        # Load the dataset dynamically based on given dataset name
        logging.debug("Load dataset for dataset_name='{}'".format(self.run_args.dataset_name))

        if 'cdm' in self.run_args.dataset_name:
            raw = 'raw' in self.run_args.dataset_name
            reg = 'reg' in self.run_args.dataset_name

            if 'pref' in self.run_args.dataset_name:
                if 'rnd' in self.run_args.dataset_name:
                    min_gap = re.sub('cdm.*rnd', '', self.run_args.dataset_name)
                    min_gap = int(min_gap) if len(min_gap) == 1 else 1
                else:
                    min_gap = None

                if 'handcraft' in self.run_args.dataset_name:
                    return CDM.handcraft_swap_based(self.run_args.session_dir, vectors, self.device,
                                                    self.run_args.batch_size, start, self.run_args.pick_first, raw,
                                                    self.run_args.lower_case, reg, self.run_args.debug, min_gap,
                                                    self.run_args.fp16)
                else:
                    return CDM.swap_based(self.run_args.session_dir, vectors, self.device, self.run_args.batch_size,
                                          start, self.run_args.pick_first, raw, self.run_args.lower_case, reg,
                                          self.run_args.debug, min_gap, self.run_args.fp16)
            elif 'human' in self.run_args.dataset_name:
                cmb = 'cmb' in self.run_args.dataset_name

                return CDM.human_based(self.run_args.session_dir, vectors, self.device, self.run_args.batch_size,
                                       self.run_args.k_fold, raw, self.run_args.lower_case, reg, self.run_args.debug,
                                       cmb, self.run_args.fp16)
            else:
                raise NotImplementedError("Can't load 'cdm' dataset with name: {}".format(self.run_args.dataset_name))
        else:
            raise NotImplementedError("Can't load dataset with name: {}".format(self.run_args.dataset_name))

    def _init_optimizer(self):
        # TODO Yoon Kim uses L2 norm constraint on the weights, but how much weight_decay is it actually
        self.logger.debug("Initialize '{}' optimizer with learning_rate={:.4e} and weight_decay={:.4e}".format(
            self.run_args.optimizer, self.run_args.learning_rate, self.run_args.weight_decay))

        optimizer_name = self.run_args.optimizer.lower()
        params = list(self.model.named_parameters())

        if self.run_args.train_phi and self.phi and self.phi.trainable:
            self.logger.debug('Add phi parameters to optimizer!')
            params += list(self.phi.named_parameters())

        if self.run_args.weight_decay > 0:
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay':
                    self.run_args.weight_decay},
                {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        else:
            optimizer_grouped_parameters = [{'params': [p for n, p in params], 'weight_decay': 0.0}]

        if optimizer_name == 'adam':
            return torch.optim.Adam(optimizer_grouped_parameters, lr=self.run_args.learning_rate)
        elif optimizer_name == 'adadelta':
            return torch.optim.Adadelta(optimizer_grouped_parameters, lr=self.run_args.learning_rate)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(optimizer_grouped_parameters, lr=self.run_args.learning_rate)
        elif optimizer_name == 'asgd':
            return torch.optim.ASGD(optimizer_grouped_parameters, lr=self.run_args.learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(optimizer_grouped_parameters, lr=self.run_args.learning_rate)
        elif optimizer_name == 'adagrad':
            return torch.optim.Adagrad(optimizer_grouped_parameters, lr=self.run_args.learning_rate)
        # elif optimizer_name == 'bert':
        #     return BertAdam(optimizer_grouped_parameters, lr=self.run_args.learning_rate,
        #                     warmup=self.run_args.warmup_proportion)
        # elif optimizer_name == 'fused_adam':
        #     optimizer = FusedAdam(optimizer_grouped_parameters, lr=self.run_args.learning_rate,
        #                           bias_correction=False, max_grad_norm=1.0)
        #     # if args.loss_scale == 0:
        #     #     optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        #     # else:
        #     #     optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        #     return FP16_Optimizer(optimizer, static_loss_scale=0)
        else:
            raise NotImplementedError('Optimizer {} is not available'.format(self.run_args.optimizer))

    def _init_loss(self):
        self.logger.debug("Initialize '{}' loss".format(self.run_args.loss))

        if self.run_args.loss.lower() == 'nll':
            return torch.nn.NLLLoss()
        if self.run_args.loss.lower() == 'ce':
            return torch.nn.CrossEntropyLoss()
        elif self.run_args.loss.lower() == 'l1':
            return torch.nn.L1Loss()
        elif self.run_args.loss.lower() == 'mse':
            return torch.nn.MSELoss()
        elif self.run_args.loss.lower() == 'pmse':
            return PairwiseMSE()
        elif self.run_args.loss.lower() == 'huber':
            return torch.nn.SmoothL1Loss()
        elif self.run_args.loss.lower() == 'mrl':
            return SimpleMarginRankingLoss()
        elif self.run_args.loss.lower().startswith('mrl_'):
            return SimpleMarginRankingLoss(str_params=self.run_args.loss.lower().replace('mrl_', ''))
        elif self.run_args.loss.lower() == 'hinge':
            return torch.nn.HingeEmbeddingLoss()
        else:
            raise NotImplementedError('Loss {} is not available'.format(self.run_args.loss))

    def run_evaluations_handler(self, trainer, iterators):
        if 'val' in iterators:
            self.validator.run(iterators['val'])
            values = ['{}={:.4f}'.format(name, value) for name, value in self.validator.state.metrics.items()]
            epoch = trainer.state.epoch if trainer is not None else self.initial_epoch
            epoch = self.run_args.restore_epoch if self.run_args.restore_epoch else epoch
            self.logger.info('Validation results: {} after epoch {}'.format(', '.join(values), epoch))
        else:
            self.logger.debug('Validation run was skipped because no data iterator exists!')

        if 'test' in iterators:
            self.tester.run(iterators['test'])
            values = ['{}={:.4f}'.format(name, value) for name, value in self.tester.state.metrics.items()]
            epoch = trainer.state.epoch if trainer is not None else self.initial_epoch
            epoch = self.run_args.restore_epoch if self.run_args.restore_epoch else epoch
            self.logger.info('Test results: {} after epoch {}'.format(', '.join(values), epoch))
        else:
            self.logger.debug('Test run was skipped because no data iterator exists!')

    def start(self):
        # Create visdom plot early to avoid preparing data before visdom server is running
        visdom_plot = VisdomPlot(env=self.run_args.env_name, win=self.run_args.session_name,
                                 k_fold=self.run_args.k_fold)

        # Load the word vectors and the dataset
        vectors = self._load_vectors()
        input_size = vectors.dim if vectors else None

        # Load the data either we prebuild function or a custom one, remove full vector tensor afterwards
        fields, iterators = self.load_data(vectors)

        # Build the phi function if specified as run argument, its output_dim becomes the new input_dim of the model
        input_field_keys = [k for k, f in fields.items() if (hasattr(f, 'actual_input') and f.actual_input)]
        input_size, num_inputs = self._build_phi(input_size, fields, input_field_keys)

        # Build the model and add an embedding layer only if no phi function is used!
        self._build_model(input_size, num_inputs, fields)

        # This function saves phi and model arguments to file and checks for differences if those files already do exist
        self._check_and_save_arguments()

        # Restore model/phi weights from the previous run if available
        self.restore_model_weights()
        self.restore_phi_weights()

        # Create the optimizer and loss/criterion function
        self.optimizer = self._init_optimizer()
        self.criterion = self._init_loss()
        self.restore_optimizer_state()

        # Create the evaluation metrics
        if self.run_args.loss.lower().startswith('mrl_max'):
            # TODO To compute the max margin loss correctly it is necessary to provide the margin
            #  (e.g. difference in swap_count or overall score)
            val_metrics = {}
            test_metrics = {}
        else:
            val_metrics = {'val_loss': Loss(self.criterion)}
            test_metrics = {'test_loss': Loss(self.criterion)}

        if self.run_args.loss.lower() in ['nll', 'ce']:
            self.logger.debug('Add accuracy metric because this is a classification problem!')
            val_metrics.update({'val_accuracy': Accuracy()})
            test_metrics.update({'test_accuracy': Accuracy()})
        elif 'mrl' in self.run_args.loss.lower() or self.run_args.loss.lower() in ['pmse']:
            self.logger.debug('Add (safer) accuracy metric because this is a classification problem!')
            val_metrics.update({'val_accuracy': SaferAccuracy()})
            test_metrics.update({'test_accuracy': SaferAccuracy()})

        # Create PyTorch Ignite trainer and evaluator (evaluator can be used for test and validation split)
        self.trainer = create_trainer(self.model, self.optimizer, self.criterion, device=self.device, phi=self.phi,
                                      train_phi=self.run_args.train_phi, fp16=self.run_args.fp16)
        self.validator = create_evaluator(self.model, val_metrics, phi=self.phi)
        self.tester = create_evaluator(self.model, test_metrics, phi=self.phi)

        # Create and attach more handlers like progress bar and metric history
        ProgressBar().attach(self.trainer, output_transform=lambda x: {'train_loss': x})
        ProgressBar().attach(self.validator)
        ProgressBar().attach(self.tester)
        metric_history = MetricHistory()
        metric_history.attach(self.trainer, metrics=[MetricHistory.output('train_loss')], reduce_fn=np.mean,
                              store_event=Events.ITERATION_COMPLETED, reduce_event=Events.EPOCH_COMPLETED)
        metric_history.attach(self.validator, metrics=list(val_metrics.keys()))
        metric_history.attach(self.tester, metrics=list(test_metrics.keys()))
        visdom_plot.metric_history = metric_history

        if self.run_args.epochs > 0:
            visdom_plot.step = self.initial_epoch

            # Set the trainer initial epoch to last epoch
            self.trainer.add_event_handler(Events.STARTED, self.set_initial_epoch_handler)

            # Save model weights on epoch completion
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.save_weights_handler)

            # Run evaluation on epoch completion
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.run_evaluations_handler, iterators)

            # Add plot handler on trainer epoch completion
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, visdom_plot.plot)

            # Lower the min_gap if the training iterator is the random iterator and lower_gap is not None
            if self.run_args.lower_gap is not None and isinstance(iterators['train'], RandomIterator):
                self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.decrease_min_gap, iterators['train'])

            # Add early stopping
            if self.run_args.early_stopping_patience:
                if self.run_args.early_stopping_metric is None:
                    vars(self.run_args)['early_stopping_metric'] = list(val_metrics.keys())[-1]

                self.logger.info('Use early stopping on {} with patience {}'.format(
                    self.run_args.early_stopping_metric, self.run_args.early_stopping_patience))
                early_stopping = EarlyStopping(patience=self.run_args.early_stopping_patience,
                                               score_function=self.score_function, trainer=self.trainer)

                if 'val' in iterators:
                    self.validator.add_event_handler(Events.COMPLETED, early_stopping)
                else:
                    self.tester.add_event_handler(Events.COMPLETED, early_stopping)

            # Start training run if the number of epochs is greater than zero
            self.trainer.run(iterators['train'], max_epochs=self.initial_epoch + self.run_args.epochs)

            return list(metric_history.values.values())
        elif self.run_args.epochs < 0:
            # Run evaluation on all saved weights
            self.logger.info('Run evaluation on validation and test set with all saved weights!')
            weight_files = glob.glob(os.path.join(self.run_args.session_dir, 'weights_*.pt'))
            epochs = [int(os.path.splitext(os.path.basename(weight_file))[0].split('_')[1]) for weight_file in
                      weight_files]
            visdom_plot.win += '_replay'

            for epoch, weight_file in sorted(list(zip(epochs, weight_files)), key=lambda x: x[0]):
                self.logger.info('Load weights from epoch {}'.format(epoch))
                self.model.load_state_dict(torch.load(weight_file), strict=False)
                self.run_evaluations_handler(None, iterators)
                visdom_plot.plot()

            return list(metric_history.values.values())
        else:
            # Start only the evaluation run manually
            self.logger.info('Run evaluation on validation and test set with most recent weights!')
            self.run_evaluations_handler(None, iterators)
            correlation_metrics = None

            if self.run_args.rank_evaluation.lower() == 'absolute' and 'test' in iterators:
                correlation_metrics = run_absolute_rank_evaluation(self.model, fields, self.run_args.session_dir,
                                                                   phi=self.phi, device=self.device,
                                                                   test_ids=iterators['test'].article_ids)
            elif self.run_args.rank_evaluation.lower() == 'pairwise' and 'test' in iterators:
                correlation_metrics = run_pairwise_rank_evaluation(self.model, fields, self.run_args.session_dir,
                                                                   phi=self.phi, device=self.device,
                                                                   test_ids=iterators['test'].article_ids)

            return correlation_metrics

    @classmethod
    def from_args(cls, args=None):
        run_args, model_args = Runner.parse_run_args(args)

        if run_args.k_fold is not None:
            if '/' in run_args.k_fold:
                folds, k = run_args.k_fold.split('/')
                folds = [int(f) for f in folds.split(',')]
                k = int(k)
            else:
                folds, k = list(range(1, int(run_args.k_fold) + 1)), int(run_args.k_fold)

            runners = []

            for f in folds:
                fold_run_args = deepcopy(run_args)
                fold_run_args_dict = vars(fold_run_args)
                fold_run_args_dict['session_dir'] = os.path.join(fold_run_args_dict['session_dir'], '{:02d}'.format(f))
                fold_run_args_dict['k_fold'] = f, k
                runners.append((fold_run_args, model_args))

            return runners
        else:
            return [(run_args, model_args)]


if __name__ == '__main__':
    runner = None
    metrics = {}
    metric_key = None

    for _run_args, _model_args in Runner.from_args():
        runner = Runner(_run_args, _model_args)
        results = runner.start()

        if results is None:
            runner.logger.warning('Runner returned no results at the end!')
            continue

        if isinstance(results, list):
            current_metrics = {}

            for r in results:
                current_metrics.update(r)
        else:
            current_metrics = results

        for metric_name, metric_values in current_metrics.items():
            if len(metric_values) == 0:
                continue

            if metric_name not in metrics:
                metrics.update({metric_name: {'all': [], 'peaks': [], 'epoch': []}})

            if 'loss' in metric_name:
                metrics[metric_name]['all'].append(metric_values)
                metrics[metric_name]['peaks'].append(np.min(metric_values))
                metrics[metric_name]['epoch'].append(np.argmin(metric_values) + runner.initial_epoch + 1)
            else:
                metrics[metric_name]['all'].append(metric_values)
                metrics[metric_name]['peaks'].append(np.max(metric_values))
                metrics[metric_name]['epoch'].append(np.argmax(metric_values) + runner.initial_epoch + 1)

    if runner is not None:
        headers = ['metric', 'peak avg', 'peak values', 'epoch']
        table = []

        for metric_name, metric_value_dict in metrics.items():
            table.append([metric_name, '{:.2f}'.format(np.mean(metric_value_dict['peaks'])),
                          ', '.join('{:.2f}'.format(v) for v in metric_value_dict['peaks']),
                          ', '.join('{}'.format(v) for v in metric_value_dict['epoch'])])

        runner.logger.info('\n' + tabulate(table, headers=headers))
