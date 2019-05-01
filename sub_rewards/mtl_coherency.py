import os
import datetime
import logging
import sys
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel,BertPreTrainedModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME

### Hyper Parameters ###
BERT_MODEL_NAME = "bert-large-uncased"
batch_size = 32

########################

def load_data(data_dir, dataset):
    """ dataset needs to be either 'train', 'test', 'validation' """
    data_file = os.path.join(data_dir, dataset, "coherency_dset.txt")
    #shuffled file, s.t. the permuted and random inserted datapoints are not directly next to the original
    data_file_shuf = os.path.join(data_dir, dataset, "coherency_dset_shuf.txt")
    assert os.path.isfile(data_file), "could not find dataset file"

class MTL_Loss(nn.Module):
    # see: https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
    def __init__(self):
        super(MTL_Loss,self).__init__()
        self.sigma1 = torch.tensor(1.0, requires_grad=True)
        self.sigma2 = torch.tensor(1.0, requires_grad=True)

    """ both x and y need to be torch Variables ! """
    def forward(self,x,y):
        #TODO: define the loss function with the two sigmas based on MSE and crossentropy
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputdir",
                        default=output_dir,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--datadir",
                        # required=True,
                        type=str,
                        help="""The input directory where the files of daily
                        dialog are located. the folder should have
                        train/test/validation as subfolders""")
    parser.add_argument("--logdir",
                        default="/home/sebi/code/transfer_rewards/sub_rewards",
                        type=str,
                        help="the folder to save the logfile to.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    # Init randomization
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_model_file = os.path.join(args.outputdir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.outputdir, CONFIG_NAME)

    # Logging file
    now = datetime.datetime.now()
    logfile = os.path.join(args.logdir, 'MTL_{}.log'.format(now.strftime("%Y-%m-%d_%H:%M:%S")))
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG, format='%(levelname)s:%(message)s')
    print("Logging to ", logfile)

    logging.info("Used Hyperparameters:")
    logging.info("BERT_MODEL_NAME = {}".format(BERT_MODEL_NAME))
    logging.info("batch_size = {}".format(batch_size))

if __name__ == '__main__':
    main()
