import os
import datetime
import logging
import sys
import argparse
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext as tt
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from torch.nn.modules.distance import CosineSimilarity
from torch.optim.lr_scheduler import MultiStepLR
from matplotlib import pyplot as plt

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel,BertPreTrainedModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear


BERT_MODEL_NAME = "bert-large-uncased"
batch_size = 32
max_seq_len = 295
num_classes = 4
warmup_proportion = 0.1
learning_rate = 5e-2
num_epochs = 6
lr_schedule = [2,3,5]
output_dir = '/home/sebi/code/transfer_rewards/sub_rewards/data/'

class BertFF(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob):
        super(BertFF, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        # self.apply(self.init_bert_weights)
        torch.nn.init.normal_(self.classifier.weight, mean=0, std=1)
        print("classifier weight")
        print(self.classifier.weight)

    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

def convert(dset, max_seq_len, word2id):
    token_ids = []
    token_masks = []
    labels = []

    for i in range(len(dset)):
    # for i in range(200):
        tok = ["[CLS]"] + [t for t in dset[i].Text]
        tok_ids = word2id(tok)
        padding = [0] * (max_seq_len - len(tok_ids))
        mask = [1] * len(tok_ids) + padding
        tok_ids = tok_ids + padding

        token_ids.append(tok_ids)
        token_masks.append(mask)

        labels.append(int(dset[i].Label) - 1)

    return token_ids, token_masks, labels

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir",
                        default=output_dir,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
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

    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    # Logging file
    now = datetime.datetime.now()
    logfile = os.path.join(args.logdir, 'BERT_{}.log'.format(now.strftime("%Y-%m-%d_%H:%M:%S")))
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG, format='%(levelname)s:%(message)s')
    print("Logging to ", logfile)

    # Log all Hyperparameters
    logging.info("Used Hyperparameters:")
    logging.info("BERT_MODEL_NAME = {}".format(BERT_MODEL_NAME))
    logging.info("batch_size = {}".format(batch_size))
    logging.info("max_seq_len = {}".format(max_seq_len))
    logging.info("num_classes = {}".format(num_classes))
    logging.info("warmup_proportion = {}".format(warmup_proportion))
    logging.info("learning_rate = {}".format(learning_rate))
    logging.info("num_epochs = {}".format(num_epochs))
    logging.info("random_seed = {}".format(args.seed))
    logging.info("lr_schedule = {}".format(lr_schedule))
    logging.info("========================")

    # BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=True)

    # Device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    TEXT = tt.data.Field(sequential=True, tokenize=lambda x:tokenizer.tokenize(x))
    LABEL = tt.data.Field(sequential=False, use_vocab=False)
    train, val, test = tt.data.TabularDataset.splits(
        path='./data/daily_dialog/', train='train/act_utt.txt',
        validation='validation/act_utt.txt', test='test/act_utt.txt', format='csv', csv_reader_params={'delimiter':'|'},
        fields=[('Label', LABEL), ('Text', TEXT)])

    model = None

    # Model configuration
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))

    if args.do_train:
        train_ids, train_masks, train_labels = convert(train, max_seq_len, lambda x: tokenizer.convert_tokens_to_ids(x))
        torch_train_ids = torch.tensor(train_ids, dtype=torch.long)
        torch_train_masks = torch.tensor(train_masks, dtype=torch.long)
        torch_train_labels = torch.tensor(train_labels, dtype=torch.long)

        train_data = TensorDataset(torch_train_ids, torch_train_masks, torch_train_labels)
        # take these for gpu processing
        # train_sampler = DistributedSampler(train_data)
        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        train_dataloader = DataLoader(train_data, batch_size=batch_size)


        bert = BertModel.from_pretrained(BERT_MODEL_NAME,
                  cache_dir=cache_dir).to(device)
        model = BertFF(bert.config.hidden_size, num_classes, 0.1).to(device)

        model.train()
        bert.eval()

        # plt
        loss_vals = np.array([])

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = MultiStepLR(optimizer, milestones=lr_schedule, gamma=0.1)
        # num_train_optimization_steps = int(len(train_data) / batch_size * num_epochs)
        # param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
            # {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            # {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            # ]
        # optimizer = BertAdam(optimizer_grouped_parameters,
                                 # lr=learning_rate,
                                 # t_total=num_train_optimization_steps)

        # Loss function
        loss_fct = torch.nn.CrossEntropyLoss()

        # live data file
        live_data = open('live_data.csv', 'w', buffering=1)
        live_data.write("{},{},{},{},{}\n".format('step', 'loss', 'avg_loss', 'acc', 'f1'))

        for epoch in trange(num_epochs, desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            scheduler.step()
            nb_eval_steps = 0
            preds = []

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, labels = batch

                with torch.no_grad():
                    _, pooled_output = bert(input_ids, token_type_ids=None, attention_mask=input_mask)
                logits = model(pooled_output)
                loss = loss_fct(logits.view(-1, num_classes), labels.view(-1))

                loss.backward()
                loss_vals = np.append(loss_vals, [loss.item()])

                #write to live data file
                if step >= 1:
                    result = acc_and_f1(np.argmax(preds[0], axis=1), torch_train_labels.numpy()[0:len(preds[0])])
                    live_data.write("{},{},{},{},{}\n".format((epoch*len(train_dataloader))+step, loss.item(), loss_vals[-100:-1].mean(), result['acc'], result['f1']))
                    live_data.flush()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()

                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)


        torch.save(model.state_dict(), output_model_file)
        logging.info("saved model in file: {}".format(output_model_file))
        with open(output_config_file, 'w') as f:
            f.write(bert.config.to_json_string())
            logging.info("saved BERT config in file: {}".format(output_config_file))

    if args.do_eval:
        if not args.do_train:
            assert os.path.isfile(output_model_file), "the learnend model file does not exist, execute with --do_train first"
            config = BertConfig(output_config_file)
            bert = BertModel.from_pretrained(BERT_MODEL_NAME,
                      cache_dir=cache_dir).to(device)
            model = BertFF(bert.config.hidden_size, num_classes, 0.1)
            model.load_state_dict(torch.load(output_model_file))
        model.eval()
        model.to(device)

        val_ids, val_masks, val_labels = convert(val, max_seq_len, lambda x: tokenizer.convert_tokens_to_ids(x))
        torch_val_ids = torch.tensor(val_ids, dtype=torch.long)
        torch_val_masks = torch.tensor(val_masks, dtype=torch.long)
        torch_val_labels = torch.tensor(val_labels, dtype=torch.long)

        val_data = TensorDataset(torch_val_ids, torch_val_masks, torch_val_labels)
        # take these for gpu processing
        # eval_sampler = SequentialSampler(eval_data)
        # train_dataloader = DataLoader(train_data, sampler=eval_sampler, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)

        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        # Loss function
        loss_fct = torch.nn.CrossEntropyLoss()

        for _, batch in enumerate(tqdm(val_dataloader, desc="Evaluating")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            with torch.no_grad():
                _, pooled_output = bert(input_ids, token_type_ids=None, attention_mask=input_mask)
                logits = model(pooled_output)
            tmp_eval_loss = loss_fct(logits.view(-1, num_classes), labels.view(-1))
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)


        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds[0], axis=1)
        result = acc_and_f1(preds, torch_val_labels.numpy())
        print(result)
        logging.info("Final Evaluation Result: {}".format(result))

if __name__ == '__main__':
    main()
