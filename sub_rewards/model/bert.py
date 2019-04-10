import os
import datetime
import logging
import sys
import argparse
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm, trange
import torch
import torchtext as tt
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear


BERT_MODEL_NAME = "bert-base-uncased"
batch_size = 4
max_seq_len = 294
num_classes = 4
warmup_proportion = 0.1
learning_rate = 5e-5
num_epochs = 2
output_dir = '/home/sebi/code/transfer_rewards/sub_rewards/data/'

def convert(dset, max_seq_len, word2id):
    token_ids = []
    token_masks = []
    labels = []

    # for i in range(len(dset)):
    for i in range(10):
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
    logging.info("========================")

    # BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TEXT = tt.data.Field(sequential=True, tokenize=lambda x:tokenizer.tokenize(x))
    LABEL = tt.data.Field(sequential=False, use_vocab=False)
    train, val, test = tt.data.TabularDataset.splits(
        path='./data/daily_dialog/', train='train/act_utt.txt',
        validation='validation/act_utt.txt', test='test/act_utt.txt', format='csv', csv_reader_params={'delimiter':'|'},
        fields=[('Label', LABEL), ('Text', TEXT)])
    model = None

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


        # Model configuration
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME,
                  cache_dir=cache_dir,
                  num_labels=num_classes).to(device)
        model.train()

        # Optimizer
        num_train_optimization_steps = int(len(train_data) / batch_size * num_epochs)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=learning_rate,
                                 warmup=warmup_proportion,
                                 t_total=num_train_optimization_steps)

        # Loss function
        loss_fct = torch.nn.CrossEntropyLoss()

        for _ in trange(num_epochs, desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, labels = batch

                logits = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=None)
                loss = loss_fct(logits.view(-1, num_classes), labels.view(-1))

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()

        torch.save(model.state_dict(), output_model_file)
        logging.info("saved model in file: {}".format(output_model_file))
        with open(output_config_file, 'w') as f:
            f.write(model.config.to_json_string())
            logging.info("saved BERT config in file: {}".format(output_config_file))

    if args.do_eval:
        if not args.do_train:
            assert os.path.isfile(output_model_file), "the learnend model file does not exist, execute with --do_train first"
            config = BertConfig(output_config_file)
            model = BertForSequenceClassification(config, num_labels=num_classes)
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
                logits = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=None)
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
