import os
import logging
import datetime
import argparse
import numpy as np
import argparse
import torch
import torch.nn as nn
import torchtext as tt
from nltk import word_tokenize
from tqdm import tqdm, trange
from sklearn.metrics import f1_score

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

hidden_size = 300
num_layers = 2
batch_size = 4
max_seq_len = 294
num_classes = 4
warmup_proportion = 0.1
learning_rate = 5e-5
num_epochs = 2
output_dir = '/home/sebi/code/transfer_rewards/sub_rewards/data/'

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
    }

class ELMo_BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ELMo_BiRNN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

        # Init Weights
        torch.nn.init.normal_(self.fc.weight, mean=0, std=1)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2) 
        # Inputs: input, (h_0, c_0)
        # From the DOC: input of shape (seq_len, batch, input_size): tensor
        # containing the features of the input sequence. 

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class DL_Text_It():
    def __init__(self, dataset, batch_size = 32, max_len=-1):
        # assert isinstance(dataset, "<class 'torchtext.data.dataset.TabularDataset'>"), "The dataset is of wrong Type!"
        self.dataset = dataset
        self.max_len = len(dataset) if max_len < 0 else max_len
        self.current_pos = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.max_len / self.batch_size + 0.5)
    
    def reset(self):
        self.current_pos = 0

    def __next__(self):
        if self.current_pos >= self.max_len:
            raise StopIteration
        size = self.current_pos+self.batch_size if self.current_pos+batch_size < self.max_len else self.max_len
        text_batch = []
        label_batch = []
        for i in range(self.current_pos, size):
            text_batch.append(self.dataset[i].Text)
            label_batch.append(int(self.dataset[i].Label)-1)
            # the '-1' is to avoid: Assertion `cur_target >= 0 && cur_target < n_classes' failed 
        self.current_pos += self.batch_size 
        return text_batch, label_batch

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
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_model_file = os.path.join(args.output_dir, 'elmo_model.ckpt')
    
    # Logging file
    now = datetime.datetime.now()
    logfile = os.path.join(args.logdir, 'CNN_{}.log'.format(now.strftime("%Y-%m-%d_%H:%M:%S")))
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG, format='%(levelname)s:%(message)s')
    print("Logging to ", logfile)

    # Log all Hyperparameters
    logging.info("Used Hyperparameters:")
    logging.info("hidden_size = {}".format(hidden_size))
    logging.info("num_layers = {}".format(num_layers))
    logging.info("batch_size = {}".format(batch_size))
    logging.info("max_seq_len = {}".format(max_seq_len))
    logging.info("num_classes = {}".format(num_classes))
    logging.info("warmup_proportion = {}".format(warmup_proportion))
    logging.info("learning_rate = {}".format(learning_rate))
    logging.info("num_epochs = {}".format(num_epochs))
    
    TEXT = tt.data.Field(sequential=True, tokenize=word_tokenize, use_vocab=False)
    LABEL = tt.data.Field(sequential=False, use_vocab=False)
    train, val, test = tt.data.TabularDataset.splits(
        path='./data/daily_dialog/', train='train/act_utt.txt',
        validation='validation/act_utt.txt', test='test/act_utt.txt', format='csv', csv_reader_params={'delimiter':'|'},
        fields=[('Label', LABEL), ('Text', TEXT)])
    
    # Compute two different representation for each token.
    # Each representation is a linear weighted combination for the
    # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    # char_ids = batch_to_ids([train[0].Text, train[1].Text, train[2].Text])
    # embeddings = elmo(char_ids)
    # print(embeddings, embeddings['elmo_representations'][0].size())

    model = None

    if args.do_train:
        train_iter = DL_Text_It(train, batch_size)
        model = ELMo_BiRNN(1024, hidden_size, num_layers, num_classes)
        model.to(device)
        model.train()
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for _ in trange(num_epochs, desc="Epoch"):
            for step, (text_b, lbl_b) in tqdm(enumerate(train_iter), desc="Iteration", total=len(train_iter)):
                char_ids = batch_to_ids(text_b)
                embedding = elmo(char_ids)['elmo_representations'][0].to(device)
                torch_labels = torch.tensor(lbl_b, dtype=torch.long).to(device)

                outputs = model(embedding)
                loss = criterion(outputs, torch_labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_iter.reset()

        torch.save(model.state_dict(), output_model_file)
        logging.info("Saved learned model in file: {}".format(output_model_file))

    if args.do_eval:
        if not args.do_train:
            assert os.path.isfile(output_model_file), "the learnend model file does not exist, execute with --do_train first"
            model.load_state_dict(torch.load(output_model_file))
            model.to(device)
        model.eval()
        val_iter = DL_Text_It(val, batch_size)
        
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        all_labels = []
        # Loss function
        loss_fct = torch.nn.CrossEntropyLoss()
        
        for step, (text_b, lbl_b) in tqdm(enumerate(val_iter), desc="Validation"):
            char_ids = batch_to_ids(text_b)
            embedding = elmo(char_ids)['elmo_representations'][0].to(device)
            torch_labels = torch.tensor(lbl_b, dtype=torch.long).to(device)
            all_labels = all_labels + lbl_b
            with torch.no_grad():
                logits = model(embedding)

            tmp_eval_loss = loss_fct(logits.view(-1, num_classes), torch_labels.view(-1))
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds[0], axis=1)
        result = acc_and_f1(preds, np.array(all_labels))
        print(result)
        logging.info("Final Evaluation Result: {}".format(result))


if __name__ == '__main__':
    main()
