import os
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchtext as tt
import torch.nn.functional as F
from torch.autograd import Variable
from nltk import word_tokenize
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
import traceback
import logging

embed_dim = 128
kernel_num = 100
kernel_sizes = (3,4,5)
static = False
dropout = 0.5
num_layers = 2
batch_size = 4
max_seq_len = 294
num_classes = 4
warmup_proportion = 0.1
learning_rate = 5e-5
num_epochs = 50
output_dir = '/home/sebi/code/transfer_rewards/sub_rewards/data/'

class CNN_Text(nn.Module):
    # copied from https://github.com/Shawn1993/cnn-text-classification-pytorch
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static=False):
        super(CNN_Text, self).__init__()
        V = embed_num
        D = embed_dim
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        self.static = static

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

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
    logging.info("embed_dim = {}".format(embed_dim))
    logging.info("kernel_num = {}".format(kernel_num))
    logging.info("kernel_sizes = {}".format(kernel_sizes))
    logging.info("static = {}".format(static))
    logging.info("dropout = {}".format(dropout))
    logging.info("num_layers = {}".format(num_layers))
    logging.info("batch_size = {}".format(batch_size))
    logging.info("max_seq_len = {}".format(max_seq_len))
    logging.info("num_classes = {}".format(num_classes))
    logging.info("warmup_proportion = {}".format(warmup_proportion))
    logging.info("learning_rate = {}".format(learning_rate))
    logging.info("num_epochs = {}".format(num_epochs))
    logging.info("========================")

    # preprocess to have all sents padded up to length at least 6
    preprocess = lambda x: x if len(x) >= 6 else x + (['<pad>']*(5-len(x)))

    TEXT = tt.data.Field(sequential=True, tokenize=word_tokenize, batch_first=True, preprocessing=preprocess)
    LABEL = tt.data.Field(sequential=False, use_vocab=False, batch_first=True)
    train, val, test = tt.data.TabularDataset.splits(
        path='./data/daily_dialog/', train='train/act_utt.txt',
        validation='validation/act_utt.txt', test='test/act_utt.txt', format='csv', csv_reader_params={'delimiter':'|'},
        fields=[('Label', LABEL), ('Text', TEXT)])
    
    TEXT.build_vocab(val, train, test)
    train_iter, val_iter, test_iter = tt.data.Iterator.splits(
        (train, val, test), sort_key= None, sort=False, #lambda x: len(x.Text),
        batch_size=batch_size, device=device)
    print("training batches: ", len(train_iter), ", training datapoints: ", len(train))
    model = None

    id2word = { v : k for k,v in TEXT.vocab.stoi.items()}

    # the labels are 1,2,3,4, but the model will predict 0,1,2,3. to avoid off-by-one, add this to all targets
    # ones = torch.ones(batch_size, dtype=torch.long)*(-1)

    if args.do_train:
        model = CNN_Text(len(TEXT.vocab), embed_dim, num_classes, kernel_num, kernel_sizes, dropout)
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        except_count = 0
        
        for _ in trange(num_epochs, desc="Epoch"):
            for step, batch in tqdm(enumerate(train_iter), desc="Iteration", total=len(train_iter)):
                try:
                    ones = torch.ones(batch.Label.size(0), dtype=torch.long)*(-1)
                    feature, target = batch.Text, batch.Label.add(ones)
                    outputs = model(feature)
                    loss = criterion(outputs, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    logging.error(traceback.format_exc())
                    f_text = [id2word[w.item()] for w in feature[0]]
                    logging.error("feature: " + str(f_text))
                    logging.error("target vector:" + str(target))
                    except_count += 1

        torch.save(model.state_dict(), output_model_file)
        logging.info("Saved learned model in file: {}".format(output_model_file))

    if args.do_eval:
        if not args.do_train:
            assert os.path.isfile(output_model_file), "the learnend model file does not exist, execute with --do_train first"
            model = CNN_Text(len(TEXT.vocab), embed_dim, num_classes, kernel_num, kernel_sizes, dropout)
            model.load_state_dict(torch.load(output_model_file))
            model.to(device)
        model.eval()

        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        all_labels = []

        # Loss function
        loss_fct = torch.nn.CrossEntropyLoss()
        
        for step, batch in tqdm(enumerate(val_iter), desc="Validation", total=len(val_iter)):
            try:
                ones = torch.ones(batch.Label.size(0), dtype=torch.long)*(-1)
                feature, target = batch.Text, batch.Label.add(ones)
                all_labels = all_labels + target.numpy().tolist()

                with torch.no_grad():
                    logits = model(feature)

                tmp_eval_loss = loss_fct(logits, target)
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
            except Exception as e:
                logging.error(traceback.format_exc())
                f_text = [id2word[w.item()] for w in feature[0]]
                logging.error("feature: " + str(f_text))
                logging.error("target vector:" + str(target))

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds[0], axis=1)
        result = acc_and_f1(preds, np.array(all_labels))
        print(result)
        logging.info("Final Evaluation Result: {}".format(result))

if __name__ == '__main__':
    main()
