#-*- coding: utf-8 -*-

from __future__ import print_function

from gensim.models import KeyedVectors
from data_reader import Data_Reader
import data_parser
import config

from model import Seq2Seq_chatbot
import tensorflow as tf
import numpy as np

import os
import time


import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = logging.INFO)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger("tensorflow").setLevel(logging.ERROR)


logger = logging.getLogger(__name__)



### Global Parameters ###
checkpoint = config.CHECKPOINT
model_path = config.train_model_path
model_name = config.train_model_name
if config.reverse:
    model_path = config.reversed_model_path
    model_name = config.reversed_model_name
start_epoch = config.start_epoch

word_count_threshold = config.WC_threshold

### Train Parameters ###
dim_wordvec = 300
dim_hidden = 1000

n_encode_lstm_step = 22 + 22
n_decode_lstm_step = 22

epochs = 200
batch_size = 100
learning_rate = 0.01

wordtoix, ixtoword, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def step(sess, outputs, inp_list, batch_X, batch_Y):
    for i in range(len(batch_X)):
        batch_X[i] = [word_vector[w] if w in word_vector else np.zeros(dim_wordvec) for w in batch_X[i]]
        # batch_X[i].insert(0, np.random.normal(size=(dim_wordvec,))) # insert random normal at the first step
        if len(batch_X[i]) > n_encode_lstm_step:
            batch_X[i] = batch_X[i][:n_encode_lstm_step]
        else:
            for _ in range(len(batch_X[i]), n_encode_lstm_step):
                batch_X[i].append(np.zeros(dim_wordvec))

    current_feats = np.array(batch_X)

    current_captions = batch_Y
    current_captions = map(lambda x: '<bos> ' + x, current_captions)
    current_captions = map(lambda x: x.replace('.', ''), current_captions)
    current_captions = map(lambda x: x.replace(',', ''), current_captions)
    current_captions = map(lambda x: x.replace('"', ''), current_captions)
    current_captions = map(lambda x: x.replace('\n', ''), current_captions)
    current_captions = map(lambda x: x.replace('?', ''), current_captions)
    current_captions = map(lambda x: x.replace('!', ''), current_captions)
    current_captions = map(lambda x: x.replace('\\', ''), current_captions)
    current_captions = map(lambda x: x.replace('/', ''), current_captions)

    current_captions = list(current_captions)    

    for idx, each_cap in enumerate(current_captions):
        word = each_cap.lower().split(' ')
        if len(word) < n_decode_lstm_step:
            current_captions[idx] = current_captions[idx] + ' <eos>'
        else:
            new_word = ''
            for i in range(n_decode_lstm_step-1):
                new_word = new_word + word[i] + ' '
            current_captions[idx] = new_word + '<eos>'

    current_caption_ind = []
    for cap in current_captions:
        current_word_ind = []
        for word in cap.lower().split(' '):
            if word in wordtoix:
                current_word_ind.append(wordtoix[word])
            else:
                current_word_ind.append(wordtoix['<unk>'])
        current_caption_ind.append(current_word_ind)

    current_caption_matrix = pad_sequences(current_caption_ind, padding='post', maxlen=n_decode_lstm_step)
    current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)
    current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
    nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix)))

    for ind, row in enumerate(current_caption_masks):

        row[:nonzeros[ind]] = 1

    [word_vectors, tf_caption, tf_caption_mask] = inp_list

    loss_val = sess.run(
            outputs,
            feed_dict= {
                        word_vectors: current_feats,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
            })

    loss_val = loss_val[-1] # we assume loss is the last item

    return loss_val




def train():

    model = Seq2Seq_chatbot(
            dim_wordvec=dim_wordvec,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector,
            lr=learning_rate)


    train_op, tf_loss, word_vectors, tf_caption, tf_caption_mask, inter_value = model.build_model()

    saver = tf.train.Saver(max_to_keep=100)

    sess_config = tf.ConfigProto()

    sess_config.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=sess_config)
    
    inp_list = [word_vectors, tf_caption, tf_caption_mask] 

    if checkpoint:
        logger.info("Use Model {}.".format(model_name))
        saver.restore(sess, os.path.join(model_path, model_name))
        logger.info("Model {} restored.".format(model_name))
    else:
        logger.info("Restart training...")
        tf.global_variables_initializer().run()

    train_dr = Data_Reader(config.training_data_path,shuffle=True)
    valid_dr = Data_Reader(config.valid_data_path)

    best_valid_loss = float("inf")

    best_epoch = 0

    best_batch = 0

    for epoch in range(start_epoch, epochs):
        
        epoch_loss = 0.0

        n_batch = train_dr.get_batch_num(batch_size)

        for t_batch in range(n_batch):
        
            start_time = time.time()

            batch_X, batch_Y = train_dr.generate_batch(batch_size)

            if config.reverse:
                
                tmp_Y = []

                for item in batch_X:

                    tmp_Y.append(' '.join(item))

                tmp_X = []

                for item in batch_Y:

                    tmp_X.append(item.split()) 

                batch_X = tmp_X

                batch_Y = tmp_Y


            loss_val = step(sess, [train_op, tf_loss], inp_list, batch_X, batch_Y)

            epoch_loss += loss_val

            previous_valid_loss = 0.0

            if t_batch % 100 == 0:

                valid_n_batch = valid_dr.get_batch_num(batch_size)

                valid_loss = 0.0
        
                for v_batch in range(valid_n_batch):

                    batch_X, batch_Y = valid_dr.generate_batch(batch_size)

                    loss_val = step(sess, [tf_loss], inp_list, batch_X, batch_Y)

                    valid_loss += loss_val


                logger.info("Epoch: %d, batch: %d/%d, train_loss: %.4f, valid_loss:%.4f"%(epoch, t_batch, n_batch, epoch_loss/float(t_batch), valid_loss/float(v_batch)))

                if valid_loss < best_valid_loss: 

                    best_epoch = epoch

                    best_batch = t_batch

                    saver.save(sess, os.path.join(model_path, 'model-best'))


        if epoch % config.checkpoint_step ==0:

            logger.info("Epoch %d is done. Saving the model ..."%epoch)
            
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

    logger.info('best model happend in epoch: %d, batch: %d'%(best_epoch, best_batch))

if __name__ == "__main__":

    logger.info(config)
    
    train()
