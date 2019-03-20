# path to training data
training_data_path = 'data/train.pkl'
valid_data_path = 'data/valid.pkl'
test_data_path = 'data/test.pkl'

# path to all_words
all_words_path = 'data/all_words.txt'

# training parameters 
CHECKPOINT = False # True: use the saved model, False: restart training from scratch
train_model_path = 'model/seq2seq/'
train_model_name = 'model-55'
start_epoch = 0#56
start_batch = 0
batch_size = 25
checkpoint_step=10 # save the model after howmany epohs
valid_step= 1 # aftte how many epochs evaluate on the valid set

# for RL training
training_type = 'normal' # 'normal' for seq2seq training, 'pg' for policy gradient
reversed_model_path = 'Adam_encode22_decode22_reversed-maxlen22_lr0.0001_batch25_wordthres6'
reversed_model_name = 'model-63'

# data reader shuffle index list
load_list = False
index_list_file = 'data/shuffle_index_list'
cur_train_index = start_batch * batch_size

# word count threshold
WC_threshold = 20
reversed_WC_threshold = 6

# dialog simulation turns
MAX_TURNS = 10