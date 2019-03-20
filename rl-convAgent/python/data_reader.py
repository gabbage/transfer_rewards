# coding=utf-8

from __future__ import print_function
import pickle as pickle
import config
import random
random.seed(1234)

class Data_Reader:
    def __init__(self, data_path, shuffle=False, cur_index=0):
        self.data = pickle.load(open(data_path, 'rb'))

        self.data_size = len(self.data)
        if shuffle:
            self.shuffle_list = self.shuffle_index()
        else:
            self.shuffle_list = range(self.data_size)
        self.index = cur_index

    def get_batch_num(self, batch_size):
        return self.data_size // batch_size

    def shuffle_index(self):
        shuffle_index_list = random.sample(range(self.data_size), self.data_size)
        pickle.dump(shuffle_index_list, open(config.index_list_file, 'wb'), True)
        return shuffle_index_list

    def generate_batch_index(self, batch_size):
        if self.index + batch_size > self.data_size:
            batch_index = self.shuffle_list[self.index:self.data_size]
            self.shuffle_list = self.shuffle_index()
            remain_size = batch_size - (self.data_size - self.index)
            batch_index += self.shuffle_list[:remain_size]
            self.index = remain_size
        else:
            batch_index = self.shuffle_list[self.index:self.index+batch_size]
            self.index += batch_size

        return batch_index

    def generate_batch(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.data[i][0] for i in batch_index]   # batch_size of conv_a
        batch_Y = [self.data[i][1] for i in batch_index]   # batch_size of conv_b

        return batch_X, batch_Y

    def generate_batch_with_former(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.data[i][0] for i in batch_index]   # batch_size of conv_a
        batch_Y = [self.data[i][1] for i in batch_index]   # batch_size of conv_b
        former = [self.data[i][2] for i in batch_index]    # batch_size of former utterance

        return batch_X, batch_Y, former

    # def generate_testing_batch(self, batch_size):
    #     batch_index = self.generate_batch_index(batch_size)
    #     batch_X = [self.training_data[i][0] for i in batch_index]   # batch_size of conv_a
    #     return batch_X

    # def generate_valid_batcg(self, batch_size):
    #     batch_index = self.generate_batch_index(batch_size)