import itertools
import psutil
import os
import time
import threading

class PipelineThread():
    def __init__(self, args_str):
        self.thread = threading.Thread(target=self.run, args=([args_str]))
        self.thread.daemon = True

    def start(self):
        self.thread.start()

    def isAlive(self):
        self.thread.isAlive()

    def run(self, args):
        command = 'hand_craft.py {}'.format(args)
        print('command: '+command)
        os.system('python '+command)
        #pipeline.pipeline(args)

def gridGenerate(feature,sample):
    max_load = 8
    min_mem = 1000000000  # 1 GB
    args = '{} {} False'.format(feature,sample)
    tt = PipelineThread(args)
    while True:
        mem = psutil.virtual_memory()[1]
        if os.name is 'posix':
            current_load = float(os.getloadavg()[0])
        else:
            current_load = 0
        if current_load < max_load - 1 and mem >= min_mem:
            tt.start()
            time_to_sleep = 10
            print('Sleep for {} minutes to update the load'.format(time_to_sleep / 60))
            time.sleep(time_to_sleep)
            break
        else:
            time_to_sleep = 3 * 60
            print('Too busy now; wait for {} minutes and try the next setting'.format(time_to_sleep / 60))
            time.sleep(time_to_sleep)

if __name__ == '__main__':
    features_list = ['js','rouge_1','rouge_2','tfidf_cos_avg','redundancy_1_2']
    sample_files = ['1_100.p','2_100.p','3_100.p','4_100.p','5_100.p']
    total_list = [features_list, sample_files]
    cnt = 0
    for args in list(itertools.product(*total_list)):
        feature = args[0]
        sample = args[1]
        cnt += 1
        print('no. {}, feature {}, sample file {}'.format(cnt,feature,sample))
        gridGenerate(feature,sample)

