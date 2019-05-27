import os
import sys
import pandas as pd
import argparse
from copy import deepcopy
from collections import Counter
from ast import literal_eval
from tqdm import tqdm
from torch.utils.data import SequentialSampler, BatchSampler

def get_batches(filename, batch_size):
    coh_ixs = []
    acts1 = []
    acts2 = []
    utts1 = []
    utts2 = []

    with open(filename, 'r') as f:
        coh_df = pd.read_csv(f, sep='|', names=['coh_idx', 'acts1', 'utts1', 'acts2', 'utts2'])

    for (idx, row) in tqdm(coh_df.iterrows(), desc="Data Loading"):
        a1 = [int(x) for x in row['acts1'].split(' ')]
        acts1.append(a1)
        a2 = [int(x) for x in row['acts2'].split(' ')]
        acts2.append(a2)

        utts1.append(literal_eval(row['utts1']))
        utts2.append(literal_eval(row['utts2']))

        coh_ixs.append(int(row['coh_idx']))

    batch_indices = list(BatchSampler(SequentialSampler(range(len(acts1))), batch_size=batch_size, drop_last=False))
    batches = []
    for batch_ixs in batch_indices:
        batch = []
        for i in batch_ixs:
            batch.append((coh_ixs[i],
                acts1[i], utts1[i],
                acts2[i], utts2[i]))
        batches.append(batch)

    return batches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",
                        required=True,
                        type=str,
                        help="""The input directory where the files of daily
                        dialog are located. the folder should have
                        train/test/validation as subfolders""")
    parser.add_argument('--task',
                        required=True,
                        type=str,
                        default="up",
                        help="""for which task the dataset should be created.
                                alternatives: up (utterance permutation)
                                              us (utterance sampling)
                                              hup (half utterance petrurbation) """)
    parser.add_argument('--batch_size',
                        required=False,
                        type=int,
                        default=16)
    args = parser.parse_args()

    data_file = os.path.join(args.datadir, "coherency_dset_{}_shuf.txt".format(args.task))
    assert os.path.isfile(data_file), "could not find dataset file: {}".format(data_file_shuf)

    batches = get_batches(data_file, args.batch_size)

    for i, batch in enumerate(batches):
        print("{}'th Batch:".format(i))
        print(batch)

        if i > 3: break
