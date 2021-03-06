import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pandas as pd
import numpy as np
import os

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()

rsync_cmd = "rsync --inplace -a --progress buecker@10.167.11.61:transfer_rewards/sub_rewards/live_data_us.csv logs/live_server.csv"

def animate(i):
    #retrieve data from server
    os.system(rsync_cmd)
    only_last_batch=False

    df = pd.read_csv('logs/live_server.csv', names=['step', 'loss', 'score', 'da_score'], header=1)
    ax1.clear()
    ax2.clear()
    ax2.set_ylim(-0.1, 1.1)
    # ax1.set_ylim(-0.1, 150.0)
    ax1.set_xlabel('#batches/10')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('%')
    print(df['score'][-250:].mean())
    if only_last_batch:
        ax1.plot(df['step'][-2725:-1], df['loss'][-2725:-1], 'o', color='black', label='loss')
        # ax2.plot(df['step'][-2725:-1], df['score'][-2725:-1], 'o', color='blue', label='accuracy [wrt current epoch]')

        # ax1.plot(df['step'][-2725:-1], df['avg_loss'][-2725:-1], 'o', color='red', label='avg loss')
        # ax2.plot(df['step'][-2725:-1], df['acc'][-2725:-1], 'o', color='blue', label='accuracy [wrt current epoch]')
        # ax2.plot(df['step'][-2725:-1], df['f1'][-2725:-1], 'o', color='green', label='F1 score [wrt current epoch]')
    else:
        ax2.plot(df['step'], df['score'], 'o', color='lightgrey', label='Coh. Acc. [curr. datapoint]')
        ax2.plot(df['step'], df['da_score'], 'o', color='lightblue', label='DA Acc. [curr. datapoint]')
        ax1.plot(df['step'], df['loss'], 'o', color='black', label='Loss [curr. datapoint]')
        
        # if len(df['step'])>100:
            # ax1.plot(df['step'][99:], np.convolve(df['loss'], np.ones((100,))/100, mode='valid'), 'o', color='red', label='Mean Loss [last 100]')
            # ax2.plot(df['step'][99:], np.convolve(df['score'], np.ones((100,))/100, mode='valid'), 'o', color='green', label='Mean Acc. [last 100]')
        ax1.plot(df['step'], df['loss'].ewm(span=50, adjust=False).mean(), 'o', color='red', label='Mean Loss [last 100]')
        ax2.plot(df['step'], df['score'].ewm(span=50, adjust=False).mean(), 'o', color='green', label='Mean Acc. [last 100]')
        ax2.plot(df['step'], df['da_score'].ewm(span=50, adjust=False).mean(), 'o', color='blue', label='Mean Acc. [last 100]')


        # ax1.plot(df['step'], df['loss'], 'o', color='black', label='loss')
        # ax1.plot(df['step'], df['avg_loss'], 'o', color='red', label='avg loss')
        ax1.set_yscale('log')
        # ax2.plot(df['step'], df['acc'], 'o', color='blue', label='accuracy [wrt current epoch]')
        # ax2.plot(df['step'], df['f1'], 'o', color='green', label='F1 score [wrt current epoch]')
    ax1.legend(loc=2)
    ax2.legend(loc=0)
    fig.tight_layout()

ani = animation.FuncAnimation(fig, animate, interval=3000)
plt.show()
