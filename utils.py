import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import csv
import seaborn as sns
import pandas


def toCSV(data, row_name, fname):
    """convert list data to csv file"""
    ep = np.arange(len(data))
    _data = [tuple(ep), tuple(data)]
    with open(fname, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(row_name)
        for r in list(zip(*_data)):
            writer.writerow(r) 


def group_data(data, _k = 5):
    return np.floor(data)

def make_graph(fnames, labels, title=None, _k = 5):
    # sns.set_theme()
    for fname, label in zip(*(fnames, labels)):
        df = pandas.read_csv(fname)
        x, y = df.columns.values
        df[x] = df[x].apply(lambda i: int(np.floor(i/_k)*_k))
        g = sns.lineplot(data=df, x=x, y=y, label=label)
        # g.set_yscale('log')
    plt.legend()
    if title:
        plt.title(title)
    # plt.show()
    plt.savefig('test')


make_graph(['csv/100actions_sac_1000.csv', 'csv/100actions_sql.csv', 'csv/100actions_ql.csv'], ['Soft QAC', 'Soft Q Learning', 'Q Learning'], 'test')
"""
test examples:
r = ['Ep', 'rewards']
d = list(np.random.randn(1, 1000)[0])
toCSV(d, r, 'test.csv')
fnames=['test.csv']
make_graph(fnames)
"""
