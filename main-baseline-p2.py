import numpy as np

from src.AGRE import train

import argparse


if __name__ == '__main__':
    auc_list = []
    test_auc_list = []
    for dim in [4, 8, 16, 32, 64]:
    # for L in [1, 2, 3]:
    # for T in [1, 2, 3, 4, 5]:
    # for ratio in [0.2, 0.4, 0.6, 0.8]:
    # for n_path in [1, 5, 10, 20, 30]:
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', type=str, default='book', help='dataset')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=10, help='epochs')
        parser.add_argument('--device', type=str, default='cuda:0', help='device')
        parser.add_argument('--dim', type=int, default=dim, help='embedding size')
        parser.add_argument('--p', type=int, default=64, help='the number of paths')
        parser.add_argument('--path_len', type=int, default=3, help='the length of paths')
        parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

        args = parser.parse_args()
        indicators = train(args)
        auc = indicators[0]
        test_auc_list.append(indicators[2])

        auc_list.append(auc)
    auc_np = np.array(auc_list)
    auc_np = auc_np.round(3)
    print(args.dataset, auc_np.tolist(), test_auc_list)

'''

'''