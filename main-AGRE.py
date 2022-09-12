import numpy as np

from src.AGRE import train

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=10, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=50, help='embedding size')
    # parser.add_argument('--p', type=int, default=50, help='the number of paths')
    # parser.add_argument('--path_len', type=int, default=3, help='the length of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=10, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=30, help='embedding size')
    # parser.add_argument('--p', type=int, default=10, help='the number of paths')
    # parser.add_argument('--path_len', type=int, default=3, help='the length of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=60, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--dim', type=int, default=30, help='embedding size')
    # parser.add_argument('--p', type=int, default=30, help='the number of paths')
    # parser.add_argument('--path_len', type=int, default=3, help='the length of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--dim', type=int, default=50, help='embedding size')
    parser.add_argument('--p', type=int, default=50, help='the number of paths')
    parser.add_argument('--path_len', type=int, default=3, help='the length of paths')
    parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()
    train(args, True)

'''
music	train_auc: 0.870 	 train_acc: 0.816 	 eval_auc: 0.841 	 eval_acc: 0.787 	 test_auc: 0.838 	 test_acc: 0.786 		[0.2, 0.29, 0.48, 0.5, 0.5, 0.55, 0.59, 0.6]
book	train_auc: 0.718 	 train_acc: 0.675 	 eval_auc: 0.731 	 eval_acc: 0.697 	 test_auc: 0.741 	 test_acc: 0.699 		[0.1, 0.15, 0.32, 0.35, 0.35, 0.41, 0.44, 0.44]
ml	train_auc: 0.932 	 train_acc: 0.856 	 eval_auc: 0.900 	 eval_acc: 0.821 	 test_auc: 0.901 	 test_acc: 0.819 		[0.17, 0.27, 0.53, 0.57, 0.57, 0.62, 0.64, 0.64]
yelp	train_auc: 0.895 	 train_acc: 0.815 	 eval_auc: 0.875 	 eval_acc: 0.800 	 test_auc: 0.875 	 test_acc: 0.801 		[0.16, 0.26, 0.46, 0.49, 0.49, 0.53, 0.55, 0.58]
'''
