import time
import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from src.evaluate import get_all_metrics
from src.load_base import load_data, get_records


class Generator(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        # self.l2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = t.sigmoid(self.l1(x))
        # x = t.sigmoid(self.l2(x))
        return x


class CFG(nn.Module):

    def __init__(self, args, n_entity, n_relation, n_user):

        super(CFG, self).__init__()
        self.dim = args.dim
        self.n_neighbor = args.n_neighbor
        user_embedding_mat = t.randn(n_user, args.dim)
        entity_embedding_mat = t.randn(n_entity, args.dim)
        relation_embedding_mat = t.randn(n_relation, args.dim)
        nn.init.xavier_uniform_(entity_embedding_mat)
        nn.init.xavier_uniform_(relation_embedding_mat)
        nn.init.xavier_uniform_(user_embedding_mat)
        self.entity_embedding_mat = nn.Parameter(entity_embedding_mat)
        self.user_embedding_mat = nn.Parameter(user_embedding_mat)
        self.relation_embedding_mat = nn.Parameter(relation_embedding_mat)
        self.generator = Generator(2 * self.n_neighbor * self.dim, self.dim)
        self.criterion = nn.BCELoss()

    def forward(self, pairs, label, neighbor_dict):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        user_embeddings = self.user_embedding_mat[users]

        predict = self.get_predict(users, items)
        false_predict = t.sigmoid((user_embeddings * self.get_false_item(items, neighbor_dict)).sum(dim=1))
        return self.criterion(predict, label) + self.criterion(false_predict, label)

    def get_predict(self, users, items):
        user_embeddings = self.user_embedding_mat[users]
        item_embeddings = self.entity_embedding_mat[items]

        predict = t.sigmoid((user_embeddings * item_embeddings).sum(dim=1))
        return predict

    def get_false_item(self, items, neighbor_dict):
        relation_embedding_list = []
        tail_embedding_list = []
        for item in items:
            relation_embedding_list.append(self.relation_embedding_mat[neighbor_dict[item][0]].reshape(1, -1, self.dim))
            tail_embedding_list.append(self.entity_embedding_mat[neighbor_dict[item][1]].reshape(1, -1, self.dim))

        relation_embeddings = t.cat(relation_embedding_list, dim=0)
        tail_embeddings = t.cat(tail_embedding_list, dim=0)

        # (batch_size, n_neighbor, 2 * dim)
        t_r = t.cat([relation_embeddings, tail_embeddings], dim=-1).reshape(-1, 2 * self.n_neighbor * self.dim)

        return self.generator(t_r)


def get_scores(model, rec, batch_size):
    scores = {}
    model.eval()
    for user in rec:
        items = list(rec[user])
        users = [user] * len(items)
        predict = []
        for i in range(0, len(items), batch_size):
            predict.extend(model.get_predict(users[i: i+batch_size], items[i: i+batch_size]).cpu().detach().view(-1).numpy().tolist())
        # predict = self.forward(pairs, ripple_sets).cpu().detach().view(-1).numpy().tolist()
        n = len(items)
        user_scores = {items[i]: predict[i] for i in range(n)}
        user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
        scores[user] = user_list
    model.train()
    return scores


def eval_ctr(model, pairs, batch_size):

    model.eval()
    pred_label = []
    users = [pair[0] for pair in pairs]
    items = [pair[1] for pair in pairs]
    for i in range(0, len(pairs), batch_size):
        batch_label = model.get_predict(users[i: i+batch_size], items[i: i+batch_size]).cpu().detach().numpy().tolist()
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return round(auc, 3), round(acc, 3)


def get_neighbors(kg_dict, items, n_neighbor):

    neighbor_dict = dict()

    for item in items:
        relations_and_tails = kg_dict[item]
        size = len(relations_and_tails)
        if size >= n_neighbor:
            indices = np.random.choice(size, n_neighbor, replace=False)
        else:
            indices = np.random.choice(size, n_neighbor, replace=True)
        neighbor_dict[item] = [[relations_and_tails[i][0] for i in indices], [relations_and_tails[i][1] for i in indices]]

    return neighbor_dict


def train(args, is_topk=False):
    np.random.seed(555)
    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, rec, kg_dict = data[4], data[5], data[6], data[7], data[8]
    test_records = get_records(test_set)
    neighbor_dict = get_neighbors(kg_dict, range(n_item), args.n_neighbor)
    model = CFG(args, n_entity, n_relation, n_user)
    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.BCELoss()

    print(args.dataset + '-----------------------------------------')
    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    all_precision_list = []
    for epoch in (range(args.epochs)):

        start = time.clock()

        loss_sum = 0

        np.random.shuffle(train_set)
        for i in range(0, len(train_set), args.batch_size):

            batch_uvls = train_set[i: i + args.batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]
            labels = t.tensor([int(uvl[2]) for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)

            loss = model(pairs, labels, neighbor_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        train_auc, train_acc = eval_ctr(model, train_set, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, args.batch_size)

        print('epoch: %d \t train_auc: %.3f \t train_acc: %.3f \t '
              'eval_auc: %.3f \t eval_acc: %.3f \t test_auc: %.3f \t test_acc: %.3f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        precision_list = []
        if is_topk:
            scores = get_scores(model, rec, args.batch_size)
            precision_list = get_all_metrics(scores, test_records)[0]
            print(precision_list, end='\t')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        all_precision_list.append(precision_list)
        end = time.clock()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.3f \t train_acc: %.3f \t eval_auc: %.3f \t eval_acc: %.3f \t '
          'test_auc: %.3f \t test_acc: %.3f \t' %
          (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
           test_auc_list[indices], test_acc_list[indices]), end='\t')

    print(all_precision_list[indices])

    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]