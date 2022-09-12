import time
import numpy as np
import torch.nn as nn
import torch as t
from torch import optim
from sklearn.metrics import roc_auc_score, accuracy_score

from src.evaluate import eval_topk, get_all_metrics
from src.load_base import load_data, get_records


class RNN(nn.Module):

    def __init__(self, dim):
        super(RNN, self).__init__()
        self.dim = dim
        self.W = nn.Linear(dim, dim, bias=False)
        self.H = nn.Linear(dim, dim, bias=False)
        self.U = nn.Linear(dim, dim, bias=False)

    def forward(self, record_embeddings, user_embeddings):

        h = t.zeros(1, self.dim)
        if t.cuda.is_available():
            h = h.to(user_embeddings.device)

        for i_record_embeddings in record_embeddings:

            h = self.W(i_record_embeddings) + self.H(h) + self.U(user_embeddings)
            h = t.sigmoid(h)

        return h


class UMIR(nn.Module):

    def __init__(self, dim, n_entity, H, n_relation):
        super(UMIR, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.dim = dim
        self.H = H
        entity_embedding_matrix = t.randn(n_entity, dim)
        relation_embedding_matrix = t.randn(n_relation, dim)
        nn.init.xavier_uniform_(entity_embedding_matrix)
        nn.init.xavier_uniform_(relation_embedding_matrix)
        self.entity_embedding_matrix = nn.Parameter(entity_embedding_matrix)
        self.relation_embedding_matrix = nn.Parameter(relation_embedding_matrix)
        self.W = nn.Linear(3 * dim, 1)

    def forward(self, pairs, ripple_sets, user_records):
        return self.get_predict(pairs, ripple_sets, user_records)

    def get_predict(self, pairs, ripple_sets, user_records):
        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        # item_set = (set(items))
        item_set = self.get_user_records(set(users), user_records)
        item_list = list(item_set)
        heads_list, relations_list, tails_list = self.get_head_relation_and_tail(item_list, ripple_sets)
        item_embeddings = self.get_item_embedding(item_list, heads_list, relations_list, tails_list)
        n_item = len(item_list)
        item_embedding_dict = {item_list[i]: item_embeddings[i].reshape(1, self.dim) for i in range(n_item)}
        pair_item_embeddings = self.entity_embedding_matrix[items]
        record_embeddings = self.get_record_embedding(users, user_records, item_embedding_dict)
        pair_user_embeddings = record_embeddings.sum(dim=0)

        predicts = t.sigmoid((pair_user_embeddings * pair_item_embeddings).sum(dim=1))

        return predicts

    def get_record_embedding(self, users, user_records, item_embedding_dict):
        item_embedding_list = []

        for user in users:
            records = user_records[user]
            record_embeddings = t.cat([item_embedding_dict[record] for record in records], dim=0)
            item_embedding_list.append(record_embeddings.reshape(-1, 1, self.dim))

        return t.cat(item_embedding_list, dim=1)

    def get_user_records(self, users, user_records):

        item_set = set()

        for user in users:
            item_set.update(user_records[user])

        return item_set

    def get_head_relation_and_tail(self, item_list, ripple_sets):

        heads_list = []
        relations_list = []
        tails_list = []
        for h in range(self.H):
            l_head_list = []
            l_relation_list = []
            l_tail_list = []

            for item in item_list:

                l_head_list.extend(ripple_sets[item][h][0])
                l_relation_list.extend(ripple_sets[item][h][1])
                l_tail_list.extend(ripple_sets[item][h][2])

            heads_list.append(l_head_list)
            relations_list.append(l_relation_list)
            tails_list.append(l_tail_list)

        return heads_list, relations_list, tails_list

    def get_item_embedding(self, item_list, heads_list, relations_list, tails_list):

        i_list = []
        item_embeddings = self.entity_embedding_matrix[item_list].view(-1, self.dim)
        i_list.append(item_embeddings)
        n_item = len(item_list)
        for h in range(self.H):
            head_embeddings = self.entity_embedding_matrix[heads_list[h]].reshape(n_item, -1, self.dim)
            relation_embeddings = self.relation_embedding_matrix[relations_list[h]].reshape(n_item, -1, self.dim)
            tail_embeddings = self.entity_embedding_matrix[tails_list[h]].reshape(n_item, -1, self.dim)

            hrt = t.cat([head_embeddings, relation_embeddings, tail_embeddings], dim=-1)
            pi = t.sigmoid(self.W(hrt))
            pi = t.softmax(pi, dim=1)
            i_embeddings = (pi * tail_embeddings).sum(dim=1)
            i_list.append(i_embeddings)

        return sum(i_list)


def get_scores(model, rec, ripple_sets, user_records, batch_size):
    scores = {}
    model.eval()
    for user in rec:

        items = list(rec[user])
        pairs = [[user, item] for item in items]
        predict = []
        for i in range(0, len(pairs), batch_size):
            predict.extend(model.forward(pairs[i: i+batch_size], ripple_sets, user_records).cpu().reshape(-1).detach().numpy().tolist())
        # print(predict)
        n = len(pairs)
        user_scores = {items[i]: predict[i] for i in range(n)}
        user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
        scores[user] = user_list
    model.train()
    # print('=========================')
    return scores


def eval_ctr(model, pairs, ripple_sets, user_records, batch_size):

    model.eval()
    pred_label = []
    for i in range(0, len(pairs), batch_size):
        batch_label = model(pairs[i: i+batch_size], ripple_sets, user_records).cpu().detach().numpy().tolist()
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


def get_user_records(train_records, K_u):

    user_records = dict()

    for user in train_records:
        records = train_records[user]

        if len(records) > K_u:
            indices = np.random.choice(len(records), K_u, replace=False)
        else:
            indices = np.random.choice(len(records), K_u, replace=True)

        user_records[user] = [records[i] for i in indices]

    return user_records


def get_ripple_set(items, kg_dict, H, size):

    ripple_set_dict = {item: [] for item in items}

    for item in (items):

        next_e_list = [item]

        for h in range(H):
            h_head_list = []
            h_relation_list = []
            h_tail_list = []
            for head in next_e_list:
                if head not in kg_dict:
                    continue
                for rt in kg_dict[head]:
                    relation = rt[0]
                    tail = rt[1]
                    h_head_list.append(head)
                    h_relation_list.append(relation)
                    h_tail_list.append(tail)

            if len(h_head_list) == 0:
                h_head_list = ripple_set_dict[item][-1][0]
                h_relation_list = ripple_set_dict[item][-1][1]
                h_tail_list = ripple_set_dict[item][-1][0]
            else:
                replace = len(h_head_list) < size
                indices = np.random.choice(len(h_head_list), size, replace=replace)
                h_head_list = [h_head_list[i] for i in indices]
                h_relation_list = [h_relation_list[i] for i in indices]
                h_tail_list = [h_tail_list[i] for i in indices]

            ripple_set_dict[item].append((h_head_list, h_relation_list, h_tail_list))

            next_e_list = ripple_set_dict[item][-1][2]

    return ripple_set_dict


def train(args, is_topk=False):
    np.random.seed(555)

    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, rec, kg_dict = data[4], data[5], data[6], data[7], data[8]
    train_records = get_records(train_set)
    test_records = get_records(test_set)
    ripple_sets = get_ripple_set(range(n_item), kg_dict, args.H, args.K_u)

    user_records = get_user_records(train_records, args.K_u)
    model = UMIR(args.dim, n_entity, args.H, n_relation)
    criterion = nn.BCELoss()

    if t.cuda.is_available():
        model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(args.dataset + '-----------------------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('H: %d' % args.H, end='\t')
    print('K_u: %d' % args.K_u, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
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
            if (i + args.batch_size + 1) > len(train_set):
                batch_uvls = train_set[i:]
            else:
                batch_uvls = train_set[i: i + args.batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]
            labels = t.tensor([int(uvl[2]) for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)
            predicts = model(pairs, ripple_sets, user_records)
            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

        train_auc, train_acc = eval_ctr(model, train_set, ripple_sets, user_records, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, ripple_sets, user_records, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, ripple_sets, user_records, args.batch_size)

        print('epoch: %d \t train_auc: %.3f \t train_acc: %.3f \t '
              'eval_auc: %.3f \t eval_acc: %.3f \t test_auc: %.3f \t test_acc: %.3f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        precision_list = []
        if is_topk:
            scores = get_scores(model, rec, ripple_sets, user_records, args.batch_size)
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
