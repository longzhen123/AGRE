import time
import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import optim

from src.evaluate import get_all_metrics
from src.load_base import load_data, get_records


class DNN(nn.Module):

    def __init__(self, dim):
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        super(DNN, self).__init__()
        self.l1 = nn.Linear(2*dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.l3 = nn.Linear(dim, 1)

    def forward(self, x):
        y = self.l1(x)
        y = t.relu(y)
        y = self.l2(y)
        y = t.relu(y)
        y = self.l3(y)
        y = t.sigmoid(y)

        return y


class CKAN(nn.Module):

    def __init__(self, n_entities, dim, n_relations, L, agg):
        super(CKAN, self).__init__()

        entity_embedding_matrix = t.randn(n_entities, dim)
        nn.init.xavier_uniform_(entity_embedding_matrix)
        rel_embedding_matrix = t.randn(n_relations, dim)
        nn.init.xavier_uniform_(rel_embedding_matrix)

        self.entity_embedding_matrix = nn.Parameter(entity_embedding_matrix)
        self.dim = dim
        self.rel_embedding_matrix = nn.Parameter(rel_embedding_matrix)

        self.L = L
        self.dnn = DNN(dim)
        self.agg = agg
        self.criterion = nn.BCELoss()
        if agg == 'concat':
            self.Wagg = nn.Linear((L+1)*dim, dim)
        else:
            self.Wagg = nn.Linear(dim, dim)

    def forward(self, pairs, user_ripple_sets, item_ripple_sets):
        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]

        u_heads_list, u_relations_list, u_tails_list, u_entity_list = self.get_head_relation_and_tail(users, user_ripple_sets)
        i_heads_list, i_relations_list, i_tails_list, i_entity_list = self.get_head_relation_and_tail(items, item_ripple_sets)

        # print(len(heads_list[1]), len(u_heads_list[1]))
        user_embeddings = self.get_item_embedding(u_heads_list, u_relations_list, u_tails_list, u_entity_list, len(users))
        item_embeddings = self.get_item_embedding(i_heads_list, i_relations_list, i_tails_list, i_entity_list, len(items))
        # user_embeddings = self.get_item_embedding(u_heads_list, u_relations_list, u_tails_list, u_entity_list,
        #                                                    len(users))
        # item_embeddings = self.get_item_embedding(i_heads_list, i_relations_list, i_tails_list, i_entity_list,
        #                                           len(items))
        predicts = t.sigmoid((user_embeddings * item_embeddings).sum(dim=1))

        return predicts

    def get_head_relation_and_tail(self, os, ripple_sets):

        heads_list = []
        relations_list = []
        tails_list = []
        entity_list = []
        for l in range(self.L+1):
            l_head_list = []
            l_relation_list = []
            l_tail_list = []

            for o in os:
                if l == 0:
                    entity_list.extend(ripple_sets[o][0])
                else:
                    l_head_list.extend(ripple_sets[o][l][0])
                    l_relation_list.extend(ripple_sets[o][l][1])
                    l_tail_list.extend(ripple_sets[o][l][2])

            heads_list.append(l_head_list)
            relations_list.append(l_relation_list)
            tails_list.append(l_tail_list)
            # print(len(l_head_list))
        return heads_list, relations_list, tails_list, entity_list

    def get_item_embedding(self, heads_list, relations_list, tails_list, entity_list, n):
        e_list = []

        e_list.append(self.entity_embedding_matrix[entity_list].reshape(n, -1, self.dim).mean(dim=1))

        for l in range(1, self.L+1):
            head_embeddings = self.entity_embedding_matrix[heads_list[l]]
            relation_embeddings = self.rel_embedding_matrix[relations_list[l]]
            tail_embeddings = self.entity_embedding_matrix[tails_list[l]]
            # print(head_embeddings.shape, tail_embeddings.shape)
            pi = self.dnn(t.cat([head_embeddings, relation_embeddings], dim=1))
            # print(len(pi))
            pi = t.softmax(pi.reshape(n, -1, 1), dim=1)
            a = (pi * tail_embeddings.reshape(n, -1, self.dim)).sum(dim=1)
            e_list.append(a)

        return self.aggregator(e_list)

    def aggregator(self, e_list):
        # print(len(e_list))
        embedding = t.cat(e_list, dim=1)
        # print(embedding.shape, self.Wagg.weight.shape)
        if self.agg == 'concat':
            return t.sigmoid(self.Wagg(embedding))
        elif self.agg == 'sum':
            return t.sigmoid(self.Wagg(embedding.sum(dim=0).view(1, self.dim)))
        else:
            return t.sigmoid(self.Wagg(embedding.max(dim=0)[0].view(1, self.dim)))


def get_scores(model, rec, user_ripple_sets, item_ripple_sets):
    scores = {}
    model.eval()
    for user in (rec):
        items = list(rec[user])
        pairs = [[user, item] for item in items]
        predict = model.forward(pairs, user_ripple_sets, item_ripple_sets).cpu().reshape(-1).detach().numpy().tolist()
        # print(predict)
        n = len(pairs)
        user_scores = {items[i]: predict[i] for i in range(n)}
        user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
        scores[user] = user_list
    model.train()
    # print('=========================')
    return scores


def eval_ctr(model, pairs, user_ripple_sets, item_ripple_sets, batch_size):

    model.eval()
    pred_label = []
    for i in range(0, len(pairs), batch_size):
        batch_label = model(pairs[i: i+batch_size], user_ripple_sets, item_ripple_sets).cpu().detach().numpy().tolist()
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np  = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return round(auc, 3), round(acc, 3)


def get_ripple_set(train_dict, kg_dict, H, size):

    ripple_set_dict = {user: [] for user in train_dict}

    for u in (train_dict):

        next_e_list = train_dict[u]
        replace = len(train_dict[u]) < size
        indices = np.random.choice(len(train_dict[u]), size, replace=replace)
        ripple_set_dict[u].append([train_dict[u][i] for i in indices])
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
                h_head_list = ripple_set_dict[u][-1][0]
                h_relation_list = ripple_set_dict[u][-1][1]
                h_tail_list = ripple_set_dict[u][-1][0]
            else:
                replace = len(h_head_list) < size
                indices = np.random.choice(len(h_head_list), size, replace=replace)
                h_head_list = [h_head_list[i] for i in indices]
                h_relation_list = [h_relation_list[i] for i in indices]
                h_tail_list = [h_tail_list[i] for i in indices]

            ripple_set_dict[u].append((h_head_list, h_relation_list, h_tail_list))

            next_e_list = ripple_set_dict[u][-1][2]

    return ripple_set_dict


def get_item_record(item_set, train_records):

    ripple_sets = dict()

    for item in item_set:
        item_ripple_set = {item}
        for items in train_records.values():
            if item in items:
                item_ripple_set.update(items)
        ripple_sets[item] = list(item_ripple_set)

    return ripple_sets


def train(args, is_topk=False):
    np.random.seed(555)

    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, rec, kg_dict = data[4], data[5], data[6], data[7], data[8]
    train_records = get_records(train_set)
    test_records = get_records(test_set)
    item_records = get_item_record(list(range(n_item)), train_records)
    user_ripple_sets = get_ripple_set(train_records, kg_dict, args.L, args.K_l)
    item_ripple_sets = get_ripple_set(item_records, kg_dict, args.L, args.K_l)
    model = CKAN(n_entities=n_entity, dim=args.dim, n_relations=n_relation, L=args.L, agg=args.agg)

    if t.cuda.is_available():
        model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.BCELoss(reduction='sum')

    print(args.dataset + '-----------------------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('L: %d' % args.L, end='\t')
    print('K_l: %d' % args.K_l, end='\t')
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
        model.train()
        loss_sum = 0
        for i in range(0, len(train_set), args.batch_size):

            if (i + args.batch_size + 1) > len(train_set):
                batch_uvls = train_set[i:]
            else:
                batch_uvls = train_set[i: i + args.batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]
            labels = t.tensor([int(uvl[2]) for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)

            predicts = model(pairs, user_ripple_sets, item_ripple_sets)

            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
        train_auc, train_acc = eval_ctr(model, train_set, user_ripple_sets, item_ripple_sets, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, user_ripple_sets, item_ripple_sets, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, user_ripple_sets, item_ripple_sets, args.batch_size)

        print('epoch: %d \t train_auc: %.3f \t train_acc: %.3f \t '
              'eval_auc: %.3f \t eval_acc: %.3f \t test_auc: %.3f \t test_acc: %.3f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        precision_list = []
        if is_topk:
            scores = get_scores(model, rec, user_ripple_sets, item_ripple_sets)
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

