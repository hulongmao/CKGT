'''
Create on Jun 18, 2024
Pytorch Implementation of CKGC model
@author: Longmao Hu
'''

import collections
import numpy as np
import random as rd
import os
import torch
import scipy.io as sio
from time import time

class Data(object):
    def __init__(self, args, path):
        begin = time()
        self.path = path
        self.args = args

        self.batch_size = args.batch_size

        train_file = path + '/train30.txt'
        test_file = path + '/test30.txt'

        kg_file = path + '/kg_refine_30.txt'
        self.kg_user_file= path + '/kg_user30.txt'

        # ----------get number of users and items & then load rating data from train_file & test_file------------.
        self.n_train, self.n_test = 0, 0
        self.n_users, self.n_items = 0, 0

        self.train_data, self.train_user_dict = self._load_ratings(train_file)
        # train_data: user-item interacted array in train file, i.e. [[0,0], [0,1],...,[70678, 6576],[70678, 15786]]
        # train_user_dict: {user1: item list interacted by the user1, ...}
        self.test_data, self.test_user_dict = self._load_ratings(test_file)
        self.exist_users = self.train_user_dict.keys()

        self._statistic_ratings()

        # ----------get number of entities and relations & then load kg data from kg_file ------------.
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file, self.kg_user_file)
        # kg_data: ordered knowledge graph array, i.e. [[0 0 24916], [0 0 24917], ..., [113486 14  106821]]
        # kg_dict: dictionary of knowledge graph, {head1: [(tail1, relation1),..., (tailm, relationm)], ..., headn:[...]}
        # relation_dict: dictionary of knowledge graph, {rela1:[(head1, tail1),..., (headm, tailm)], ..., relas:[...]}

        # ----------print the basic info about the dataset-------------.
        # self.batch_size_kg = self.n_triples // (self.n_train // self.batch_size )
        self.batch_size_kg = int(args.batch_size_kg / 1.5) # for HTN and KGAT

        self._print_data_info()

        print('Data load consumed:', time() - begin, 's')

    # reading train & test interaction data.
    def _load_ratings(self, file_name):
        user_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids
        return np.array(inter_mat), user_dict

    def _statistic_ratings(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

    # reading train & test interaction data.
    def _load_kg(self, file_name, user_file):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0) # 去除重复行，并按照行的大小排序输出

        kg_user = np.loadtxt(user_file, dtype=np.int32)

        self.n_relations = max(max(kg_np[:, 1]), max(kg_user[:, 1])) + 1  # 11

        # 同时考虑kg_refine_30.txt以及kg_user30.txt,  self.n_entities = 8938
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2]), max(kg_user[:, 2])) + 1
        self.n_triples = len(kg_np) + len(kg_user)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def get_test(self, user_num, item_num=1590):
        train_data_dic = {}
        with open(self.path + '/train.txt', encoding='utf-8') as f:
            for line in f:
                u = line.strip().split(' ')[0]
                items = line.strip().split(' ')[1:]
                items = [int(item) for item in items]
                train_data_dic[int(u)] = items
        test_data_dic = {}
        with open(self.path + '/test.txt', encoding='utf-8') as f:
            for line in f:
                u = line.strip().split(' ')[0]
                items = line.strip().split(' ')[1:]
                items = [int(item) for item in items]
                test_data_dic[int(u)] = items

        pos_tests_list, test_items_list = [], []
        for u in range(user_num):
            pos_tests = test_data_dic[u]
            # 从全部item中去掉train_items, 得到需要进行测试的item
            test_items = [item for item in range(item_num) if item not in train_data_dic[u]]
            pos_tests_list.append(pos_tests)
            test_items_list.append(test_items)

        return pos_tests_list, test_items_list

    def _print_data_info(self):
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))
        print('[batch_size, batch_size_kg]=[%d, %d]' % (self.batch_size, self.batch_size_kg))

    def generate_train_cf_batch(self, items_per_user=1):
        '''
        generate a batch Collaborative filtering data
        returns:
        users: user list
        pos_items: pos item list, one item per user
        neg_items: neg item list, one item per user
        '''
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items
        
        # num = 4  # 1 default by kgat
        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, items_per_user)  
            neg_items += sample_neg_items_for_u(u, items_per_user)  
        
        if items_per_user > 1:
            user_list = []
            for u in users:
                user_list += [u] * items_per_user
            users = user_list
        users = torch.LongTensor(users)
        pos_items = torch.LongTensor(pos_items)
        neg_items = torch.LongTensor(neg_items)
        
        # print('users:', users.shape, 'pos_items:', pos_items.shape, 'neg_items:', neg_items.shape)
        return users, pos_items, neg_items

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            test_iids = self.test_user_dict[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state

if __name__ == '__main__':
    import parsers
    args = parsers.parse_args()
    data = Data(args=args, path='../../../Data30')
    print('\ntrain users:', len(data.train_user_dict.keys()))
    print('test  users:', len(data.test_user_dict.keys()))