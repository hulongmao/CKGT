'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
from utility.load_data import Data
from time import time
import scipy.sparse as sp
import random as rd
import collections
import torch

class KGAT_loader(Data):
    def __init__(self, args, path):
        super().__init__(args, path)

        # load user_kg
        self.user_sex, self.sex_rel, self.user_school, self.school_rel = self._load_user_kg()
        print('self.user_school shape:', self.user_school.shape)
        print('self.school_rel:', self.school_rel)
        print('self.user_school first 10:\n', self.user_school[:10])
        print('self.user_sex shape:', self.user_sex.shape)
        print('self.sex_rel:', self.sex_rel, '\n')

        # generate the sparse adjacency matrices for user-item interaction & relational kg data.
        # adj_list: adjacency matrix list. list length: (1+39)*2 = 80, here 1 represents user-item interaction,
        #           39 is relation number. Each matrix (coo_matrix) size：(n_users+n_entities) * (n_users+n_entities),
        #           i.e. [u-i, i-u, h-t in rel1, t-h in rel1, ... , h-t in rel39, t-h in rel39]
        # adj_r_list: relation list, each relation corresponds to a matrix in adj_list.
        #             i.e.[0, 40, 1, 41, 2, 42, ... , 39, 79]
        self.adj_list, self.adj_r_list = self._get_relational_adj_list()

        # generate the sparse laplacian matrices (coo_matrix) in each relation. A = D^-1A
        # D^-1 is a diagonal matrix, where the values of the diagonal elements are the inverse of degrees
        # The effect of D^-1A is to perform normalization, D^-1AX is the average of neighbors vectors.
        # reference: 图卷积网络(Graph Convolutional Networks, GCN)详细介绍
        self.lap_list = self._get_relational_lap_list()  # length: 80

        # generate the triples dictionary, key is 'head', value is '(tail, relation)'.
        self.all_kg_dict = self._get_all_kg_dict()
        triples = sum([len(v) for h, v in self.all_kg_dict.items()])
        print('triples by self.all_kg_dict:', triples)

        # all list is ordered by h and t, each list len is (652514 + 2557746) * 2 = 6420520
        self.all_h_list, self.all_r_list, self.all_t_list, self.all_v_list = self._get_all_kg_data()

    def _load_user_kg(self):
        sex_rel = 9
        user_sex = []
        school_rel = 10
        user_school = []
        with open(self.kg_user_file, encoding='utf-8') as f:
            for line in f:
                h = int(line.strip().split(' ')[0])
                r = int(line.strip().split(' ')[1])
                t = int(line.strip().split(' ')[2])
                if r == sex_rel:
                    user_sex.append([h, t])
                elif r == school_rel:
                    user_school.append([h, t])
        return np.array(user_sex), sex_rel, np.array(user_school), school_rel


    def _get_relational_adj_list(self):
        print('original n_relations:', self.n_relations)
        t1 = time()
        adj_mat_list = []
        adj_r_list = []

        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_users + self.n_entities
            # single-direction
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        R, R_inv = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        adj_mat_list.append(R)
        adj_r_list.append(0)  # 0

        adj_mat_list.append(R_inv)
        adj_r_list.append(self.n_relations + 1)  # 13
        print('\tconvert ratings into adj mat done.')

        Uksex, Uksex_inv = _np_mat2sp_adj(self.user_sex, row_pre=0, col_pre=self.n_users)
        adj_mat_list.append(Uksex)
        adj_r_list.append(1)  # 1
        adj_mat_list.append(Uksex_inv)
        adj_r_list.append(self.n_relations + 2)  # 14
        print('\tconvert user kg into adj mat done.')

        Ukschool, Ukschool_inv = _np_mat2sp_adj(self.user_school, row_pre=0, col_pre=self.n_users)
        adj_mat_list.append(Ukschool)
        adj_r_list.append(2)  # 2
        adj_mat_list.append(Ukschool_inv)
        adj_r_list.append(self.n_relations + 3)  # 15

        for r_id in self.relation_dict.keys(): # 0~8
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]), row_pre=self.n_users, col_pre=self.n_users)
            adj_mat_list.append(K)
            adj_r_list.append(r_id + 3)  # 3~11

            adj_mat_list.append(K_inv)
            adj_r_list.append(r_id + 4 + self.n_relations)  # 16~24
        print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        self.n_relations = len(adj_r_list)
        print('modified n_relations:', adj_r_list)

        return adj_mat_list, adj_r_list

    def _get_relational_lap_list(self):
        def _bi_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            # Number of triplets corresponding to the head entity at in a specific relationship. Note relation num is 80.
            rowsum = np.array(adj.sum(1)) # axis=1, shape: (184166, 1)
            # print('rowsum[70679] in relation 2 :', rowsum[70678:70680]) # [[0.], [6.]]

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.  # set infinite value to 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)  # d_mat_inv: dia_matrix, adj: coo_matrix
            return norm_adj.tocoo()

        if self.args.adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate bi-normalized adjacency matrix.')
        else:
            lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate si-normalized adjacency matrix.')
        return lap_list

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        for l_id, lap in enumerate(self.lap_list):

            rows = lap.row
            cols = lap.col

            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = self.adj_r_list[l_id]

                all_kg_dict[head].append((tail, relation))
        return all_kg_dict

    def _get_all_kg_data(self):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list, all_t_list, all_r_list = [], [], []
        all_v_list = []

        for l_id, lap in enumerate(self.lap_list):
            all_h_list += list(lap.row)  # the row where non-zero values are located, the same below
            all_t_list += list(lap.col)
            all_v_list += list(lap.data)
            all_r_list += [self.adj_r_list[l_id]] * len(lap.row)

        assert len(all_h_list) == sum([len(lap.data) for lap in self.lap_list])

        # resort the all_h/t/r/v_list,
        # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
        print('\treordering indices...')
        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[],[],[]]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])
        print('\treorganize all kg data done.')

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])


        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)
        # try:
        #     assert sum(new_v_list) == sum(all_v_list)
        # except Exception:
        #     print(sum(new_v_list), '\n')
        #     print(sum(all_v_list), '\n')
        print('\tsort all data done.')


        return new_h_list, new_r_list, new_t_list, new_v_list

    def _generate_train_A_batch(self):
        exist_heads = self.all_kg_dict.keys()

        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            pos_triples = self.all_kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]

                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]

                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break

                t = np.random.randint(low=0, high=self.n_users + self.n_entities, size=1)[0]
                if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts

        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts

            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
            neg_t_batch += neg_ts

        heads = torch.LongTensor(heads)
        pos_r_batch = torch.LongTensor(pos_r_batch)
        pos_t_batch = torch.LongTensor(pos_t_batch)
        neg_t_batch = torch.LongTensor(neg_t_batch)
        return heads, pos_r_batch, pos_t_batch, neg_t_batch

    def generate_train_feed_dict(self, model, batch_data):
        feed_dict = {
            model.users: batch_data['users'],
            model.pos_items: batch_data['pos_items'],
            model.neg_items: batch_data['neg_items'],

            model.mess_dropout: eval(self.args.mess_dropout),
            model.node_dropout: eval(self.args.node_dropout),
        }

        return feed_dict

    def generate_train_A_batch(self):
        heads, relations, pos_tails, neg_tails = self._generate_train_A_batch()

        batch_data = {}

        batch_data['heads'] = heads
        batch_data['relations'] = relations
        batch_data['pos_tails'] = pos_tails
        batch_data['neg_tails'] = neg_tails
        return batch_data

    def generate_train_A_feed_dict(self, model, batch_data):
        feed_dict = {
            model.h: batch_data['heads'],
            model.r: batch_data['relations'],
            model.pos_t: batch_data['pos_tails'],
            model.neg_t: batch_data['neg_tails'],

        }

        return feed_dict

if __name__ == '__main__':
    import parsers
    args = parsers.parse_args()
    # data = Data(args=args, path='../../../Data30')
    kgat_data = KGAT_loader(args=args, path='../../../Data30')
    # print('\ntrain users:', len(data.train_user_dict.keys()))
    # print('test  users:', len(data.test_user_dict.keys()))
