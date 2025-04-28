'''
hulongmao, 2024.6.2
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class KGAT(nn.Module):
    def __init__(self, data_config, pretrain_data, args):
        super(KGAT, self).__init__()
        self._parse_args(data_config, pretrain_data, args)

        self.user_entity_embed = nn.Embedding(self.n_users + self.n_entities, self.emb_dim) # (184166, 64)
        self.relation_embed = nn.Embedding(self.n_relations, self.kge_dim)  # (80, 64)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.kge_dim, self.emb_dim)) # (80, 64, 64)

        if self.pretrain_data is not None:
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - self.n_items, self.emb_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            user_entity_embed = torch.cat([torch.Tensor(self.pretrain_data['user_embed']),
                                           torch.Tensor(self.pretrain_data['item_embed']),
                                           other_entity_embed], dim=0)
            print('user_entity_embed while user & item pretrained:', user_entity_embed.shape)
            self.user_entity_embed.weight = nn.Parameter(user_entity_embed)
        else:
            nn.init.xavier_uniform_(self.user_entity_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        self.weight_size_list = [self.emb_dim] + self.layer_size   # [64, 64] or [64, 64, 32]
        self.aggregator_layers1 = nn.ModuleList()
        self.aggregator_layers2 = nn.ModuleList()
        for k in range(self.n_layers):
            linear1 = nn.Linear(self.weight_size_list[k], self.weight_size_list[k+1])            
            nn.init.xavier_uniform_(linear1.weight)           
            self.aggregator_layers1.append(linear1)
            linear2 = nn.Linear(self.weight_size_list[k], self.weight_size_list[k+1])            
            nn.init.xavier_uniform_(linear2.weight)           
            self.aggregator_layers2.append(linear2)

    def _parse_args(self, data_config, pretrain_data, args):
        self.model_type = 'kgat'
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.n_fold = 30 # orignal:100, 30 for GPU 3060(12G), 15 for GPU A10(24G)

        # initialize the attentive matrix A for phase I.
        self.A_in = data_config['A_in']

        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']

        self.adj_uni_type = args.adj_uni_type  #Specify a loss type (uni, sum), default sum

        self.lr = args.lr

        # settings for CF part.
        self.emb_dim = args.embed_size  # CF Embedding size, default 64
        self.batch_size = args.batch_size  # CF batch size, also is data_loader batch_size.

        # settings for KG part.
        self.kge_dim = args.kge_size  # KG Embedding size, default 64
        self.batch_size_kg = args.batch_size_kg  # KG batch size, seems not to used.

        self.layer_size = eval(args.layer_size) # Output sizes of every layer, default [64]
        self.n_layers = len(self.layer_size)

        self.alg_type = args.alg_type # Specify the method of aggreation of eh and eNh, default bi
        self.model_type += '_%s_%s_%s_l%d' %(args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)

        self.mess_dropout = eval(args.mess_dropout)
        self.regs = eval(args.regs) # default [1e-5,1e-5,1e-2]
        self.verbose = args.verbose

    def _create_ego_side_bi_embed(self):
        A = self.A_in
        A_fold_hat = self._split_A_hat(A)

        ego_embeddings = self.user_entity_embed.weight  # (184166, 64)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = torch.concat(temp_embed, dim=0) # eNh: (184166, 64)

            add_embeddings = ego_embeddings + side_embeddings  # eh+eNh            
            sum_embeddings = F.leaky_relu(self.aggregator_layers1[k](add_embeddings))

            bi_embedings = torch.multiply(ego_embeddings, side_embeddings)            
            bi_embedings = F.leaky_relu(self.aggregator_layers2[k](bi_embedings))

            ego_embeddings = bi_embedings + sum_embeddings
            ego_embeddings = F.dropout(ego_embeddings, self.mess_dropout[0])

            norm_embeddings = F.normalize(ego_embeddings, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.concat(all_embeddings, dim=1)  # eh0||eh1: (184166, 128)

        ua_embeddings, ea_embeddings = torch.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _split_A_hat(self, X):
        '''
        :param X: A_in
        :return:  split A_in to n_fold spare tensor with row
        '''
        A_fold_hat = []
        fold_len = (self.n_users + self.n_entities) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col])
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape).to(device)

    def _build_model_phase_II(self, h, r, pos_t, neg_t):
        """
        Compute Knowledge Graph Embeddings via TransR and Optimize KGE via BPR Loss.
        """
        h_e = self.user_entity_embed(h)   # (bs, emb_dim)
        pos_t_e = self.user_entity_embed(pos_t)
        neg_t_e = self.user_entity_embed(neg_t)

        r_e = self.relation_embed(r)      # （bs, kge_dim)

        W_r = self.trans_M[r]  # (bs, kge_dim, emb_dim)

        # project entity into relation space
        h_e = torch.bmm(W_r, h_e.unsqueeze(2)).squeeze(2)     # (bs, kge_dim)
        # print('h_e shape:', h_e.shape)
        pos_t_e = torch.bmm(W_r, pos_t_e.unsqueeze(2)).squeeze(2)
        neg_t_e = torch.bmm(W_r, neg_t_e.unsqueeze(2)).squeeze(2)

        # Equation (1)
        pos_score = torch.sum(torch.pow(h_e + r_e - pos_t_e, 2), dim=1)  # (bs,)
        neg_score = torch.sum(torch.pow(h_e + r_e - neg_t_e, 2), dim=1)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score) # (bs, )
        kg_loss = torch.mean(kg_loss)  # scalar

        l2_loss = _L2_loss_mean(h_e) + _L2_loss_mean(pos_t_e) +\
                    _L2_loss_mean(neg_t_e) + _L2_loss_mean(r_e)

        loss = kg_loss + self.regs[1] * l2_loss
        return loss

    def update_attention_A(self):
        fold_len = len(self.all_h_list) // self.n_fold
        kg_att_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            h = self.all_h_list[start:end]
            r = self.all_r_list[start:end]
            t = self.all_t_list[start:end]
            h = torch.LongTensor(h).to(device)
            r = torch.LongTensor(r).to(device)
            t = torch.LongTensor(t).to(device)
            part_kg_att_score = self._get_kg_attn_score(h, t, r)
            kg_att_score += list(part_kg_att_score.cpu().detach().numpy())

        kg_att_score = np.array(kg_att_score)  # (6420520, ), all pi(h,r,t)

        # softmax to A, i.e. obtain normalized pi(h,r,t)
        indices = np.mat([self.all_h_list, self.all_t_list])  # (2, 6420520)

        new_A = torch.sparse.softmax(torch.sparse_coo_tensor(indices, kg_att_score, self.A_in.shape), dim=1) # uncoalesced, (184166, 184166)

        new_A_values = new_A._values()
        new_A_indices = new_A._indices()

        rows = new_A_indices[0, :]
        cols = new_A_indices[1, :]
        new_A_sp = sp.coo_matrix((new_A_values, (rows, cols)), shape=(self.n_users + self.n_entities,
                                                                       self.n_users + self.n_entities))
        self.A_in = new_A_sp.tocsr()

    def _get_kg_attn_score(self, h, t, r):
        # Equation (4):  pi(h,r,t)=(Wr*et)T * tanh(Wr*eh+er)
        h_e = self.user_entity_embed(h)
        t_e = self.user_entity_embed(t)
        r_e = self.relation_embed(r)

        W_r = self.trans_M[r]  # (bs, kge_dim, emb_dim)

        h_e = torch.bmm(W_r, h_e.unsqueeze(2)).squeeze(2)     # (bs, kge_dim)
        t_e = torch.bmm(W_r, t_e.unsqueeze(2)).squeeze(2)     # (bs, kge_dim)

        kg_att_score = torch.bmm(t_e.unsqueeze(1), torch.tanh(h_e + r_e).unsqueeze(2)) # （bs, 1, 1)
        kg_att_score = kg_att_score.squeeze() # (bs, )
        return kg_att_score










