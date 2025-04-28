'''
hulongmao, 2024.9.26
method2: 知识图谱，kg_refine_30.txt + kg_user30.txt。将用户文本表示和课程文本表示作为可训练的嵌入层，
        使用kgat_emb || W * u_texts_emb(i_texts_emb) 进行集成。
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from model.KGAT import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class CKGT(nn.Module):
    def __init__(self, data_config, pretrain_data, args):
        super(CKGC, self).__init__()
        self._parse_args(data_config, pretrain_data, args)

        self.kgat = KGAT(data_config, pretrain_data, args)

        self.u_texts_embed = nn.Embedding(self.n_users, self.emb_dim) # (83497, 64)
        self.i_texts_embed = nn.Embedding(self.n_items, self.emb_dim) # (1574, 64)
        self.u_texts_embed.weight = nn.Parameter(self.u_pertrain_texts_embed)
        self.i_texts_embed.weight = nn.Parameter(self.i_pertrain_texts_embed)
        
        # aggregation 1: kgat_emb || W * u_texts_emb(i_texts_emb) 
        self.w_utext = nn.Embedding(self.n_users, 1)
        self.w_ctext = nn.Embedding(self.n_items, 1)
        nn.init.xavier_uniform_(self.w_utext.weight)
        nn.init.xavier_uniform_(self.w_utext.weight)

    def _parse_args(self, data_config, pretrain_data, args):
        self.model_type = 'ckgc'
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

        self.u_pertrain_texts_embed = data_config['u_texts_embed']
        self.i_pertrain_texts_embed = data_config['i_texts_embed']

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

        self.dropout = args.node_dropout # for HTN

        self.verbose = args.verbose

    def _get_ui_texts_embeddings(self, u, pos_i, neg_i):
        utext_e = self.u_texts_embed(u)
        pos_i_text_e = self.i_texts_embed(pos_i)
        neg_i_text_e = self.i_texts_embed(neg_i)
        return utext_e, pos_i_text_e, neg_i_text_e


    def _build_model_phase_I(self, u, pos_i, neg_i):
        """
        First Compute Graph-based Representations of All Users & Items & KG Entities,
        Then Compute Context-based Representations of All Users & Items,
        Final Optimize Recommendation (CF) via BPR Loss.
        """
        ua_embeddings, ea_embeddings = self.kgat._create_ego_side_bi_embed()
        u_e = ua_embeddings[u]
        pos_i_e = ea_embeddings[pos_i]
        neg_i_e = ea_embeddings[neg_i]

        utext_e, pos_i_text_e, neg_i_text_e = self._get_ui_texts_embeddings(u, pos_i, neg_i)
        # print('u_e shape:', u_e.shape,'  utext_e shape:', utext_e.shape)
        utext_e = F.normalize(utext_e, dim=1)
        pos_i_text_e = F.normalize(pos_i_text_e, dim=1)
        neg_i_text_e = F.normalize(neg_i_text_e, dim=1)

        # aggregate 1
        utext_e = torch.multiply(self.w_utext(u), utext_e)
        pos_i_text_e = torch.multiply(self.w_ctext(pos_i), pos_i_text_e)
        neg_i_text_e = torch.multiply(self.w_ctext(neg_i), neg_i_text_e)
        u_e = torch.concat([u_e, utext_e], dim=1)
        pos_i_e = torch.concat([pos_i_e, pos_i_text_e], dim=1)
        neg_i_e = torch.concat([neg_i_e, neg_i_text_e], dim=1)

        # calc cf loss
        pos_scores = torch.sum(torch.multiply(u_e, pos_i_e), dim=1, keepdim=False) # (bs, )
        neg_scores = torch.sum(torch.multiply(u_e, neg_i_e), dim=1, keepdim=False) # (bs, )

        l2_loss = _L2_loss_mean(u_e) + _L2_loss_mean(pos_i_e) + _L2_loss_mean(neg_i_e)

        base_loss = (-1.0) * F.logsigmoid(pos_scores - neg_scores ) # (bs, )
        base_loss = torch.mean(base_loss)  # scalar

        loss = base_loss + self.regs[0] * l2_loss
        return loss

    def _build_model_phase_II(self, h, r, pos_t, neg_t):
        return self.kgat._build_model_phase_II(h, r, pos_t, neg_t)

    def _calc_cf_score(self, u, i):
        '''
        :param u: (bs, )
        :param i: (1590,)
        :param user_bytext: (bs, 20)
        :param user_pos: dict, len is bs
        :return:
        '''
        # print('_calc_cf_score(self, u, i, user_bytext, user_pos)')
        # print('u:', u.shape, 'i:', i.shape, 'user_bytext:', user_bytext.shape, 'user_pos:', user_pos)
        ua_embeddings, ea_embeddings = self.kgat._create_ego_side_bi_embed()
        u_e = ua_embeddings[u]   # (bs, 128)
        i_e = ea_embeddings[i]   # (1590, 128)

        utext_e, i_text_e, _ = self._get_ui_texts_embeddings(u, i, i)

        utext_e = F.normalize(utext_e, dim=1)
        i_text_e = F.normalize(i_text_e, dim=1)

        # aggregate 1
        utext_e = torch.multiply(self.w_utext(u), utext_e)
        i_text_e = torch.multiply(self.w_ctext(i), i_text_e)
        u_e = torch.concat([u_e, utext_e], dim=1)
        i_e = torch.concat([i_e, i_text_e], dim=1)

        cf_scores = torch.matmul(u_e, i_e.transpose(0,1))
        return cf_scores

    def forward(self, *input, phase):
        if phase == 'cf':
            return self._build_model_phase_I(*input)
        if phase == 'kge':
            return self._build_model_phase_II(*input)
        if phase == 'predict':
            return self._calc_cf_score(*input)











