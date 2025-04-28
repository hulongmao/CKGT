import data_utils_bbpr
import torch
import torch.nn as nn
import evaluate
from transformers import BertTokenizer
import numpy as np
import scipy.io as sio

user_num = 83497 # 83497
item_num = 1574  # 1574
factor_num=128

piece_text_list, piece_len_list = data_utils_bbpr.obtain_text_first2_510()  # 只包括课程文本片段，每门课程不超过前2片段组成
all_inputs_ids, all_token_type_ids, all_attention_mask = data_utils_bbpr.get_piece_texts_tokenize()
all_tokens = (all_inputs_ids, all_token_type_ids, all_attention_mask)

users_byitem = data_utils_bbpr.obtain_users_bytext(83497)  #83497

# bge-small-zh-v1.5
args = {"model_name_or_path": "bge-small-zh-v1.5/",
        "config_name": "bge-small-zh-v1.5/",
        "tokenizer_name": "bge-small-zh-v1.5/"
        }

tokenizer = BertTokenizer.from_pretrained(args["tokenizer_name"], do_lower_case=True,cache_dir=None)

# 载入训练好的模型
model = torch.load('bert-fined-model/bert_fine_bpr2_83497_4.pt')
model.cuda()

# 获得items_latent以及users_latent
texts_represent = evaluate.get_piece_text_represent(None, model, piece_text_list, tokenizer)
items_latent = model.item_trans(torch.as_tensor(texts_represent).to('cuda'))
items_latent = items_latent.detach().cpu().numpy()

users_latent = np.zeros((user_num, factor_num))
for u in range(user_num):  # 一个用户
    u_bytext = users_byitem[u]
    u_represent = texts_represent[u_bytext]
    u_represent = torch.as_tensor(u_represent).to('cuda')

    # method 3: use attention to intergrate vector of user by items
    a = nn.functional.softmax(torch.mm(model.ws2, torch.mm(model.Ws1, u_represent.transpose(0,1))),
                              dim=1)
    u_represent = torch.mm(a, u_represent)

    user_latent = model.user_trans(u_represent)
    u_latent = user_latent.detach().cpu().numpy()
    users_latent[u] = u_latent

# 将items_latent以及users_latent写入文件
sio.savemat('users_items_represent/ui_texts_latent.mat',{"users_latent":users_latent, "items_latent":items_latent})
print('user and item texts latent finished.')
