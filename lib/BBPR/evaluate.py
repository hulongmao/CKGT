import numpy as np
import heapq
import multiprocessing
import data_utils_bbpr
import torch
import torch.nn as nn
import scipy.io as sio
import time

cores = multiprocessing.cpu_count() // 2
#cores = 1
print('cpu cores:', cores)

# mooc
# user_num = 83497
# item_num = 1574
# test_pos_items_list, test_items_list = data_utils_bbpr.get_test(user_num)

# goodreads
user_num = 31867
item_num = 20547

top_k = 20

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get) # K_max items with largest score
    # print('K_max_item_score:', K_max_item_score)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r

def recall_at_k(r, k, all_pos_num):
    # hulongmao, 2024.11.25
    if all_pos_num == 0:
        return 0
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def test_one_user(x):
    rating = x[0]
    u = x[1]
    # test_pos_items = test_pos_items_list[u]
    # test_items = test_items_list[u]
    test_pos_items, test_items = data_utils_bbpr.get_one_test(u=u, item_num=item_num)
    recalls = ranklist_by_heapq(test_pos_items, test_items, rating, [top_k])
    r = recall_at_k(recalls, top_k, len(test_pos_items))
    return r

def get_text_represent(text_represent_dir, model, text_list, tokenizer):
    model.eval()
    d_model = 384 # bge-small-en-v1.5 for goodreads
    if text_represent_dir is not None:
        print('load text represent.')
        variables = sio.loadmat(text_represent_dir)
        texts_represent = variables['texts_represent']
    else:
        print('Calculate text represent by eval model...')
        texts_represent = np.zeros((item_num, d_model), dtype=np.float32)
        for i in range(item_num):
            item_text = text_list[i]
            item_input = tokenizer(item_text, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
            with torch.no_grad():
                _, item_present, _, _ = model(itemj_input=item_input, is_training=False)  # 训练两轮，保存模型，再加载进行预测。两次的值一样
            item_present = item_present.detach().cpu().numpy()
            texts_represent[i] = item_present[0]

        # 保存
        # goodreads
        sio.savemat('goodreads_text510_represent.mat',{"texts_represent":texts_represent})
        variables = sio.loadmat('goodreads_text510_represent.mat')
        texts_represent_load = variables['texts_represent']
        # print('orig:', texts_represent[:5, :5])
        # print('loaded:', texts_represent[:5, :5])
        assert np.array_equal(texts_represent_load,texts_represent)

    return texts_represent

def get_piece_text_represent(text_represent_dir, model, piece_text_list, tokenizer):
    model.eval()
    d_model = 512 # bge-small-zh-v1.5
    # d_model = 384 # bge-small-en-v1.5
    piece_text_lens = [len(pieces) for pieces in piece_text_list]
    if text_represent_dir is not None:
        print('load text represent.')
        variables = sio.loadmat(text_represent_dir)
        texts_represent = variables['texts_represent']
    else:
        print('Calculate text represent by eval model...')
        texts_represent = np.zeros((item_num, d_model), dtype=np.float32)
        for i in range(len(piece_text_list)):
            piece_text = piece_text_list[i]
            piece_text_len = [piece_text_lens[i]]
            item_input = tokenizer(piece_text, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
            with torch.no_grad():
                _, item_present, _, _ = model(itemj_input=item_input, itemj_piece_len=piece_text_len, is_training=False)
            # # method b: attention merge
            # a = nn.functional.softmax(torch.mm(model.ws02, torch.mm(model.Ws01, item_present.transpose(0,1))),
            #                           dim=1)
            # item_present = torch.mm(a, item_present)[0]
            # # end method b
            # method c: simple weight
            # if piece_text_lens[i] == 2:
            #     print('piece_text_len:', piece_text_len)
            #     print('item_present:', item_present.shape)
            #     item_present = torch.mm(model.sw, item_present)[0]
            # end method c
            item_present = item_present.detach().cpu().numpy()  # (2, d_model) or (1, d_model)
            # # method a: mean pool
            # item_present = np.mean(item_present, axis=0)
            # if i == 524:
            #     print('item_present of 524 by model out:', item_present[0].shape, item_present[0,:5])
            texts_represent[i] = item_present[0]

        # 保存
        # mooc:text2_510_represent.mat
        sio.savemat('book_text2_510_represent.mat',{"texts_represent":texts_represent})
        variables = sio.loadmat('book_text2_510_represent.mat')
        texts_represent_load = variables['texts_represent']
        assert np.array_equal(texts_represent_load,texts_represent)

    return texts_represent

def get_recall_simple(model, texts_represent, user_bytext_list):
    # user_bytext_list = []
    # with open('data/train.txt', encoding='utf-8') as f:
    #     for line in f:
    #         items = line.strip().split(' ')[1:]
    #         items = [int(item) for item in items]
    #         user_bytext_list.append(items[:20])

    items_represent = texts_represent
    items_represent = torch.as_tensor(items_represent).to('cuda')
    # print('items (texts) 0 represent:', items_represent[0, :5])
    item_latent = model.item_trans(items_represent)

    r_sum = 0
    count = 0

    btime = time.time()
    for u in range(user_num):  # 一个用户
        u_bytext = user_bytext_list[u]
        u_represent = texts_represent[u_bytext]
        u_represent = torch.as_tensor(u_represent).to('cuda')

        # # method 1: mean
        # u_represent = torch.mean(u_represent, dim=0, keepdim=True) # (1,288)
        # if u == 0:
        #     print('user 0 represent:', u_represent.shape, u_represent[0, :5])
        # # method 2: add userid vec
        # userid_vec = model.matrix_D(torch.as_tensor(u).unsqueeze(0).to('cuda'))
        # parag_items_vec = torch.cat([userid_vec, u_represent], dim=0)
        # u_represent = torch.mean(parag_items_vec, dim=0, keepdim=True) # (1,288)
        # method 3: use attention to intergrate vector of user by items
        a = nn.functional.softmax(torch.mm(model.ws2, torch.mm(model.Ws1, u_represent.transpose(0,1))),
                                dim=1)
        u_represent = torch.mm(a, u_represent)

        user_latent = model.user_trans(u_represent)
        # print('user_latent:', user_latent.shape, ' item_latent:', item_latent.shape)  # [1, 128], [1590, 128]
        rate_one = torch.mm(user_latent, item_latent.transpose(0, 1))

        rate_one = rate_one.detach().cpu().numpy().squeeze(axis=0)
        # print('rate one:', rate_one.shape)
        # print('rate one with shape of (24915,):', rate_one)

        # test_pos_items = test_pos_items_list[u]
        # test_items = test_items_list[u]
        test_pos_items, test_items = data_utils_bbpr.get_one_test(u=u, item_num=item_num)

        recalls = ranklist_by_heapq(test_pos_items, test_items, rate_one, [top_k])
        r = recall_at_k(recalls, top_k, len(test_pos_items))

        r_sum += r
        count += 1

    r = r_sum / count
    # print('Final!', ' r_sum:', r_sum, ' count:', count)
    print('Test {} users consume {}s, recall is {}'.format(user_num, time.time() - btime, r))

    return r


def get_recall(model, users_bytextno_random20, text_list, tokenizer):
    pool = multiprocessing.Pool(cores)

    u_batch_size = 6 # A10: bpr1~5, 3; bpr6: 1?; bpr7:6
    if user_num % u_batch_size == 0:
        n_user_batch = user_num // u_batch_size
    else:
        n_user_batch = user_num // u_batch_size + 1
    r_sum = 0
    count = 0
    for u_batch_id in range(n_user_batch):
        if u_batch_id % 1000 == 0:
            print('finished {}%'.format(u_batch_id * 100 // n_user_batch ))
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        if end > user_num:
            end = user_num
        item_is = []
        item_js = []
        users = []
        # u_pos = {}
        i = 0
        for u in range(start, end):
            users += [u] * item_num
            item_is += list(range(item_num))
            item_js += list(range(item_num))
            # u_pos[u] = list(range(i*item_num, (i+1)*item_num))
            i += 1

        print('users:', users)
        users = torch.as_tensor(users)
        item_is = torch.as_tensor(item_is)
        item_js = torch.as_tensor(item_js)

        user_inputs, bs, user_pos, user_bytext_unique, itemi_pos, itemj_input = \
            data_utils_bbpr.triplet_bertInputs_first510(users, item_is, item_js, users_bytextno_random20, text_list, tokenizer)

        _, _, rate_batch, _ = model(user_inputs, bs, user_pos, user_bytext_unique, itemi_pos, itemj_input)

        rate_batch = rate_batch.detach().cpu().numpy().reshape((-1, item_num))
        user_batch_rating_uid = zip(rate_batch, list(range(start, end)))
        batch_r = pool.map(test_one_user, user_batch_rating_uid)

        del user_inputs, bs, user_pos, user_bytext_unique, itemi_pos, itemj_input, rate_batch

        for r in batch_r:
            r_sum += r
            count += 1

    r = r_sum / count

    return r

