import numpy as np
import sys
import scipy.sparse as sp
import scipy.io as sio
import json
import math
import random
import re

import torch

import torch.utils.data as data
data_path = 'data_mooc/'  # 'data_goodreads/
import time
from transformers import BertTokenizer, BertConfig

def load_user_items(dataset='mooc'): # mooc: 83497
    '''获得所有用户的训练items和测试items, 以字典方式组织
    '''
    if dataset == 'mooc':
        train_file = 'train30.txt'
        test_file = 'test30.txt'
    elif dataset == 'goodreads':
        train_file = 'train60.txt' # 20, 30, 40, 50
        test_file = 'test.txt'
    else:
        print('Down train data error!!!')
        sys.exit(1)
    train_user_dict = {}
    with open(data_path + train_file, encoding='utf-8') as f: # train_first20.txt
        for line in f:
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            items = [int(item) for item in items]
            train_user_dict[int(u)] = items
    test_user_dict = {}
    with open(data_path + test_file, encoding='utf-8') as f: # test.txt
        for line in f:
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            items = [int(item) for item in items]
            test_user_dict[int(u)] = items

    return train_user_dict, test_user_dict

# 全局变量
train_user_dict, test_user_dict = load_user_items(dataset='mooc')  # 'goodreads'

def extract_goodreads_desc():
    '''
    抽取goodreads中的图书概要数据
    '''
    fw = open('data_goodreads/books_desc.json', 'w', encoding='utf-8')
    with open('data_goodreads/books_final.json', encoding='utf-8') as f:
        for line in f:
            book = json.loads(line)
            book_id = book['book_id']
            name = book['title']
            desc = book['description']
            desc = desc.replace('\n', ' ')
            desc = desc.replace('\u0001', ' ')
            desc = desc.replace('<!--EndFragment-->', ' ')
            desc = desc.replace('<!--StartFragment-->', ' ')
            desc = desc.replace('   ', ' ')
            desc = desc.replace('  ', ' ') 
            jsonstr = json.dumps({'id':book_id, 'name':name, 'desc': name + '. ' + desc})
            fw.write(jsonstr + '\n')
    fw.close()
            

# 本函数只需执行一次
def obtain_goodreads_book_train20():
    '''
    在goodreads数据集的train.txt取每个用户的前20(30, 40, 50, 60)个item,构成train20(30, 40, 50, 60).txt
    '''

    # 查看item出现的次数
    item_users = {}  # 每个item所交互的users
    with open(data_path + 'train.txt', encoding='utf-8') as f:
        for line in f:
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            for item in items:
                if item not in item_users:
                    item_users[item] = [u]
                else:
                    item_users[item].append(u)
    sorted_item_users = sorted(item_users.items(), key=lambda x:len(x[1]), reverse=True)
    items_ones = [item for item, users in item_users.items() if len(users) == 1]  # 交互次数为1次的item
    items_twice = [item for item, users in item_users.items() if len(users) == 2]  # 交互次数为2次的item
    items_three = [item for item, users in item_users.items() if len(users) == 3]  # 交互次数为3次的item
    items_four = [item for item, users in item_users.items() if len(users) == 4]  # 交互次数为4次的item
    items_five = [item for item, users in item_users.items() if len(users) == 5]  # 交互次数为5次的item
    items_six = [item for item, users in item_users.items() if len(users) == 6]  # 交互次数为6次的item
    items_seven = [item for item, users in item_users.items() if len(users) == 7]  # 交互次数为7次的item
    items_eight = [item for item, users in item_users.items() if len(users) == 8]  # 交互次数为8次的item
    items_nine = [item for item, users in item_users.items() if len(users) == 9]  # 交互次数为9次的item
    items_ten = [item for item, users in item_users.items() if len(users) == 10]  # 交互次数为10次的item
    items_11 = [item for item, users in item_users.items() if len(users) == 11]  # 交互次数为11次的item
    items_12 = [item for item, users in item_users.items() if len(users) == 12]  # 交互次数为12次的item
    items_13 = [item for item, users in item_users.items() if len(users) == 13]  # 交互次数为13次的item
    items_14 = [item for item, users in item_users.items() if len(users) == 14]  # 交互次数为14次的item
    items_15 = [item for item, users in item_users.items() if len(users) == 15]  # 交互次数为15次的item
    items_16 = [item for item, users in item_users.items() if len(users) == 16]  # 交互次数为16次的item
    items_17 = [item for item, users in item_users.items() if len(users) == 17]  # 交互次数为17次的item
    items_18 = [item for item, users in item_users.items() if len(users) == 18]  # 交互次数为18次的item
    items_19 = [item for item, users in item_users.items() if len(users) == 19]  # 交互次数为19次的item
    items_20 = [item for item, users in item_users.items() if len(users) == 20]  # 交互次数为20次的item
    items_21 = [item for item, users in item_users.items() if len(users) == 21]  # 交互次数为21次的item
    items_22 = [item for item, users in item_users.items() if len(users) == 22]  # 交互次数为22次的item
    items_23 = [item for item, users in item_users.items() if len(users) == 23]  # 交互次数为23次的item
    items_24 = [item for item, users in item_users.items() if len(users) == 24]  # 交互次数为24次的item
    items_25 = [item for item, users in item_users.items() if len(users) == 25]  # 交互次数为25次的item
    items_26 = [item for item, users in item_users.items() if len(users) == 26]  # 交互次数为26次的item
    items_27 = [item for item, users in item_users.items() if len(users) == 27]  # 交互次数为27次的item
    items_28 = [item for item, users in item_users.items() if len(users) == 28]  # 交互次数为28次的item
    items_29 = [item for item, users in item_users.items() if len(users) == 29]  # 交互次数为29次的item
    items_30 = [item for item, users in item_users.items() if len(users) == 30]  # 交互次数为30次的item
    print('item 1~5 times:', len(items_ones), len(items_twice), len(items_three), len(items_four), len(items_five))
    # print('sorted_item_users in item 499', len(sorted_item_users[499][1]), sorted_item_users[499])
    # print('sorted_item_users in item 999', len(sorted_item_users[999][1]), sorted_item_users[999])
    # print('sorted_item_users in item 1999', len(sorted_item_users[1999][1]), sorted_item_users[1999])
    # print('sorted_item_users in -100:', sorted_item_users[-100:])


    try_user_items = {}
    with open(data_path + 'train.txt') as f:
        for line in f:
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            if len(items) <= 60: # 20, 30, 40, 50
                try_user_items[u] = items
            else:
                choice_items = []
                for item in items:
                    if item in items_ones:
                        choice_items.append(item)
                for item in items:
                    if item in items_twice:
                        choice_items.append(item)
                for item in items:
                    if item in items_three:
                        choice_items.append(item)
                for item in items:
                    if item in items_four:
                        choice_items.append(item)
                for item in items:
                    if item in items_five:
                        choice_items.append(item)
                for item in items:
                    if item in items_six:
                        choice_items.append(item)
                for item in items:
                    if item in items_seven:
                        choice_items.append(item)
                for item in items:
                    if item in items_eight:
                        choice_items.append(item)
                for item in items:
                    if item in items_nine:
                        choice_items.append(item)
                for item in items:
                    if item in items_ten:
                        choice_items.append(item)
                # for train30.txt
                for item in items:
                    if item in items_11:
                        choice_items.append(item)
                for item in items:
                    if item in items_12:
                        choice_items.append(item)
                for item in items:
                    if item in items_13:
                        choice_items.append(item)
                for item in items:
                    if item in items_14:
                        choice_items.append(item)
                for item in items:
                    if item in items_15:
                        choice_items.append(item)
                # for train40.txt
                for item in items:
                    if item in items_16:
                        choice_items.append(item)
                for item in items:
                    if item in items_17:
                        choice_items.append(item)
                for item in items:
                    if item in items_18:
                        choice_items.append(item)
                for item in items:
                    if item in items_19:
                        choice_items.append(item)
                for item in items:
                    if item in items_20:
                        choice_items.append(item)
                # for train50.txt
                for item in items:
                    if item in items_21:
                        choice_items.append(item)
                for item in items:
                    if item in items_22:
                        choice_items.append(item)
                for item in items:
                    if item in items_23:
                        choice_items.append(item)
                for item in items:
                    if item in items_24:
                        choice_items.append(item)
                for item in items:
                    if item in items_25:
                        choice_items.append(item)
                # for train60.txt
                for item in items:
                    if item in items_26:
                        choice_items.append(item)
                for item in items:
                    if item in items_27:
                        choice_items.append(item)
                for item in items:
                    if item in items_28:
                        choice_items.append(item)
                for item in items:
                    if item in items_29:
                        choice_items.append(item)
                for item in items:
                    if item in items_30:
                        choice_items.append(item)
                if len(choice_items) >= 60: # 20, 30, 40, 50
                    try_user_items[u] = choice_items[:60] # 20, 30, 40, 50
                else:
                    remain_items = set(items) - set(choice_items)
                    rest_choice_items = np.random.choice(list(remain_items), 60-len(choice_items), replace=False).tolist() # 20,30,40,50
                    try_user_items[u] = choice_items + rest_choice_items
    print('try user_items length:', len(try_user_items)) # 70679
    print('interact num:', sum([len(items) for u, items in try_user_items.items()]))
    print('items by user 0:', try_user_items['0'])

    item20_users = {}
    for user, items in try_user_items.items():
        for item in items:
            if item not in item20_users:
                item20_users[item] = [user]
            else:
                item20_users[item].append(user)
    print('item20_users length:', len(item20_users))  # 24915
    items_ones = [item for item, users in item20_users.items() if len(users) == 1]  # 交互次数为1次的item
    items_twice = [item for item, users in item20_users.items() if len(users) == 2]  # 交互次数为2次的item
    items_three = [item for item, users in item20_users.items() if len(users) == 3]  # 交互次数为3次的item
    items_four = [item for item, users in item20_users.items() if len(users) == 4]  # 交互次数为4次的item
    items_five = [item for item, users in item20_users.items() if len(users) == 5]  # 交互次数为5次的item
    items_gt100 = [item for item, users in item20_users.items() if len(users) >= 100]  # 交互次数超过100次的item
    print('item20 1~5 times:', len(items_ones), len(items_twice), len(items_three), len(items_four), len(items_five))
    print('item20 >100 times:', len(items_gt100))

    fw = open(data_path + 'train60.txt', 'w', encoding='utf-8') # 20, 30, 40, 50
    for user, items in try_user_items.items():
        strs = user + ' ' + ' '.join(items)
        fw.write(strs + '\n')
    fw.close()

# 本函数只需执行一次
def longtext_first510():
    '''
    :return: 获得长文本的前510个字符
    '''
    def clean_space(text):
        """"
        处理多余的空格
        """
        match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
        should_replace_list = match_regex.findall(text)
        order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
        for i in order_replace_list:
            if i == u' ':
                continue
            new_i = i.strip()
            text = text.replace(i,new_i)
        return text

    fw = open('data30/course_longtext_stopword_reason_first510_30.json', 'w', encoding='utf-8')
    with open('data30/course_longtext_stopword_reason_movespam_30.json', encoding='utf-8') as f:
        for line in f:
            course = json.loads(line)
            cid = course['id']
            name = course['name']
            longtext = course['longtext']
            longtext = clean_space(longtext)[:510]
            jsonstr = json.dumps({'id':cid, 'name':name, 'first510':longtext}, ensure_ascii=False)
            fw.write(jsonstr + '\n')

def obtain_book_text_510():
    text_list = []
    with open('data_goodreads/books_desc.json', encoding='utf-8') as f: # goodreads
        for line in f:
            course = json.loads(line)
            text = course['desc'] # goodreads
            text_list.append(text)
    return text_list

# 本函数只需执行一次
def longtext_first2_510():
    '''
    :return: 获得长文本的前两段510个字符，overlap 50个字符
    '''
    def clean_space(text):
        """"
        处理多余的空格
        """
        match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
        should_replace_list = match_regex.findall(text)
        order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
        for i in order_replace_list:
            if i == u' ':
                continue
            new_i = i.strip()
            text = text.replace(i,new_i)
        return text

    # # 83497 <-> 1574
    # fw = open('data30/course_longtext_stopword_reason_first2_510_30.json', 'w', encoding='utf-8')
    # with open('data30/course_longtext_stopword_reason_movespam_30.json', encoding='utf-8') as f:
    # # 166379 <-> 1590
    fw = open('data/course_longtext_stopword_reason_first2_510.json', 'w', encoding='utf-8')
    with open('data/course_longtext_stopword_reason_movespam.json', encoding='utf-8') as f:
        for line in f:
            course = json.loads(line)
            cid = course['id']
            name = course['name']
            longtext = course['longtext']
            longtext = clean_space(longtext)
            text_pieces = []
            if len(longtext) < 560:
                text_pieces.append(longtext[:510])
            else:
                text_pieces.append(longtext[:510])
                text_pieces.append(longtext[460:970])
            jsonstr = json.dumps({'id':cid, 'name':name, 'first2_510':text_pieces}, ensure_ascii=False)
            fw.write(jsonstr + '\n')


# 本函数只需执行一次
def longtext_text510_2_comment510():
    '''
    :return: 获得课程文本的前两段510个字符，overlap 50个字符，以及课程评论的前510字符
    '''
    def clean_space(text):
        """"
        处理多余的空格
        """
        match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
        should_replace_list = match_regex.findall(text)
        order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
        for i in order_replace_list:
            if i == u' ':
                continue
            new_i = i.strip()
            text = text.replace(i,new_i)
        return text

    courseIds = [json.loads(line)['id'] for line in
                 open('data30/course_longtext_stopword_reason_movespam_30.json', encoding='utf-8')]
    print(len(courseIds))
    print(courseIds[:5])
    comment_list = []
    with open('data30/course-commentText-firstClean800-final.json', encoding='utf-8') as f:
        for line in f:
            course = json.loads(line)
            cid = course['id']
            comment = course['text']
            if cid in courseIds:
                comment_list.append((cid,comment))
    print(len(courseIds), len(comment_list))
    for cid1, (cid2,_) in zip(courseIds, comment_list):
        if cid1 != cid2:
            print('error!', cid1, cid2)

    fw = open('data30/course_longtextComment_stopword_reason_510_2_510_30.json', 'w', encoding='utf-8')
    with open('data30/course_longtext_stopword_reason_movespam_30.json', encoding='utf-8') as f:
        i = 0
        for line in f:
            course = json.loads(line)
            cid = course['id']
            name = course['name']
            longtext = course['longtext']
            longtext = clean_space(longtext)
            comment = comment_list[i][1]
            text_pieces = []
            if len(longtext) < 560:
                text_pieces.append(longtext[:510])
            else:
                text_pieces.append(longtext[:510])
                text_pieces.append(longtext[460:970])
            if len(comment) > 20:
                text_pieces.append(comment[:510])
            i += 1
            jsonstr = json.dumps({'id':cid, 'name':name, 'text_comment':text_pieces}, ensure_ascii=False)
            fw.write(jsonstr + '\n')

def obtain_text_first2_510(dataset='mooc'):
    if dataset == 'mooc':
        text_file = 'course_longtext_stopword_reason_first2_510_30.json'
    else:
        print('Down text file error!!!')
        sys.exit(1)
    text_list = []
    textlen_list = []

    with open(data_path + text_file, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text_pieces = data['first2_510']
            text_list.append(text_pieces)
            textlen_list.append(len(text_pieces))
    return text_list, textlen_list

# 存贮文本片段的tokenize,只需要执行一次
def tokenize_text(tokenizer, piece_text_list, dataset='mooc'):
    piece_lens = [len(piece) for piece in piece_text_list]
    piece_num = sum(piece_lens)

    all_inputs_ids = torch.zeros([piece_num, 512], dtype=torch.int)
    all_token_type_ids = torch.zeros([piece_num, 512], dtype=torch.int)
    all_attention_mask = torch.zeros([piece_num, 512], dtype=torch.int)
    low = 0
    for piece_text, piece_len in zip(piece_text_list, piece_lens):
        high = low + piece_len
        inputs = tokenizer(piece_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        token_shape = inputs['input_ids'].shape
        all_inputs_ids[low:high, :token_shape[1]] = inputs['input_ids']
        all_token_type_ids[low:high, :token_shape[1]] = inputs['token_type_ids']
        all_attention_mask[low:high, :token_shape[1]] = inputs['attention_mask']
        low = high

    if dataset == 'mooc':
        data_dir = 'piece_texts_tokenize30/'
    else:
        print('Down text file error!!!')
        sys.exit(1)

    torch.save(all_inputs_ids, data_dir + "all_inputs_ids.pt")  # 保存张量
    torch.save(all_token_type_ids, data_dir + "all_token_type_ids.pt")  # 保存张量
    torch.save(all_attention_mask, data_dir + "all_attention_mask.pt")  # 保存张量

# 存贮goodreads图书简介的510分词的tokenize,只需要执行一次
def book510_tokenize_text(tokenizer):
    item_num = 20547  # goodreads
    all_inputs_ids = torch.zeros([item_num, 512], dtype=torch.int)
    all_token_type_ids = torch.zeros([item_num, 512], dtype=torch.int)
    all_attention_mask = torch.zeros([item_num, 512], dtype=torch.int)
    with open('data_goodreads/books_desc.json', encoding='utf-8') as f: # goodreads
        i = 0
        for line in f:
            book = json.loads(line)
            text = book['desc']  # goodreads
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
            token_shape = inputs['input_ids'].shape
            print('token_shape:', token_shape)
            all_inputs_ids[i, :token_shape[1]] = inputs['input_ids']
            all_token_type_ids[i, :token_shape[1]] = inputs['token_type_ids']
            all_attention_mask[i, :token_shape[1]] = inputs['attention_mask']
            i += 1

    data_dir = 'texts_tokenize_goodreads/'  # goodreads
    torch.save(all_inputs_ids, data_dir + "all_inputs_ids.pt")  # 保存张量
    torch.save(all_token_type_ids, data_dir + "all_token_type_ids.pt")  # 保存张量
    torch.save(all_attention_mask, data_dir + "all_attention_mask.pt")  # 保存张量

def get_piece_texts_tokenize(dataset='mooc'):
    if dataset == 'mooc':
        data_dir = 'piece_texts_tokenize30/'
    else:
        print('Down text file error!!!')
        sys.exit(1)

    all_inputs_ids = torch.load(data_dir + "all_inputs_ids.pt")  # 加载张量
    all_token_type_ids = torch.load(data_dir + "all_token_type_ids.pt")  # 加载张量
    all_attention_mask = torch.load(data_dir + "all_attention_mask.pt")  # 加载张量
    return all_inputs_ids, all_token_type_ids, all_attention_mask

def get_book510_texts_tokenize():
    data_dir = 'texts_tokenize_goodreads/'  # goodreads
    all_inputs_ids = torch.load(data_dir + "all_inputs_ids.pt")  # 加载张量
    all_token_type_ids = torch.load(data_dir + "all_token_type_ids.pt")  # 加载张量
    all_attention_mask = torch.load(data_dir + "all_attention_mask.pt")  # 加载张量
    return all_inputs_ids, all_token_type_ids, all_attention_mask


def load_all(user_num=83497, dataset='mooc'): # 83497
    if dataset == 'mooc':
        train_file = 'train30.txt'
        test_file = 'test30.txt'
    elif dataset == 'goodreads':
        train_file = 'train60.txt' # 20, 30, 40, 50
        test_file = 'test.txt'
    else:
        print('Down train data error!!!')
        sys.exit(1)

    train_data = []
    min_items = 100
    with open(data_path + train_file, encoding='utf-8') as f:  # train30.txt
        k = 0
        for line in f:
            k += 1
            if k > user_num:
                break
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            if len(items) < min_items:
                min_items = len(items)
            for i in items:
                train_data.append([int(u), int(i)])
        print('min numbers of items:', min_items)  # 7

    test_data = []
    with open(data_path + test_file, encoding='utf-8') as f:  # test30.txt
        k = 0
        for line in f:
            k += 1
            if k > user_num:
                break
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            for i in items:
                test_data.append([int(u), int(i)])

    user_num, item_num = np.array(train_data)[:, 0].max() + 1, np.array(train_data)[:, 1].max() + 1

    # 以下生成训练集的稀疏矩阵。其作用是在实例化BBPRData时，生成负样本时排除在此稀疏矩阵中训练数据。
    # 生成的负样本不能是训练时出现的item, 所以必须获得完整的训练集的稀疏矩阵。
    # 对于goodreads,由于train50.txt只包含了部分训练集，故使用完整的训练集train.txt。
    all_train_data = []
    with open(data_path + 'train30.txt', encoding='utf-8') as f:  # 'train.txt'
        k = 0
        for line in f:
            k += 1
            if k > user_num:
                break
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            for i in items:
                all_train_data.append([int(u), int(i)])


    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)  # 基于稀疏矩阵的键字典
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0


    return train_data, test_data, user_num, item_num, train_mat

def obtain_train_data(user_num=166379):
    '''
    每轮训练数据为每个user随机取20个与之交互的item, 不足20按实际个数算
    :return: train_data_random20
    '''
    train_data_random20 = []
    user_texts_random_list = []
    with open(data_path + 'train_first20.txt', encoding='utf-8') as f:
        k = 0
        for line in f:
            k += 1
            if k > user_num:
                break
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            if len(items) > 20:
                choice_items = np.random.choice(items, 20, replace=False).tolist()
            else:
                choice_items = items
            user_texts_random_list.append([int(item) for item in choice_items])
            for i in choice_items:
                train_data_random20.append([int(u), int(i)])

    user_num, item_num = np.array(train_data_random20)[:, 0].max() + 1, np.array(train_data_random20)[:, 1].max() + 1

    return train_data_random20, user_texts_random_list, user_num, item_num

def obtain_users_bytext(user_num=166379, dataset='mooc'):
    if dataset == 'mooc':
        train_file = 'train30.txt'
    elif dataset == 'goodreads':
        train_file = 'train60.txt'  # 20, 30, 40, 50
    else:
        print('Down train data error!!!')
        sys.exit(1)
    user_texts_list = []
    with open(data_path + train_file, encoding='utf-8') as f:  # train_first20.txt
        k = 0
        for line in f:
            k += 1
            if k > user_num:
                break
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            user_texts_list.append([int(item) for item in items])
    return user_texts_list

def get_test(user_num, item_num=1574, dataset='mooc'): # mooc: 83497
    '''一次获得所有用户的已存在的测试数据，以及待测数据
    '''
    if dataset == 'mooc':
        train_file = 'train30.txt'
        test_file = 'test30.txt'
    else:
        print('Down train data error!!!')
        sys.exit(1)
    train_data_dic = {}
    with open(data_path + train_file, encoding='utf-8') as f: # train_first20.txt
        for line in f:
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            #items = [int(item) for item in items]
            train_data_dic[int(u)] = items
    test_data_dic = {}
    with open(data_path + test_file, encoding='utf-8') as f: # test.txt
        for line in f:
            u = line.strip().split(' ')[0]
            items = line.strip().split(' ')[1:]
            #items = [int(item) for item in items]
            test_data_dic[int(u)] = items
    print('train_data_dic[0]:', train_data_dic[0])
    print('Train and test data dict finished.')

    pos_tests_list, test_items_list = [], []
    for u in range(user_num):
        pos_tests = test_data_dic[u]
        # 从全部item中去掉train_items, 得到需要进行测试的item
        test_items = [str(item) for item in range(item_num) if str(item) not in train_data_dic[u]]
        pos_tests_list.append(pos_tests)
        test_items_list.append(test_items)

    return pos_tests_list, test_items_list

def get_one_test(u, item_num=1574): # mooc: 1574
    '''获得某个用户已存在的测试数据，以及待测数据
    '''
    pos_tests_list = test_user_dict[u]
    all_items = set(range(item_num))
    test_items_list = list(all_items - set(train_user_dict[u]))

    return pos_tests_list, test_items_list


class BBPRData(data.Dataset):
    def __init__(self, features, num_items, train_mat=None, num_ng=0, is_training=None):
        '''
        :param features: train_data or test_data, e.g. [[0, 738], [0, 605], ..., [166378, 185]]
        :param train_mat: sparse mat for user inter item
        '''
        super(BBPRData, self).__init__()
        self.features = features
        self.num_items = num_items
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_items)  # 0 ~ self.num_items-1
                while(u,j) in self.train_mat:
                    j = np.random.randint(self.num_items)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.num_ng * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if self.is_training else features[idx][1]
        return user, item_i, item_j


def triplet_bertInputs_first510(user, item_i, item_j, users_byitem, all_tokens):

    user_pos, user_byitem = obtain_users_pos(user, users_byitem)
    # print('user_pos:', user_pos)
    # print('user_byitem:', user_byitem)

    bs = user.shape[0]
    uni_pos = [value[0] for key, value in user_pos.items()] # e.g. [0, 15], while bs = 16
    user_byitem_unique = [user_byitem[pos] for pos in uni_pos]   # len: bs_uni
    #e.g. [[816, 895, 52, ... , 1227], [1446, 1334, 936, ..., 1572]], 1st list len is 20, 2nd list len is 15
    # print('user_bytext_unique:', user_bytext_unique)
    # print('user_pos:', user_pos)

    # item_i可以从user_byitem_uni_flat中提取处理
    user_byitem_uni_flat = []
    for itemno_list in user_byitem_unique:
        user_byitem_uni_flat.extend(itemno_list)
    # print('user_byitem_uni_flat:', len(user_byitem_uni_flat))
    # print('item_i:', item_i)
    itemi_pos = is_l1_in_l2(item_i.tolist(), user_byitem_uni_flat)
    # print('itemi_pos:', itemi_pos)
    if itemi_pos == -1:
        print('item_i not in !!!')
        sys.exit()

    # 获得已tokenize的user_inputs
    uni_bs = len(user_byitem_uni_flat)
    input_ids = torch.zeros([uni_bs, 512], dtype=torch.int)
    token_type_ids = torch.zeros([uni_bs, 512], dtype=torch.int)
    attention_mask = torch.zeros([uni_bs, 512], dtype=torch.int)
    i = 0
    for itemno in user_byitem_uni_flat:
        input_ids[i] = all_tokens[0][itemno]
        token_type_ids[i] = all_tokens[1][itemno]
        attention_mask[i] = all_tokens[2][itemno]
        i += 1
    user_inputs = {'input_ids':input_ids.to('cuda'),
                    'token_type_ids':token_type_ids.to('cuda'), 'attention_mask':attention_mask.to('cuda')}

    # 获得已tokenize的itemj_input
    input_ids = torch.zeros([bs, 512], dtype=torch.int)
    token_type_ids = torch.zeros([bs, 512], dtype=torch.int)
    attention_mask = torch.zeros([bs, 512], dtype=torch.int)
    i = 0
    for itemno in item_j:
        input_ids[i] = all_tokens[0][itemno]
        token_type_ids[i] = all_tokens[1][itemno]
        attention_mask[i] = all_tokens[2][itemno]
        i += 1
    itemj_input = {'input_ids':input_ids.to('cuda'),
                    'token_type_ids':token_type_ids.to('cuda'), 'attention_mask':attention_mask.to('cuda')}

    return user_inputs, bs, user_pos, user_byitem_unique, itemi_pos, itemj_input

def triplet_bertInputs_first2_510(user, item_i, item_j, users_byitem, piece_len_list, all_tokens):
    user_pos, user_byitem = obtain_users_pos(user, users_byitem)
    bs = user.shape[0]
    uni_pos = [value[0] for key, value in user_pos.items()] # e.g. [0, 15], while bs = 16
    user_byitem_unique = [user_byitem[pos] for pos in uni_pos]   # len: bs_uni

    # bt = time.time()
    # user_piece_texts = []
    # user_piece_lens = []
    # user_byitem_uni_flat = [] # item_i可以从user_bytext_uni_flat中提取处理
    # for itemno_list in user_byitem_unique:
    #     user_byitem_uni_flat.extend(itemno_list)
    #     for itemno in itemno_list:
    #         piece_text = piece_text_list[itemno]
    #         user_piece_lens.append(len(piece_text))
    #         user_piece_texts.extend(piece_text)
    # print('user_piece_texts {}s'.format(time.time() - bt))

    # item_i可以从user_byitem_uni_flat中提取处理
    user_byitem_uni_flat = []
    for itemno_list in user_byitem_unique:
        user_byitem_uni_flat.extend(itemno_list)
    itemi_pos = is_l1_in_l2(item_i.tolist(), user_byitem_uni_flat)
    if itemi_pos == -1:
        print('item_i not in !!!')
        sys.exit()

    piece_len_accumulate = []  # 片段长度累计值
    sum = 0
    for pl in piece_len_list:
        sum += pl
        piece_len_accumulate.append(sum)

    uni_piece_bs = 0
    for itemno in user_byitem_uni_flat:
        piece_len = piece_len_list[itemno]
        uni_piece_bs += piece_len
    input_ids = torch.zeros([uni_piece_bs, 512], dtype=torch.int)
    token_type_ids = torch.zeros([uni_piece_bs, 512], dtype=torch.int)
    attention_mask = torch.zeros([uni_piece_bs, 512], dtype=torch.int)
    user_piece_lens = []
    low = 0
    for itemno in user_byitem_uni_flat:
        piece_len = piece_len_list[itemno]
        high = low + piece_len
        # print('itemno:', itemno, '  low:', low, 'high:', high)
        user_piece_lens.append(piece_len)
        if itemno == 0:
            rlow = 0
        else:
            rlow = piece_len_accumulate[itemno-1]
        rhigh = piece_len_accumulate[itemno]
        #print('input_ids[low:high]:', input_ids[low:high].shape, 'all_tokens[0][rlow:rhigh]:', all_tokens[0][rlow:rhigh].shape)
        input_ids[low:high] = all_tokens[0][rlow:rhigh]
        token_type_ids[low:high] = all_tokens[1][rlow:rhigh]
        attention_mask[low:high] = all_tokens[2][rlow:rhigh]
        low = high
    user_inputs = {'input_ids':input_ids.to('cuda'),
                   'token_type_ids':token_type_ids.to('cuda'), 'attention_mask':attention_mask.to('cuda')}


    itemj_piece_bs = 0
    for itemno in item_j:
        piece_len = piece_len_list[itemno]
        itemj_piece_bs += piece_len
    input_ids = torch.zeros([itemj_piece_bs, 512], dtype=torch.int)
    token_type_ids = torch.zeros([itemj_piece_bs, 512], dtype=torch.int)
    attention_mask = torch.zeros([itemj_piece_bs, 512], dtype=torch.int)
    itemj_piece_len = []
    low = 0
    for itemno in item_j:
        piece_len = piece_len_list[itemno]
        high = low + piece_len
        itemj_piece_len.append(piece_len)
        if itemno == 0:
            rlow = 0
        else:
            rlow = piece_len_accumulate[itemno-1]
        rhigh = piece_len_accumulate[itemno]
        input_ids[low:high] = all_tokens[0][rlow:rhigh]
        token_type_ids[low:high] = all_tokens[1][rlow:rhigh]
        attention_mask[low:high] = all_tokens[2][rlow:rhigh]
        low = high
    itemj_input = {'input_ids':input_ids.to('cuda'),
                   'token_type_ids':token_type_ids.to('cuda'), 'attention_mask':attention_mask.to('cuda')}

    # bt = time.time()
    # user_inputs = tokenizer(user_piece_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    # itemj_input = tokenizer(itemj_piece_text, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
    # print('tokenizer {}s'.format(time.time() - bt))
    # print('user_inputs by tokenizer: \n', user_inputs)

    return user_inputs, bs, user_pos, user_byitem_unique, user_piece_lens, itemi_pos, itemj_input, itemj_piece_len

def obtain_users_pos(user, users_byitem):
    user_pos = {}  # 同一个user所在的行
    for i, u in enumerate(list(user.numpy())):
        if u in user_pos:
            user_pos[u].append(i)
        else:
            user_pos[u] = [i]

    user_bytext = []
    for u in user:
        user_bytext += [users_byitem[u]]

    # print('user_pos:', user_pos)
    # print('user_bytext:', user_bytext)
    return user_pos, user_bytext


if __name__ == '__main__':    
    # extract_goodreads_desc()
    
    # split_course_longtext2piece()

    # longtext_first510()

    # piece_text_list = obtain_piece_text()
    # print('piece_text_list 0...')
    # for i, piece in enumerate(piece_text_list[-1]):
    #     print('i:', i, piece)

    # pos_tests_list, test_items_list = get_test(100, item_num=1574)
    # print('post_tests_list first:', pos_tests_list[0])
    # print('test_items_list:', test_items_list[0])


    # piece_text_list = obtain_text_first2_510()
    # for i, piece in enumerate(piece_text_list):
    #     print('i:', i, [len(p) for p in piece])
    # print(piece_text_list[0][0])
    # print(piece_text_list[0][1])

    # longtext_text510_2_comment510()
    # text_list = obtain_textComment_510_2_510()
    # texts_len = [len(t) for t in text_list]
    # print('len:', texts_len)



    # 只需执行一次
    obtain_goo_book_train20()

    # # 只需执行一次
    # get_train_frist20()

    # # 只需执行一次
    # bookDesc_first_510()

    print('haha')