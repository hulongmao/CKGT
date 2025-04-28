'''
goodreads desc of first 510
'''

import time
import logging
import gc
import os
import numpy as np

import torch
import torch.utils.data as data
import torch.optim as optim

import data_utils_bbpr
import evaluate
# from config import Config

from transformers import BertTokenizer, BertConfig
from transformers import get_scheduler
import bert_fine_bpr_goodreads

# config = Config()

batch_size = 20  # mooc30:28 in P100
epochs = 5  #5
lr = 1e-4  # 以训练5轮，每轮只训练第一批的实验来看，属于比较好的学习率
accumulator = 4  # 梯度累积次数

fined_model_dir = 'bert-fine-bpr/'

def logger_config(log_path,logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

logger = logger_config(log_path='log_goodreads_0202.txt', logging_name='train_and_test')
logger.info('input: books_desc.json train60.txt train.txt test.txt\noutput: bert_fine_bpr2_goodreads_train60_4.pt\nbatch_size={}, lr={} RTX3090(24G)'.format(batch_size, lr))

############################## PREPARE DATASET ##########################
begin_time = time.time()
train_data, test_data, user_num ,item_num, train_mat = data_utils_bbpr.load_all(user_num=31867,dataset='goodreads') # mooc:83497
print('Data finished!', len(train_data), len(test_data), user_num, item_num, train_mat.shape)

text_list = data_utils_bbpr.obtain_book_text_510() # 70679

all_inputs_ids, all_token_type_ids, all_attention_mask = data_utils_bbpr.get_book510_texts_tokenize()
all_tokens = (all_inputs_ids, all_token_type_ids, all_attention_mask)
print('all_inputs_ids shape:', all_inputs_ids.shape)

users_byitem = data_utils_bbpr.obtain_users_bytext(31867, dataset='goodreads')  # mooc: 83497
print('users_bytextno first 5:', users_byitem[:5])

train_dataset = data_utils_bbpr.BBPRData(train_data,
                                         item_num, train_mat, num_ng=1, is_training=True)
train_loader = data.DataLoader(train_dataset,
                               batch_size=batch_size, shuffle=False, num_workers=4)  #4096, True

print('load data finished, consume {:.1f}s. train_dataset len: {}, user_num:{}, test_num:{}'.
      format(time.time()-begin_time, len(train_dataset), user_num, item_num))

########################## CREATE MODEL #################################
# mooc:bge-small-zh-v1.5, goodreads: bge-small-en-v1.5
args = {"model_name_or_path": "bge-small-en-v1.5/",
        "config_name": "bge-small-en-v1.5/",
        "tokenizer_name": "bge-small-en-v1.5/"
        }

tokenizer = BertTokenizer.from_pretrained(args["tokenizer_name"], do_lower_case=True,cache_dir=None)
print('\n\n\nbegin to build model...')
model = bert_fine_bpr_goodreads.BPR()
print('\n\n\nbuild model finished!')
# 加载已训练的模型，继续进行训练
model = torch.load('bert-fined-model/bert_fine_bpr_goodreads_train60_4.pt')
print('\n\n\nload pretrained model!')
model.cuda()

#验证一下
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name,param.size())

# 学习率调整
optimizer = optim.AdamW(model.parameters(), lr=lr)

num_update_steps_per_epoch = len(train_loader)
num_training_steps = epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

for epoch in range(epochs):

    model.train()
    train_loader.dataset.ng_sample()
    total_loss = 0

    count = 0  
    start_time = time.time()
    max_piece_uni = 0
    for user, item_i, item_j in train_loader:
        # print('count:', count)
        # print('user:', user)
        # print('item_i:', item_i)
        # print('item_j:', item_j)

        count += 1

        user_inputs, bs, user_pos, user_byitem_unique, itemi_pos, itemj_input = \
            data_utils_bbpr.triplet_bertInputs_first510(user, item_i, item_j, users_byitem, all_tokens)
        # print('count:', count, ' user_inputs:', user_inputs['input_ids'].shape)  # ~bs:104 for en(3060),
        if user_inputs['input_ids'].shape[0] > max_piece_uni:
            max_piece_uni = user_inputs['input_ids'].shape[0]
        # print('user_inputs:', user_inputs['input_ids'].shape)
        # print(user_inputs['input_ids'])
        # print('bs:', bs)
        # print('user_pos:', user_pos)
        # print('user_byitem_unique:', user_byitem_unique)
        # print('itemi_pos:', itemi_pos)
        # print('itemj_input:', itemj_input['input_ids'])

        itemi_represent, _, prediction_i, prediction_j = \
            model(user_inputs, bs, user_pos, user_byitem_unique, itemi_pos, itemj_input)
        # print('itemi_represent:', itemi_represent.shape, 'prediction_i:', prediction_i.shape, 'prediction_j:', prediction_j.shape)
        loss = -(prediction_i - prediction_j).sigmoid().log().sum()

        if count == 1:  # train60.txt
            print('First batch loss:', loss)
            print('itemi_represent for 0:', itemi_represent[0,:5])
            itemi_0 = itemi_represent[0]
            itemi_0 = itemi_0.detach().cpu().numpy()
        if count % 500 == 0:
            logger.info('Train. Epoch:{}, progress:{}%, loss:{}'.
                        format(epoch, round(count/len(train_loader)*100,0), loss))

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)

        total_loss += loss.item()

        del prediction_i, prediction_j, loss
        del user_inputs, bs, user_pos, itemi_pos, itemj_input
        gc.collect()
        torch.cuda.empty_cache()

        # if count >= 10:
        #     print('comsume {}s, max_piece_uni:{}'.format(time.time() - start_time, max_piece_uni))  # mooc: piece500:9s, piece300:10.67s, piece200:10.7s
        #     break
    logger.info('Train. Epoch:{}, loss:{}, max textpiece num per batch:{},  consume {}s'.
                format(epoch, total_loss/count, max_piece_uni, time.time() - start_time))
    # max_piece_uni = 101, 不会溢出

    # 一轮训练结束， 保存训练好的模型参数
    torch.save(model, 'bert-fined-model/bert_fine_bpr_goodreads_train60_' + str(epoch) + '.pt')

    print('test...')
    # 载入训练好的模型
    model_load = torch.load('bert-fined-model/bert_fine_bpr_goodreads_train60_' + str(epoch) + '.pt')
    model_load.cuda()
    print('funed model loaded!!!')

#     # # （1）手动测试
#     # user = torch.as_tensor([1,1,1,1])
#     # item_i = torch.as_tensor([1446, 1334, 936, 570])
#     # item_j = torch.as_tensor([123, 867, 98, 326])
#     # user_inputs, bs, user_pos, user_bytext_unique, itemi_pos, itemj_input = \
#     #     data_utils_bbpr.triplet_bertInputs_first510(user, item_i, item_j, users_bytextno_random20, text_list, tokenizer)
#     # _, _, prediction_i, prediction_j = model(user_inputs, bs, user_pos, user_bytext_unique, itemi_pos, itemj_input)
#     # print(prediction_i)
#     # print(prediction_j)
#     # loss = -(prediction_i - prediction_j).sigmoid().log().sum()
#     # print('test loss:', loss)
#     # # 结果说明： epochs=1, loss=3.68; epochs=5, loss=3.09;
#     # # epochs=10, loss=2.83 (prediction_i: tensor([-1.8349, -1.6988, -1.3127, -1.5057]
#     # #                       prediction_j: tensor([-1.5288, -1.5472, -1.7128, -1.5717]
#     # # 感觉很怪！
    #
    # （2） 编程进行测试
    texts_represent = evaluate.get_text_represent(None, model_load, text_list, tokenizer)
    print('texts_represent finished, shape is:', texts_represent.shape)
    itemj_0 = texts_represent[0]
    print('itemi_0 shape:', itemi_0.shape, 'itemj_0 shape:', itemj_0.shape)
    # 计算余弦值， 在[-1,1]之间
    num = float(np.dot(itemi_0, itemj_0))  # 向量点乘
    denom = np.linalg.norm(itemi_0) * np.linalg.norm(itemj_0)  # 求模长的乘积
    cos = 0.5 + 0.5 * (num / denom) if denom != 0 else 0

    # 输出模型结构，方便找到对应的层
    # parm={}
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())
    #     parm[name]=parameters.detach().cpu().numpy()
    # print('parm:\n', parm)

    # # 测试模型中某个层的使用
    # item_i = torch.as_tensor([1505, 605, 922, 738])
    # texts_represent = torch.as_tensor(texts_represent, dtype=torch.float32)
    # item_i_represent = texts_represent[item_i].to('cuda')
    # print('item_i_represent:', item_i_represent)
    # item_latent = model.item_trans(item_i_represent)
    # print('item_latent:', item_latent.shape)

    # 正式测试
    r = evaluate.get_recall_simple(model_load, texts_represent, users_byitem)
    logger.info('Test. Epoch:{}, recall:{}, cos value of itemi_0 and itemj_0:{}'.format(epoch, r, cos))
    print()

# 关机
os.system('shutdown')

# 训练5轮，模型为bge-small-en-v1.5, RTX3090 24G, batch_size:20.
# (1)使用注意力机制集成user_present, self.da=256。部分冻结bert层（保留layer.10之后的层）, lr=1e-4, 运用学习率调整.







