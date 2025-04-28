'''
基于bert微调的课程文本推荐
(1)每个user由交互的课程文本片段表示，每个item（不超过22门课程）由课程文本片段表示
(2)微调的bert类课程文本片段转换为向量，然后利用均值池化融合成课程文本向量
(3)利用用注意力机制将课程文本向量使融合user向量
(4)user向量与item向量内积进行推荐
'''
import random

import torch
import torch.nn as nn

from transformers import (WEIGHTS_NAME,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          BertConfig,
                          BertModel,
                          BertPreTrainedModel,
                          BertTokenizer,)

class BertForPieceText(BertPreTrainedModel):

    def __init__(self, config): 

        '''
        :param factor_num: 用户及课程因子向量（隐向量）的维度
        '''
        super().__init__(config)

        self.bert = BertModel(config) # Building the model from the config, the model is randomly initialized!        

        self.init_weights()

    def forward(
            self,
            input_ids=None,        # 输入的id, 模型会帮你把id转成embedding
            attention_mask=None,   # attention里的mask,用1和0填充：1s表示应注意相应的标记，0s表示不应注意相应的标记（即，模型的注意力层应忽略它们）。
            token_type_ids=None,   # [CLS]A[SEP]B[SEP] 告诉模型输入的哪一部分是第一句，那一部分是第二句。0表示第一句，1表示第二句。
            position_ids=None,     # 位置id
            head_mask=None,        # 哪个head需要被mask掉
            inputs_embeds=None,    # 可以选择不输入id, 直接输入embedding
            labels=None,           # 做分类时需要的label
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # bge-small-zh-v1.5 or bge-small-en-v1.5
        sentence_embeddings = outputs[0][:, 0]  # zh: (bs, 512),  en: (bs, 384)
        return sentence_embeddings
    
class BPR(nn.Module):
    
    def __init__(self, factor_num=128, dropout=0.1):
        super(BPR, self).__init__()

        # bert layer
        # mooc: bge-small-zh-v1.5, goodreads: bge-small-en-v1.5
        args = {"model_name_or_path": "bge-small-zh-v1.5/",
                "config_name": "bge-small-zh-v1.5/",
                "tokenizer_name": "bge-small-zh-v1.5/"
                }
        config = BertConfig.from_pretrained(
            args["config_name"],
            finetuning_task="",
            cache_dir=None,
        )
        self.bert_for_piece = BertForPieceText.from_pretrained(
            args["model_name_or_path"],
            from_tf=bool(".ckpt" in args["model_name_or_path"]),
            config=config,
            cache_dir=None,
        )

        # 冻结bert模型的输出, 只保留最后一层
        for name ,param in self.bert_for_piece.named_parameters():
            param.requires_grad = False
            # if 'layer.2' in name:  # 保留layer2之后的层, for bge-small-zh-v1.5
            # if 'layer.10' in name:  # 保留layer10之后的层, for bge-small-en-v1.5
            #     param.requires_grad = True
            #     break

        # bge-small-zh-v1.5: self.d_model = 512,  bge-small-en-v1.5: self.d_model = 384
        self.d_model = 512

        # method 3: use attention to intergrate vector of user by items, inspiration comes from "A Structured Self-Attentive Sentence Embedding"
        # original: a = softmax (ws2tanh (Ws1HT)), here: a = softmax(ws2tanh(Ws1(BT)), and B is vector of user by items
        self.da = 256
        self.Ws1 = nn.Parameter(torch.Tensor(self.da, self.d_model))
        self.ws2 = nn.Parameter(torch.Tensor(1, self.da))
        nn.init.normal_(self.Ws1, std=0.01)
        nn.init.normal_(self.ws2, std=0.01)

        # method b: text piece vecs -> text vec by using attention merge
        self.Ws01 = nn.Parameter(torch.Tensor(256, self.d_model))
        self.ws02 = nn.Parameter(torch.Tensor(1, 256))
        nn.init.normal_(self.Ws01, std=0.01)
        nn.init.normal_(self.ws02, std=0.01)
        # # method c: simple fix
        # self.sw = nn.Parameter(torch.tensor([[0.5, 0.5]]))


        # factor layer
        self.user_trans = nn.Linear(self.d_model, factor_num)  # bert-base-chinese:768
        self.item_trans = nn.Linear(self.d_model, factor_num)  # bert-base-chinese:768
        nn.init.normal_(self.user_trans.weight.data, std=0.01)
        nn.init.normal_(self.item_trans.weight.data, std=0.01)

    def forward(self, user_inputs=None, bs=None, user_pos=None, user_bytext_unique=None,
                user_piece_lens=None, itemi_pos=None, itemj_input=None, itemj_piece_len=None, is_training=True):

        if is_training:
            user_pooled_out = self.bert_for_piece(**user_inputs)  # (bs_uni, self.d_model)
            # print('uni user_pooled_out:', user_pooled_out.shape)
            # print('uni user_pooled first column:', user_pooled_out[:, 0])
            # print('uni user_pooled for user 0:', torch.mean(user_pooled_out[:15, 0]))
            # print('uni user_pooled for user 1:', torch.mean(user_pooled_out[15:, 0]))
            # print(user_pooled_out[:, 0])
            # print('bs:', bs, 'user_pos:', user_pos)
            # print('user_pos.keys():', list(user_pos.keys()))

            user_represent, uni_user_ver = self.decompress_vector_fromPieceUni(bs, user_pos, user_bytext_unique,
                                                                 user_piece_lens, user_pooled_out.clone())
            # print('user_represent:', user_represent.shape)
            # print('user_pos[0][0]:', user_pos[0][0], 'user_pos[1][0]:', user_pos[1][0])
            # print('user 0:', user_represent[user_pos[0][0]][:5])
            # print('user 0:', user_represent[user_pos[0][1]][:5])
            # print('user 1:', user_represent[user_pos[1][0]][:5])

            itemi_represent = uni_user_ver.clone()[itemi_pos:itemi_pos+bs]
            # Pytorch张量（Tensor）复制 https://blog.csdn.net/winycg/article/details/100813519
            # print('itemi_represent:', itemi_represent.shape)
            # print('item 524:', itemi_represent[0, :5])
            # print('item 1285:', itemi_represent[1, :5])
            piece_itemj_represent = self.bert_for_piece(**itemj_input)
            itemj_represent = self.decompress_vector_fromPiece(piece_itemj_represent, itemj_piece_len)

            user_latent = self.user_trans(user_represent)
            i_latent = self.item_trans(itemi_represent)
            j_latent = self.item_trans(itemj_represent)

            prediction_i = (user_latent * i_latent).sum(dim=-1)
            prediction_j = (user_latent * j_latent).sum(dim=-1)
        else:
            user_latent, prediction_i, prediction_j = None, None, None
            itemi_represent = None
            piece_itemj_represent = self.bert_for_piece(**itemj_input)
            itemj_represent = self.decompress_vector_fromPiece(piece_itemj_represent, itemj_piece_len)
        return itemi_represent, itemj_represent, prediction_i, prediction_j

    def decompress_vector_fromPieceUni(self, bs, user_pos, user_bytext_unique, user_piece_lens, piece_uni_user_vec):
        '''
        :param user_piece_lens: piece num of item interacted by batch users, i.e. bs=16, len(user_piece_lens)=24
        '''
        # 将压缩后的user文本向量转换为全向量：（bs_piece_uni, d_model) -> (bs, d_model)

        # text piece vecs -> text vec
        uni_user_vec = torch.zeros(len(user_piece_lens), self.d_model, dtype=torch.float32).cuda()
        piece_low = 0
        for i, l in enumerate(user_piece_lens): # e.g. [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2]
            piece_high = piece_low + l

            # # method a: mean pool
            # text_vec = torch.mean(piece_uni_user_vec[piece_low:piece_high], dim=0, keepdim=False)
            # method b: attention merge
            text_vecs = piece_uni_user_vec[piece_low:piece_high]
            a = nn.functional.softmax(torch.mm(self.ws02, torch.mm(self.Ws01, text_vecs.transpose(0,1))),
                                      dim=1)
            text_vec = torch.mm(a, text_vecs)[0]
            # # method c: simple weight
            # if l == 1:
            #     text_vec = piece_uni_user_vec[piece_low]
            # elif l == 2:
            #     text_vecs = piece_uni_user_vec[piece_low:piece_high]
            #     text_vec = torch.mm(self.sw, text_vecs)[0]
            # else:
            #     print('error!!!')

            uni_user_vec[i] = text_vec
            piece_low = piece_high

        user_vec = torch.zeros(bs, self.d_model, dtype=torch.float32).cuda()
        user_low = 0
        for (_, pos), user_uni in zip(user_pos.items(), user_bytext_unique):
            user_high = user_low + len(user_uni)

            # method 3: use attention to intergrate vector of user by items
            one_user_vecs = uni_user_vec[user_low:user_high]  # (n, d_model)
            a = nn.functional.softmax(torch.mm(self.ws2, torch.mm(self.Ws1, one_user_vecs.transpose(0,1))),
                                                 dim=1)
            one_user_vec = torch.mm(a, one_user_vecs)[0]

            user_vec[pos] = one_user_vec
            user_low = user_high
        return user_vec, uni_user_vec

    def decompress_vector_fromPiece(self, piece_itemj_represent, itemj_piece_len):
        # 将压缩后的user文本向量转换为全向量：（bs_piece, d_model) -> (bs, d_model)
        # text piece vecs -> text vec
        item_vec = torch.zeros(len(itemj_piece_len), self.d_model, dtype=torch.float32).cuda()
        piece_low = 0
        for i, l in enumerate(itemj_piece_len): # e.g. [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            piece_high = piece_low + l
            # # method a: mean pool
            # text_vec = torch.mean(piece_itemj_represent[piece_low:piece_high], dim=0, keepdim=False)
            # method b: attention merge
            text_vecs = piece_itemj_represent[piece_low:piece_high]
            a = nn.functional.softmax(torch.mm(self.ws02, torch.mm(self.Ws01, text_vecs.transpose(0,1))),
                                      dim=1)
            text_vec = torch.mm(a, text_vecs)[0]
            # # method c: simple weight
            # if l == 1:
            #     text_vec = piece_itemj_represent[piece_low]
            # elif l == 2:
            #     text_vecs = piece_itemj_represent[piece_low:piece_high]
            #     text_vec = torch.mm(self.sw, text_vecs)[0]
            # else:
            #     print('error!!!')

            item_vec[i] = text_vec
            piece_low = piece_high
        return item_vec

if __name__ == '__main__':
    # args = {"model_name_or_path": "bert-base-chinese/",
    #         "config_name": "bert-base-chinese/",
    #         "tokenizer_name": "bert-base-chinese/"
    #         }
    #
    # config = BertConfig.from_pretrained(
    #     args["config_name"],
    #     finetuning_task="",
    #     cache_dir=None,
    # )
    # tokenizer = BertTokenizer.from_pretrained(
    #     args["tokenizer_name"],
    #     do_lower_case=True,
    #     cache_dir=None,
    # )
    # model_base = BertForPieceText.from_pretrained(
    #     args["model_name_or_path"],
    #     from_tf=bool(".ckpt" in args["model_name_or_path"]),
    #     config=config,
    #     cache_dir=None,
    # )
    # model_base.to("cuda")
    #
    # piece_text_list = data_utils.obtain_piece_text()
    # print('piece_text_list[0][0]:\n', piece_text_list[0][0])
    # print('piece_text_list[0][1]:\n', piece_text_list[0][1])
    # inputs = tokenizer(piece_text_list[0], padding=True, return_tensors='pt')
    #
    # for ids in inputs["input_ids"]:
    #     print(tokenizer.decode(ids))

    print('haha')