'''
基于bert微调的课程文本推荐
(1)每个user由交互的图书描述文本表示，每个item由图书描述文本（前510个字符）表示
(2)微调的bert类将图书文本转换为向量
(3)利用用注意力机制将图书文本向量融合成user向量
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
        args = {"model_name_or_path": "bge-small-en-v1.5/",
                "config_name": "bge-small-en-v1.5/",
                "tokenizer_name": "bge-small-en-v1.5/"
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
            if 'layer.10' in name:  # 保留layer10之后的层, for bge-small-en-v1.5
                param.requires_grad = True
                break

        # bge-small-zh-v1.5: self.d_model = 512,  bge-small-en-v1.5: self.d_model = 384
        self.d_model = 384

        # method 3: use attention to intergrate vector of user by items, inspiration comes from "A Structured Self-Attentive Sentence Embedding"
        # original: a = softmax (ws2tanh (Ws1HT)), here: a = softmax(ws2tanh(Ws1(BT)), and B is vector of user by items
        self.da = 256
        self.Ws1 = nn.Parameter(torch.Tensor(self.da, self.d_model))
        self.ws2 = nn.Parameter(torch.Tensor(1, self.da))
        nn.init.normal_(self.Ws1, std=0.01)
        nn.init.normal_(self.ws2, std=0.01)


        # factor layer
        self.user_trans = nn.Linear(self.d_model, factor_num)  # bert-base-chinese:768
        self.item_trans = nn.Linear(self.d_model, factor_num)  # bert-base-chinese:768
        nn.init.normal_(self.user_trans.weight.data, std=0.01)
        nn.init.normal_(self.item_trans.weight.data, std=0.01)

    def forward(self, user_inputs=None, bs=None, user_pos=None, user_bytext_unique=None,
                itemi_pos=None, itemj_input=None, is_training=True):

        if is_training:
            user_pooled_out = self.bert_for_piece(**user_inputs)  # (bs_uni, self.d_model)
            # print('uni user_pooled_out:', user_pooled_out.shape)
            # print('uni user_pooled first column:', user_pooled_out[:, 0])
            # print('uni user_pooled for user 0:', torch.mean(user_pooled_out[:15, 0]))
            # print('uni user_pooled for user 1:', torch.mean(user_pooled_out[15:, 0]))
            # print(user_pooled_out[:, 0])
            # print('bs:', bs, 'user_pos:', user_pos)
            # print('user_pos.keys():', list(user_pos.keys()))

            user_represent = self.merge_user_represent(bs, user_pos, user_bytext_unique, user_pooled_out.clone())
            itemi_represent = user_pooled_out.clone()[itemi_pos:itemi_pos+bs]
            itemj_represent = self.bert_for_piece(**itemj_input)

            user_latent = self.user_trans(user_represent)
            i_latent = self.item_trans(itemi_represent)
            j_latent = self.item_trans(itemj_represent)

            prediction_i = (user_latent * i_latent).sum(dim=-1)
            prediction_j = (user_latent * j_latent).sum(dim=-1)
        else:
            user_latent, prediction_i, prediction_j = None, None, None
            itemi_represent = None
            itemj_represent = self.bert_for_piece(**itemj_input)
        return itemi_represent, itemj_represent, prediction_i, prediction_j

    def merge_user_represent(self, bs, user_pos, user_bytext_unique, flat_user_vec):
        user_represent = torch.zeros(bs, self.d_model, dtype=torch.float32).cuda()
        user_low = 0
        for (_, pos), user_uni in zip(user_pos.items(), user_bytext_unique):
            user_high = user_low + len(user_uni)

            # method 3: use attention to intergrate vector of user by items
            one_user_vecs = flat_user_vec[user_low:user_high]  # (n, d_model)
            a = nn.functional.softmax(torch.mm(self.ws2, torch.mm(self.Ws1, one_user_vecs.transpose(0,1))),
                                      dim=1)
            one_user_vec = torch.mm(a, one_user_vecs)[0]
            user_represent[pos] = one_user_vec
            user_low = user_high

        return user_represent

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