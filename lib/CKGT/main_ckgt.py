import torch.optim as optim
import torch.utils.data as data

from utility.batch_test import *
from model.CKGT_method1 import CKGT
from utility.helper import  *
# import utility.load_data as load_data

import os
from time import time
import sys
import gc
import scipy.io as sio

print('torch version:', torch.__version__, ' cuda:', torch.cuda.is_available())

if __name__ == '__main__':
    print('All required package has loaded.\n')
    os.environ['PYTHONHASHSEED'] = str(2023) # 禁止hash随机化
    torch.cuda.manual_seed(2023)
    np.random.seed(2023)
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info('CKGT in MOOCCubeX. Piece emb is trained. Correcte self.batch_size_kg to int(args.batch_size_kg / 1.5). A10(24G). seed=2023')
    logging.info(args)

    def load_pretrain_data(args):
        pre_model = 'mf'
        pretrain_path = '%spretrain/%s_0703.npz' % (args.proj_path, pre_model)
        try:
            pretrain_data = np.load(pretrain_path)
            print('load the pretrained bprmf embeddings in ', pretrain_path)
        except Exception:
            pretrain_data = None
            print('Error to load pretrained bprmf embeddings!')
        return pretrain_data

    """
    ************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities

    if args.model_type in ['kgat']:
        "Load the laplacian matrix."
        config['A_in'] = sum(data_generator.lap_list)  # (184166, 184166)

        "Load the KG triplets"
        config['all_h_list'] = data_generator.all_h_list
        config['all_r_list'] = data_generator.all_r_list
        config['all_t_list'] = data_generator.all_t_list
        config['all_v_list'] = data_generator.all_v_list

    ui_texts_latent = sio.loadmat('../../Data_mooc/ui_texts_latent.mat')
    config['u_texts_embed'] = torch.from_numpy(ui_texts_latent['users_latent']).float()
    config['i_texts_embed'] = torch.from_numpy(ui_texts_latent['items_latent']).float()
    print('config utextembed:', config['u_texts_embed'])

    """
    Use the pretrained data to initialize the embeddings.
    """
    if args.pretrain == 1: # Pretrain with the learned embeddings. Default set.
        pretrain_data = load_pretrain_data(args)
    else:
        pretrain_data = None


    """
    ****************************************************
    Select one of the models.
    """
    if args.model_type in ['kgat']:
        model = CKGC(data_config=config, pretrain_data=pretrain_data, args=args)
        print('model type:', model.model_type)

    if args.pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr / 2) # args.lr

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    """
    ****************************************************
    Train.
    """
    print('\nbegin train...')
    epoch_list = []
    recall_list = []

    for epoch in range(1, args.epoch + 1): # 1      
        model.train()
      
        gc.collect()
        torch.cuda.empty_cache()

        loss, cf_total_loss, kg_total_loss = 0, 0, 0
        n_batch = data_generator.n_train // args.batch_size + 1
        
        # train_dataset.ng_sample(block_size=8)  # according to torch.utils DataLoader

        """
        ************************************************
        Alternative Training for KGAT:
        ... phase 1: to train the recommender.
        """

        time1 = time()
        print('total batch:', n_batch//args.items_per_user)
        for idx in range(n_batch//args.items_per_user): #n_batch//args.items_per_user
            users, pos_items, neg_items = data_generator.generate_train_cf_batch(args.items_per_user) # (bs_users, bs_pos_i, bs_neg_i)

            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)  # use or not use

            cf_loss = model(users, pos_items, neg_items, phase='cf')
            if np.isnan(cf_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d}/{:04d} Loss is nan.'.format(epoch, idx+1, n_batch))
                sys.exit()
            # if idx >= 0:
            #     break   

            cf_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_loss.item()
            # 清理显存
            del cf_loss
            
        print('Phase I consume:{}s'.format(time() - time1))

        """
        ************************************************
        Alternative Training for KGAT:
        ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
        """

        time2 = time()
        # print('data_generator.all_h_list:', len(data_generator.all_h_list))
        n_A_batch = len(data_generator.all_h_list) // args.batch_size_kg + 1
        for idx in range(n_A_batch):
            heads, relations, pos_tails, neg_tails = data_generator._generate_train_A_batch()
            heads, relations, pos_tails, neg_tails = \
                heads.to(device), relations.to(device), pos_tails.to(device), neg_tails.to(device)
            kg_loss = model(heads, relations, pos_tails, neg_tails, phase='kge')
            if np.isnan(kg_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d}/{:04d} Loss is nan.'.format(epoch, idx, n_A_batch))
                sys.exit()
            # if idx >= 0:
            #     break

            kg_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_loss
            # 清理显存
            del kg_loss
        print('Phase II consume:{}s'.format(time() - time2))

        # update A
        time3 = time()
        model.kgat.update_attention_A()
        logging.info('Epoch {:04d} | Train [{:.1f}s], Loss = {:.4f} + {:.4f}'
                     .format(epoch, time() - time1, cf_total_loss / (n_batch//args.items_per_user), kg_total_loss / n_A_batch))

        """
        *********************************************************
        Test.
        """
        if (epoch % args.evaluate_every) == 0 or epoch == args.epoch:
            model.eval()
            time4 = time()
            users_to_test = list(data_generator.test_user_dict.keys())
            # users_to_test = users_to_test[:1000]

            with torch.no_grad():
                result = test(model, users_to_test)
            logging.info('Epoch {:04d} | Test [{:.1f}s], recall=[{:.4f}, {:.4f}], ndcg=[{:.4f}, {:.4f}], precision=[{:.4f}, {:.4f}]'
                         .format(epoch, time() - time4, result['recall'][0], result['recall'][1],
                                 result['ndcg'][0], result['ndcg'][1], result['precision'][0], result['precision'][1]))

            epoch_list.append(epoch)
            recall_list.append(result['recall'][0])
            best_recall, should_stop = early_stopping(recall_list, args.stopping_steps)

            if should_stop:  # 跳出训练
                break

            if recall_list.index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    logging.info('Best Epoch={:04d} | recall={:.4f}'.format(best_epoch, best_recall))
    
    # 关机
    os.system('shutdown')























