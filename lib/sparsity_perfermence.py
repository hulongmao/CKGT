'''
加载最佳模型，测试稀疏度性能
'''
from utility.batch_test import *
from model.CKGC_method1 import CKGC
from utility.parsers import *
import torch
from utility.helper import *
import scipy.io as sio


def get_group_sparsity_prefermence():
    args = parse_args()
    os.environ['PYTHONHASHSEED'] = str(2024) # 禁止hash随机化
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    ************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities
    config['A_in'] = sum(data_generator.lap_list)
    config['all_h_list'] = data_generator.all_h_list
    config['all_r_list'] = data_generator.all_r_list
    config['all_t_list'] = data_generator.all_t_list
    config['all_v_list'] = data_generator.all_v_list
    ui_texts_latent = sio.loadmat('../../Data30/ui_texts_latent.mat')
    config['u_texts_embed'] = torch.from_numpy(ui_texts_latent['users_latent']).float()
    config['i_texts_embed'] = torch.from_numpy(ui_texts_latent['items_latent']).float()

    # mooc30: trained_model/model_epoch436.pth
    best_model = 'trained_model/model_epoch436.pth'
    model = CKGC(data_config=config, pretrain_data=None, args=args)
    model = load_model(model, best_model)
    model.to(device)

    split_uids, split_states = data_generator.get_sparsity_split()

    print('\nBegin to test sparsity...')
    for split_user_to_test, split_state in zip(split_uids, split_states):
        print(split_state)
        with torch.no_grad():
            result = test(model, split_user_to_test)
        print('recall@20:{:.4f}, ndcg@20:{:.4f}'.format(result['recall'][0], result['ndcg'][0]))
        print()

    print('\nBegin to test total...')
    users_to_test = list(data_generator.test_user_dict.keys())
    with torch.no_grad():
        result = test(model, users_to_test)
    print('recall@20:{:.4f}, ndcg@20:{:.4f}'.format(result['recall'][0], result['ndcg'][0]))
    print()

if __name__ == '__main__':
    get_group_sparsity_prefermence()





