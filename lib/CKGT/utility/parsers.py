'''
Create on Jun 18, 2024
Pytorch Implementation of CKGC model
@author: Longmao Hu
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run KGAT.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../../Data_mooc',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    parser.add_argument('--dataset', nargs='?', default='', # yelp2018
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0, # 0
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=800, # 30, 100
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='CF Embedding size.')
    parser.add_argument('--kge_size', type=int, default=64,
                        help='KG Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,32]',
                        help='Output sizes of every layer')
    parser.add_argument('--factor_size', type=int, default=128,
                        help='HTN implicit vector size of user and item.')

    parser.add_argument('--text_limit', type=int, default=20,
                        help='HTN max text count per user.')
    parser.add_argument('--items_per_user', type=int, default=5,
                        help='once sample items per user.')

    parser.add_argument('--batch_size', type=int, default=1536,  # 2048 in kgat
                        help='CF batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=4096, # 4096 in kgat
                        help='KG batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='kgat', # kgat
                        help='Specify a loss type from {kgat, bprmf, fm, nfm, cke, cfkg}.')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--alg_type', nargs='?', default='bi', # ngcf
                        help='Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout', type=float, default=0.1,
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40]',  # [20, 40, 60, 80, 100]
                        help='Output sizes of every layer')

    parser.add_argument('--evaluate_every', type=int, default=2,  # 10
                        help='Epoch interval.')
    parser.add_argument('--stopping_steps', type=int, default=10,  # 10
                        help='Number of epoch for early stopping')

    parser.add_argument('--save_flag', type=int, default=1, # 0
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--use_att', type=bool, default=True,
                        help='whether using attention mechanism')
    parser.add_argument('--use_kge', type=bool, default=True,
                        help='whether using knowledge graph embedding')
    
    parser.add_argument('--l1_flag', type=bool, default=True,
                        help='Flase: using the L2 norm, True: using the L1 norm.')

    parser.add_argument('--accumulator', type=int, default=4,
                        help='The accumulate number of gradients in backward propagation.')

    args = parser.parse_args()

    save_dir = 'trained_model/CKCG/embed-dim{}_kge-dim{}_{}_{}_htn-implicit-dim{}_lr{}_pretrain{}/'.format(
        args.embed_size, args.kge_size,
        args.alg_type, '-'.join([str(i) for i in eval(args.layer_size)]),
        args.factor_size, args.lr, args.pretrain)
    args.save_dir = save_dir

    return args
