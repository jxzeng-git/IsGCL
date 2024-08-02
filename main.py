import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch_geometric.data import DataLoader

import json
import time
from numpy import exp
import scipy.stats
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from aug import TUDataset_aug as TUDataset
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
import model  
from util_spgcl import *

import argparse
import pdb

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    # ======================= common parameters ==========================
    parser.add_argument('--DS', dest='DS', default='MUTAG',help='Dataset')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate.') #{0.01， 0.005， 0.001， 0.0005， 0.0001}
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run') 
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=256, help='batch_size') #{128， 256， 512}
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=32, 
        help='')
    # parser.add_argument('--dropout_rate', dest='dropout_rate', type=float, default=0.1, 
    #     help='Dropout rate.')

    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--num_workers', type=int, default=8, 
        help='number of workers (default: 0)')
    
    # ======================= parameters for GCL ==========================
    parser.add_argument('--num_gc_layers', dest='num_gc_layers', type=int, default=5, 
        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--aug', type=str, default='dnodes') 
    parser.add_argument('--stro_aug', type=str, default='stro_dnodes') 
    # parser.add_argument('--weak_aug2', type=str, default=None)

    # ======================= our specified parameters ==========================
    parser.add_argument('--pacing_type', type=str, default='logar') 
    parser.add_argument('--beta', dest='beta', type=float, default=0.79475*0.5, help='lambda1') # Set the value range to [0,1]
    parser.add_argument("--num_prot", default=10, type=int, help="number of prototypes")
    # parser.add_argument('--sp_metric', type=str, default='EpochRatio', 
    #     help="the metric used in curriculum learning: Compare, EntRatio, EpochRatio, HybridRatio")
    # parser.add_argument('--info_ent_threshold', dest='info_ent_threshold', type=float, default=4.24972,
    #     help='info_ent_threshold') # The value range is [0,log_2(prot_num)]
    parser.add_argument('--pos_div_threshold', dest='pos_div_threshold', type=float, default=0.01,
        help='pos_div_threshold') # When using JS divergence, the value range is [0,1]
    parser.add_argument('--neg_div_threshold', dest='neg_div_threshold', type=float, default=0.2,
        help='neg_div_threshold') # When using JS divergence, the value range is [0,1]
    # parser.add_argument('--cand_set', type=str, default='tri_req', 
    #                     help="the number of candidate set requirements: one_req, tri_req")


    return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()

    args = arg_parse()
    setup_seed(args.seed)

    log_interval = 1
    accuracies = {'val':[], 'test':[]}
    epochs = args.epochs
    batch_size = args.batch_size
    # lr = args.lr
    DS = args.DS
    prototype_num = args.num_prot
    beta = args.beta
    # print('================')
    # print('lr: {}'.format(args.lr))
    # print('epochs: {}'.format(epochs))
    # print('batch_size: {}'.format(args.batch_size))
    # print('hidden_dim: {}'.format(args.hidden_dim))
    # print('num_gc_layers: {}'.format(args.num_gc_layers))
    # print("num_prot:{}".format(args.num_prot))
    print('beta: {}'.format(args.beta))
    print('pos_div_threshold: {}'.format(args.pos_div_threshold))
    print('neg_div_threshold: {}'.format(args.neg_div_threshold))
    # print('info_ent_threshold: {}'.format(args.info_ent_threshold))
    # print('sp_metric: {}'.format(args.sp_metric))
    # print('================')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    dataset = TUDataset(path, name=DS, aug=args.aug, stro_aug=args.stro_aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none', stro_aug='none').shuffle()

    #******************
    # batch_size = len(dataset)
    # print(f"batch_size={batch_size}")
    # print(f"num_feature={dataset.get_num_feature()}")
    #******************

    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.spgcl(dataset_num_features, args.hidden_dim, args.num_gc_layers, num_prot=args.num_prot).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # model.eval()
    # init_emb, y = model.encoder.get_embeddings(dataloader_eval)

    train_time = 0
    for epoch in range(1, epochs+1):
        # batch_time = AverageMeter()
        # data_time = AverageMeter()
        # losses = AverageMeter()

        epoch_loss = 0
        model.train()
        end = time.time()
        epoch_time = 0
        for it, data in enumerate(dataloader):
            time1 = time.time()
            # data_time.update(time.time() - end)
            data, data_aug, data_stro_aug = data
            optimizer.zero_grad()
            node_num, _ = data.x.size()
            data = data.to(device)
            bs = data.y.size(0)

            # time1 = time.time()
            embedding1, prot_scores1 = model(data.x, data.edge_index, data.batch, data.num_graphs)
            # print(f"embedding learning time: {time.time() - time1}")

            # time1 = time.time()
            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'ppr_aug' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4' or args.aug == 'dedge_nodes':
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
            # print(f"augmentation time: {time.time() - time1}")
            data_aug = data_aug.to(device)
 
            # time1 = time.time()
            embedding2, prot_scores2 = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
            # print(f"embedding learning time: {time.time() - time1}")

            # embedding = torch.cat((embedding1, _embedding))
            temp = 0.2
            # optimizer.zero_grad()

            batch_loss = model.loss(embedding1, embedding2, prot_scores1, prot_scores2, beta, args.pacing_type, args.pos_div_threshold, args.neg_div_threshold, epoch, epochs)

            epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            epoch_time += time.time() - time1
            
        if epoch % log_interval == 0:
           
            model.eval()
            
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            
            # time1 = time.time()
            acc_val, acc = evaluate_embedding(emb, y)
            # print(f"evaluation time: {time.time() - time1}")
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            print('==========Epoch {}, Loss {}, acc {}'.format(epoch, epoch_loss / len(dataloader), acc))
        train_time += epoch_time

    print(f"training need {train_time} seconds")
    print(f"total need {time.time() - start_time} seconds")