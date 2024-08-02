import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', default='MUTAG',help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const',
        const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const',
        const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const',
        const=True, default=False)
    parser.add_argument('--dropout_rate', dest='dropout_rate', type=float, default=0.1,
                        help='Dropout rate.')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0005,
        help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--lamda1', dest='lamda1', type=float, default=0.79475*0.5,
        help='lamda1') # The value range is set to [0,1]
    
    parser.add_argument('--info_ent_threshold', dest='info_ent_threshold', type=float, default=4.24972,
        help='info_ent_threshold') # The value range is [0,log_2(cluster_num)]
    parser.add_argument('--pos_div_threshold', dest='pos_div_threshold', type=float, default=0.879894,
        help='pos_div_threshold') # When using JS divergence, the value range is [0,1]
    parser.add_argument('--neg_div_threshold', dest='neg_div_threshold', type=float, default=0.8,
        help='neg_div_threshold') # When using JS divergence, the value range is [0,1]
    
    parser.add_argument('--reliab_pacing_type', type=str, default='logar')
    parser.add_argument('--pos_reliab_pacing_type', type=str, default='logar')
    parser.add_argument('--neg_reliab_pacing_type', type=str, default='logar')

    parser.add_argument('--bs', dest='batch_size', type=int, default=32,
                        help='batch_size')

    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
        help='')

    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--stro_aug', type=str, default='stro_dnodes')
    # parser.add_argument('--weak_aug2', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--use_momentum', type=bool, default=True)

    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers (default: 0)')
    # PCL args
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    # parser.add_argument("--feat_dim", default=128, type=int,
    #                     help="feature dimension")
    parser.add_argument("--nmb_prototypes", default=10, type=int,
                        help="number of prototypes")

    parser.add_argument('--sp_metric', type=str, default='HybridRatio', 
                        help="the metric used in curriculum learning: Compare, EntRatio, EpochRatio, HybridRatio")
    # parser.add_argument('--cand_set', type=str, default='tri_req', 
    #                     help="the number of candidate set requirements: one_req, tri_req")


    return parser.parse_args()

