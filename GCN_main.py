import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch_geometric.utils import dense_to_sparse, negative_sampling
from model import GCN_basic, GCN_gravity
from test_file import GCN_gravity_1
from baseline import Deep_gravity
from utils import dotdict
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

def main(args):
    # 模型初始化
    if args.method == "GCN_basic":
        model = GCN_basic(args)
    elif args.method == "GCN_gravity":
        model = GCN_gravity(args)
    else:
        raise Exception("Error model name")
    # elif args.method == "GCN_MC":
    #     model = GCN_MC(args)

    # 训练模型

    # model.predict_no_negative()

    if args.topology == "functional connectivity":
        model.train()
        model.test()
    elif args.topology == "complete graph":
        model.train_no_negative()
        model.test_no_negative()
    elif args.topology == "distance-based":
        model.train_dis()
        model.test_dis()
    # model.predict()

    return

if __name__== "__main__":

    args = dotdict()
    args.method = "GCN_gravity"
    args.data_path = "data/model/"
    args.city = "Xian"
    # Guangzhou loss coefficient 0.2 0.8
    # Model parameter
    args.node_feature = 3
    args.link_feature = 3
    args.gcn_hidden_feature = 16
    args.layer_num = 5
    args.alpha = 1
    args.beta = 1
    args.topology = "functional connectivity" # ["functional connectivity", "complete graph", "distance-based"]
    args.max_distance = 27

    if torch.backends.mps.is_available():
        args.device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("Using cuda")
    else:
        args.device = None
        print("device not available, using cpu")


    # Training parameter
    args.epochs = 50
    args.lr = 0.01

    args.optimizer = "Adam"
    args.batch_size = 64
    args.train_percentage = 0.6
    args.val_percentage = 0.2
    args.search_neighbour = [-1, -1]
    args.save_path = "trained model/"
    args.scale = True
    args.patience = 10
    args.dropout = 0.1
    if args.scale:
        args.scale_method = "log"
    print("Training", args.method, " layer_num", args.layer_num, " dimension", args.gcn_hidden_feature)
    main(args)



