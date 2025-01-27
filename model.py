import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import SAGEConv, EdgeConv, MessagePassing, GCNConv
from torch_geometric.utils import dense_to_sparse, negative_sampling, add_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, DataLoader
from scipy.spatial import distance
from tqdm import tqdm
import pandas as pd
import os
import pickle
import numpy as np
from utils import MAE, MAPE, MSE, EarlyStopping, RMSE, CPC, JSD
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats, special
import math
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

K = 3
class EdgeOnlyConv(MessagePassing):
    def __init__(self, in_fea, hi):
        super(EdgeOnlyConv, self).__init__(aggr='mean')  # "mean" aggregation
        self.lin = torch.nn.Linear(in_fea, hi * 2)
        self.lin_update = torch.nn.Linear(hi * 2, hi)


    def forward(self, edge_index, edge_attr):
        # 添加自环
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0.0, num_nodes=edge_attr.size(0))

        # 传播（消息传递）
        return self.propagate(edge_index, edge_attr=edge_attr)

    def message(self, edge_attr):
        return self.lin(edge_attr)

    def update(self, aggr_out):
        return self.lin_update(aggr_out)


class GraphSAGE(torch.nn.Module):
    def __init__(self, node_in_channels, hidden_channels, out_channels, drop_out, layer_num):
        super(GraphSAGE, self).__init__()
        self.layer_num = layer_num
        self.layer_list = nn.ModuleList()

        self.layer_list.append(SAGEConv(node_in_channels, hidden_channels))

        for i in range(1, layer_num):
            self.layer_list.append(SAGEConv(hidden_channels, hidden_channels))

        self.drop_out = nn.Dropout(drop_out)
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for i in range(self.layer_num):
            x = self.layer_list[i](x, edge_index)

        row, col = edge_index
        a = torch.mul(x[row], x[col])
        a = self.drop_out(a)
        edge_feats1 = self.fc(a)
        return edge_feats1


class GraphSAGE_gravity(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels, drop_out, layer_num):
        super(GraphSAGE_gravity, self).__init__()
        self.layer_list = nn.ModuleList()

        self.layer_list.append(SAGEConv(node_in_channels, hidden_channels))

        for i in range(1, layer_num // 2):
            self.layer_list.append(SAGEConv(hidden_channels, hidden_channels))

        self.layer_list.append(SAGEConv(hidden_channels, hidden_channels // 2))

        for i in range(layer_num // 2 + 1, layer_num):
            self.layer_list.append(SAGEConv(hidden_channels // 2, hidden_channels // 2))

        self.layer_list_1 = nn.ModuleList()
        self.layer_list_1.append(EdgeOnlyConv(edge_in_channels, hidden_channels))
        for i in range(1, layer_num // 2):
            self.layer_list_1.append(EdgeOnlyConv(hidden_channels, hidden_channels))

        self.layer_list_1.append(EdgeOnlyConv(hidden_channels, hidden_channels // 2))

        for i in range(layer_num // 2 + 1, layer_num):
            self.layer_list_1.append(EdgeOnlyConv(hidden_channels // 2, hidden_channels // 2))

        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.drop_out = torch.nn.Dropout(p=drop_out)
        self.leaky_relu = nn.LeakyReLU()
        # self.attention = nn.MultiheadAttention(hidden_channels * 2)

    def forward(self, x, edge_index, dis):

        for layer in self.layer_list:
            x = layer(x, edge_index)
            x = self.leaky_relu(x)

        for layer in self.layer_list_1:
            dis = layer(edge_index, dis)
            dis = self.leaky_relu(dis)

        row, col = edge_index
        x = torch.mul(x[row], x[col])
        edge_feats = torch.cat([x, dis], dim=1)
        edge_feats = self.drop_out(edge_feats)
        x = self.fc(edge_feats)

        return x

class GCN_basic(nn.Module):
    def __init__(self, args):
        super(GCN_basic, self).__init__()
        self.gcn = GraphSAGE(args.node_feature, args.gcn_hidden_feature, 1, args.dropout, args.layer_num)
        total_params = sum(p.numel() for p in self.gcn.parameters() if p.requires_grad)
        print(f"模型的总可训练参数量: {total_params}")
        if args.device != None:
            self.device = args.device
            self.gcn = self.gcn.to(args.device)
        else:
            self.device = args.device
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.search_neighbour = args.search_neighbour

        self.train_loader, self.test_loader, self.val_loader, self.data = self.make_data_loader(args)
        self.optimizer = self.select_optimizer(args)
        self.loss = torch.nn.MSELoss()
        self.args = args
        self.early_stopping = EarlyStopping(patience=args.patience, min_delta=0, checkpoint_path=os.path.join(args.save_path, "best_checkpoint_basic_"+args.city+".pt"))

    def make_data_loader(self, args):
        N_filename = args.city + "_N_reduced.csv"
        N_data = pd.read_csv(os.path.join(args.data_path, N_filename), index_col=0)
        A_filename = args.city + "_A_reduced.csv"
        A_data = pd.read_csv(os.path.join(args.data_path, A_filename), index_col=0)
        E_filename = args.city + "_E_reduced.csv"
        E_data = pd.read_csv(os.path.join(args.data_path, E_filename), index_col=0)

        N_data = N_data.drop(N_data.columns[[0, 1, 2, 3, 7]], axis=1)

        self.A_data = A_data.values

        self.N_data = N_data.values
        num_nodes = self.N_data.shape[0]
        self.num_nodes = num_nodes

        if args.scale:
            if args.scale_method == "log":
                self.N_data = np.log10(self.N_data + 1)
            else:
                raise Exception("Unknown scaling method!")

        self.E_data = E_data.values
        # self.E_data = torch.where(self.E_data > 10000, torch.tensor(0), self.E_data)
        if args.scale:
            if args.scale_method == "log":
                self.E_data = np.log10(self.E_data + 1)
            else:
                raise Exception("Unknown scaling method!")

        edge_index, _ = dense_to_sparse(torch.tensor(self.A_data))
        num_nodes = self.N_data.shape[0]
        train_size = int(num_nodes * args.train_percentage)
        val_size = int(num_nodes * args.val_percentage)

        print(f"Total nodes: {num_nodes}")

        self.edge_target = [self.E_data[i, j] for i, j in zip(edge_index[0], edge_index[1])]
        self.edge_target = torch.tensor(self.edge_target).unsqueeze(-1)

        data = Data(x=self.N_data, edge_index=edge_index, edge_target=self.edge_target)

        train_idx = torch.arange(train_size)
        val_idx = torch.arange(train_size, train_size + val_size)
        test_idx = torch.arange(train_size + val_size, num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        train_loader = NeighborLoader(data, input_nodes=train_mask, num_neighbors=args.search_neighbour, batch_size=self.batch_size, shuffle=True)
        val_loader = NeighborLoader(data, input_nodes=val_mask, num_neighbors=args.search_neighbour, batch_size=self.batch_size, shuffle=False)
        test_loader = NeighborLoader(data, input_nodes=test_mask, num_neighbors=args.search_neighbour, batch_size=self.batch_size, shuffle=False)

        print("nodes to train", len(train_loader))
        print("nodes to validation", len(val_loader))
        print("nodes to test", len(test_loader))
        return train_loader, test_loader, val_loader, data


    def select_optimizer(self, args):
        if args.optimizer == "Adam":
            optimizer = Adam(list(self.gcn.parameters()), lr=self.lr)
        else:
            raise ValueError("not recognized optimizer")

        return optimizer

    def train(self):
        print("Start training")
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        for epoch in range(self.epochs):
            self.gcn.train()
            y_pred = []
            y_true = []
            for batch in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                if self.device != None:
                    batch.x = batch.x.float()
                    batch.edge_target = batch.edge_target.float()
                    batch = batch.to(self.device)

                x = batch.x  # 获取节点特征
                edge_index = batch.edge_index  # 获取边索引

                neg_edge_index = negative_sampling(
                    edge_index=batch.edge_index,  # 将边索引转换为无向图
                    num_nodes=batch.num_nodes,
                    num_neg_samples=edge_index.size(1)
                )

                out_pos = self.gcn(x, edge_index)
                out_neg = self.gcn(x, neg_edge_index)

                pos_target = batch.edge_target.float()  # 取出正样本边的目标值
                neg_target = torch.zeros([neg_edge_index.shape[1], 1]).float()

                if self.device != None:
                    neg_target = neg_target.to(self.device)
                # 计算损失
                pos_loss = self.loss(out_pos, pos_target)
                neg_loss = self.loss(out_neg, neg_target)

                loss = pos_loss * 0.2 + neg_loss * 0.8

                loss.backward()
                self.optimizer.step()

                y_pred.append(out_pos.cpu().detach().numpy())
                y_true.append(pos_target.cpu().detach().numpy())
                y_pred.append(out_neg.cpu().detach().numpy())
                y_true.append(neg_target.cpu().detach().numpy())


            print(f'Epoch {epoch + 1}/{self.epochs}, Pos Loss: {pos_loss:.8f}, Neg Loss: {neg_loss:.8f}')

            test_loss = self.eval()

            if self.early_stopping(test_loss, self.gcn):
                print(f'Early stopping at epoch {epoch + 1}')
                break

        return

    def eval(self):
        self.gcn.eval()
        true = []
        pred = []
        for batch in self.val_loader:

            if self.device != None:
                batch = batch.to(self.device)

            x = batch.x.float()  # 获取节点特征
            edge_index = batch.edge_index  # 获取边索引

            pos_edge_index = edge_index
            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index,  # 将边索引转换为无向图
                num_nodes=batch.num_nodes,
                num_neg_samples=edge_index.size(1)
            )

            pos_target = batch.edge_target.float().cpu() # 取出正样本边的目标值
            neg_target = torch.zeros([neg_edge_index.shape[1], 1]).float()

            out_pos = self.gcn(x, pos_edge_index)
            out_neg = self.gcn(x, neg_edge_index)

            out_pos = out_pos.cpu().detach().numpy()
            out_neg = out_neg.cpu().detach().numpy()
            if self.args.scale:
                if self.args.scale_method == "Minmaxscaler":
                    out_pos = self.E_data_scaler.inverse_transform(out_pos)
                elif self.args.scale_method == "log":
                    out_pos = np.power(10, out_pos) - 1
                    out_neg = np.power(10, out_neg) - 1
                    pos_target = np.power(10, pos_target) - 1
                    neg_target = np.power(10, neg_target) - 1
                elif self.args.scale_method == "boxcox":
                    out_pos_ = special.inv_boxcox1p(out_pos.flatten(), self.lam_best_E)
                    out_pos = out_pos_.reshape(out_pos.shape)
            pred.append(out_pos)
            pred.append(out_neg)
            true.append(pos_target)
            true.append(neg_target)

        true = np.concatenate(true, axis=0)
        pred = np.concatenate(pred, axis=0)
        mae = MAE(true, pred)
        mse = MSE(true, pred)

        print("val loss mae:", mae)
        return mse

    def test(self):
        self.gcn.load_state_dict(torch.load(os.path.join(self.args.save_path, "best_checkpoint_basic_"+self.args.city+".pt"), weights_only=True))
        self.gcn.eval()
        print("start testing")
        true = []
        pred = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                if self.device != None:
                    batch = batch.to(self.device)
                x = batch.x.float()  # 获取节点特征
                edge_index = batch.edge_index  # 获取边索引

                pos_edge_index = edge_index
                neg_edge_index = negative_sampling(
                    edge_index=batch.edge_index,  # 将边索引转换为无向图
                    num_nodes=batch.num_nodes,
                    num_neg_samples=edge_index.size(1)
                )

                pos_target = batch.edge_target.float().cpu()  # 取出正样本边的目标值
                neg_target = torch.zeros([neg_edge_index.shape[1], 1]).float()

                out_pos = self.gcn(x, pos_edge_index)
                out_neg = self.gcn(x, neg_edge_index)

                out_pos = out_pos.cpu().detach().numpy()
                out_neg = out_neg.cpu().detach().numpy()
                if self.args.scale:
                    if self.args.scale_method == "Minmaxscaler":
                        out_pos = self.E_data_scaler.inverse_transform(out_pos)
                    elif self.args.scale_method == "log":
                        out_pos = np.power(10, out_pos) - 1
                        out_neg = np.power(10, out_neg) - 1
                        pos_target = np.power(10, pos_target) - 1
                        neg_target = np.power(10, neg_target) - 1
                    elif self.args.scale_method == "boxcox":
                        out_pos_ = special.inv_boxcox1p(out_pos.flatten(), self.lam_best_E)
                        out_pos = out_pos_.reshape(out_pos.shape)
                pred.append(out_pos)
                pred.append(out_neg)
                true.append(pos_target)
                true.append(neg_target)

        true = np.concatenate(true, axis=0).flatten()
        pred = np.concatenate(pred, axis=0).flatten()

        mae = MAE(true, pred)
        rmse = RMSE(true, pred)

        true = np.log10(true + 1)
        pred = np.log10(pred + 1)

        mape = MAPE(true, pred)
        cpc = CPC(true, pred)

        if np.all(true == 0):
            true = np.ones_like(true) * 1e-10
        if np.all(pred == 0):
            pred = np.ones_like(pred) * 1e-10

        jsd = JSD(true, pred)

        print("test loss rmse:", rmse, "mae:", mae, "mape:", mape, "CPC:", cpc, "JSD:", jsd)
        return rmse


def sample_negative_edges(data, num_neg_samples=None):
    # 使用负采样函数生成负样本
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,  # 基于正样本边进行采样
        num_nodes=data.num_nodes,
        num_neg_samples=num_neg_samples
    )
    return neg_edge_index

class GCN_gravity(nn.Module):
    def __init__(self, args):
        super(GCN_gravity, self).__init__()
        self.gcn = GraphSAGE_gravity(args.node_feature, args.link_feature, args.gcn_hidden_feature, 1, args.dropout, args.layer_num).float()

        total_params = sum(p.numel() for p in self.gcn.parameters() if p.requires_grad)
        print(f"模型的总可训练参数量: {total_params}")
        if args.device != None:
            self.device = args.device
            self.gcn = self.gcn.to(args.device)
        else:
            self.device = args.device
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.search_neighbour = args.search_neighbour

        self.train_loader, self.test_loader, self.val_loader, self.data = self.make_data_loader(args)
        self.optimizer = self.select_optimizer(args)
        self.loss = torch.nn.MSELoss()
        self.args = args
        self.early_stopping = EarlyStopping(patience=args.patience, min_delta=0, checkpoint_path=os.path.join(args.save_path, "best_checkpoint_"+args.city+".pt"))


    def make_data_loader(self, args):
        N_filename = args.city + "_N_reduced.csv"
        N_data = pd.read_csv(os.path.join(args.data_path, N_filename), index_col=0)
        N_filename_all = args.city + "_N.csv"
        N_data_all = pd.read_csv(os.path.join(args.data_path, N_filename_all), index_col=0)
        A_filename = args.city + "_A_reduced.csv"
        A_data = pd.read_csv(os.path.join(args.data_path, A_filename), index_col=0)
        E_filename = args.city + "_E_reduced.csv"
        E_data = pd.read_csv(os.path.join(args.data_path, E_filename), index_col=0)
        self.N_data_ = N_data.reset_index(drop=True)
        self.N_data_all = N_data_all
        N_data = N_data.drop(N_data.columns[[0, 1, 2, 3, 7]], axis=1)
        A_data = A_data.values
        
        center_dis_data = pd.read_csv(os.path.join(args.data_path, args.city + "_center_dis_reduced.csv"), index_col=0)

        # 计算邻接矩阵
        # distance_matrix = center_dis_data.copy().values
        # np.fill_diagonal(distance_matrix, 0)
        # sigma = 100000  # 距离归一化因子
        # epsilon = np.exp(-((args.max_distance * 1000) ** 2) / (sigma ** 2))  # 阈值
        #
        # A_data = np.exp(-(distance_matrix ** 2) / (sigma ** 2))
        # A_data[A_data < epsilon] = 0  # 应用阈值条件

        if self.args.topology == "functional connectivity":
            self.A_data = A_data
        elif self.args.topology == "complete graph":
            self.A_data == np.ones_like(A_data) - np.eye(A_data.shape[0])
        elif self.args.topology == "distance-based":
            distance_matrix = center_dis_data.copy().values
            np.fill_diagonal(distance_matrix, 0)
            sigma = 100000  # 距离归一化因子
            epsilon = np.exp(-((args.max_distance * 1000) ** 2) / (sigma ** 2))  # 阈值
            A_data = np.exp(-(distance_matrix ** 2) / (sigma ** 2))
            A_data[A_data < epsilon] = 0  # 应用阈值条件
            self.A_data = A_data
        
        self.N_data = N_data.values
        num_nodes = self.N_data.shape[0]
        self.num_nodes = num_nodes

        edge_index, _ = dense_to_sparse(torch.tensor(self.A_data))

        
        underground_dis_data = pd.read_csv(os.path.join(args.data_path,args.city + "_underground_dis_reduced.csv"), index_col=0)
        road_dis_data = pd.read_csv(os.path.join(args.data_path,args.city + "_road_dis_reduced.csv"), index_col=0)

        center_dis_data = center_dis_data.values
        underground_dis_data = underground_dis_data.values
        road_dis_data = road_dis_data.values

        edge_attr_center = [center_dis_data[i, j] for i, j in zip(edge_index[0], edge_index[1])]
        edge_attr_underground = [underground_dis_data[i, j] for i, j in zip(edge_index[0], edge_index[1])]
        edge_attr_road = [road_dis_data[i, j] for i, j in zip(edge_index[0], edge_index[1])]

        edge_attr = np.stack([edge_attr_center, edge_attr_underground, edge_attr_road], axis=1)
        edge_attr = torch.tensor(edge_attr)
        self.edge_attr = edge_attr

        all_edge_attr = np.stack([center_dis_data, underground_dis_data, road_dis_data], axis=2)
        all_edge_attr = np.log10(all_edge_attr + 1)
        all_edge_attr = torch.tensor(all_edge_attr).float()
        self.all_edge_attr = all_edge_attr

        if self.device != None:
            self.all_edge_attr = self.all_edge_attr.to(self.device)

        if args.scale:
            if args.scale_method == "log":
                self.N_data = np.log10(self.N_data + 1)
                self.edge_attr = np.log10(self.edge_attr + 1)
            else:
                raise Exception("Unknown scaling method!")


        self.E_data = E_data.values
        if args.scale:
            if args.scale_method == "log":
                self.E_data = np.log10(self.E_data + 1)
            else:
                raise Exception("Unknown scaling method!")

        train_size = int(num_nodes * args.train_percentage)
        val_size = int(num_nodes * args.val_percentage)

        print(f"Total nodes: {num_nodes}")

        self.edge_target = [self.E_data[i, j] for i, j in zip(edge_index[0], edge_index[1])]
        self.edge_target = torch.tensor(self.edge_target).unsqueeze(-1)

        data = Data(x=self.N_data, edge_index=edge_index, edge_attr=self.edge_attr, edge_target=self.edge_target)

        train_idx = torch.arange(train_size)
        val_idx = torch.arange(train_size, train_size + val_size)
        test_idx = torch.arange(train_size + val_size, num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        # 创建训练集子图

        train_loader = NeighborLoader(data, input_nodes=train_mask, num_neighbors=args.search_neighbour, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = NeighborLoader(data, input_nodes=val_mask, num_neighbors=args.search_neighbour, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = NeighborLoader(data, input_nodes=test_mask, num_neighbors=args.search_neighbour, batch_size=self.batch_size, shuffle=False, num_workers=0)

        pred_data = Data(x=self.N_data, edge_index=edge_index, edge_attr=self.edge_attr, edge_target=self.edge_target)
        pred_loader = DataLoader([pred_data], batch_size=1, shuffle=False)

        self.pred_loader = pred_loader
        print("nodes to train", len(train_loader))
        print("nodes to validation", len(val_loader))
        print("nodes to test", len(test_loader))
        return train_loader, test_loader, val_loader, data

    def select_optimizer(self, args):
        if args.optimizer == "Adam":
            optimizer = Adam(list(self.gcn.parameters()), lr=self.lr)
        else:
            raise ValueError("not recognized optimizer")

        return optimizer

    def train_no_negative(self):
        print("Start training")
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        for epoch in range(self.epochs):
            self.gcn.train()
            for batch in tqdm(self.train_loader):
                    self.optimizer.zero_grad()
                    if self.device != None:
                        batch.x = batch.x.float()
                        batch.edge_attr = batch.edge_attr.float()
                        batch.edge_target = batch.edge_target.float()
                        batch = batch.to(self.device)
    
                    x = batch.x  # 获取节点特征
                    pos_edge_index = batch.edge_index
    
                    edge_attr = batch.edge_attr
                    pos_target = batch.edge_target.float()

                    out_pos = self.gcn(x, pos_edge_index, edge_attr)
                    loss = self.loss(out_pos, pos_target)
    
                    loss.backward()

                    self.optimizer.step()
            torch.cuda.empty_cache()
            
            print(f'Epoch {epoch + 1}/{self.epochs},Pos Loss: {loss:.8f}')

            test_loss = self.eval_no_negative()

            if self.early_stopping(test_loss, self.gcn):
                print(f'Early stopping at epoch {epoch + 1}')
                break
        return
    
    def eval_no_negative(self):
        self.gcn.eval()
        true = []
        pred = []
        with torch.no_grad():
            for batch in self.val_loader:

                if self.device != None:
                    batch = batch.to(self.device)

                x = batch.x.float()
                pos_edge_index = batch.edge_index


                edge_attr = batch.edge_attr.float()

                pos_target = batch.edge_target.float()

                out_pos = self.gcn(x, pos_edge_index, edge_attr)

                torch.cuda.empty_cache()

                out_pos = out_pos.cpu().numpy()

                pos_target = pos_target.cpu().numpy()
                if self.args.scale:
                    if self.args.scale_method == "Minmaxscaler":
                        out_pos = self.E_data_scaler.inverse_transform(out_pos)
                    elif self.args.scale_method == "log":
                        out_pos = np.power(10, out_pos) - 1

                        pos_target = np.power(10, pos_target) - 1


                    elif self.args.scale_method == "boxcox":
                        out_pos_ = special.inv_boxcox1p(out_pos.flatten(), self.lam_best_E)
                        out_pos = out_pos_.reshape(out_pos.shape)
                pred.append(out_pos)
                true.append(pos_target)

        true = np.concatenate(true, axis=0)
        pred = np.concatenate(pred, axis=0)
        mae = MAE(true, pred)
        mse = MSE(true, pred)
        mape = MAPE(true, pred)
        print("val loss mae:", mae, "mape:", mape)
        return mse
    
    def test_no_negative(self):
        self.gcn.load_state_dict(
            torch.load(os.path.join(self.args.save_path, "best_checkpoint_" + self.args.city + ".pt"),
                       weights_only=True))
        if self.device != None:
            self.gcn = self.gcn.to(self.device)
        self.gcn.eval()
        print("start testing")
        true = []
        pred = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                if self.device != None:
                    batch = batch.to(self.device)

                x = batch.x.float()  # 获取节点特征

                pos_edge_index = batch.edge_index

                edge_attr = batch.edge_attr.float()

                pos_target = batch.edge_target.float()
                out_pos = self.gcn(x, pos_edge_index, edge_attr)
                out_pos = out_pos.cpu().numpy()

                pred.append(out_pos)

                pos_target = pos_target.cpu().numpy()

                true.append(pos_target)

        true = np.concatenate(true, axis=0)
        pred = np.concatenate(pred, axis=0)

        true = true.flatten()
        pred = pred.flatten()

        mape = MAPE(true, pred)

        if np.all(true == 0):
            true = np.ones_like(true) * 1e-10
        if np.all(pred == 0):
            pred = np.ones_like(pred) * 1e-10

        cpc = CPC(true, pred)
        jsd = JSD(true, pred)

        if self.args.scale:
            if self.args.scale_method == "Minmaxscaler":
                edge_attr_pos = self.E_data_scaler.inverse_transform(edge_attr_pos)
            elif self.args.scale_method == "log":
                true = np.power(10, true) - 1
                pred = np.power(10, pred) - 1

            elif self.args.scale_method == "boxcox":
                edge_target_ = special.inv_boxcox1p(edge_target.flatten(), self.lam_best_E)
                edge_target = edge_target_.reshape(edge_target.shape)
        mae = MAE(true, pred)
        rmse = RMSE(true, pred)
        print("true_sum", np.sum(true))
        print("test loss rmse:", rmse, "mae:", mae, "mape:", mape, "CPC:", cpc, "JSD:", jsd)
        return mae
    
    
    def train(self):
        print("Start training")
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        for epoch in range(self.epochs):
            self.gcn.train()
            for batch in tqdm(self.train_loader):

                self.optimizer.zero_grad()
                if self.device != None:
                    batch.x = batch.x.float()
                    batch.edge_attr = batch.edge_attr.float()
                    batch.edge_target = batch.edge_target.float()
                    batch = batch.to(self.device)

                x = batch.x  # 获取节点特征
                pos_edge_index = batch.edge_index

                edge_attr = batch.edge_attr

                neg_edge_index = sample_negative_edges(batch)
                


                pos_target = batch.edge_target.float()
                neg_target = torch.zeros([neg_edge_index.shape[1], 1]).float()
                neg_edge_attr = self.all_edge_attr[neg_edge_index[0], neg_edge_index[1], :].reshape(neg_edge_index.shape[1], -1).float().to(self.device)

                if self.device != None:
                    neg_target = neg_target.to(self.device)

                neg_edge_index = neg_edge_index.to(self.device)
                


                out_pos = self.gcn(x, pos_edge_index, edge_attr)
                out_neg = self.gcn(x, neg_edge_index, neg_edge_attr)

                pos_loss = self.loss(out_pos, pos_target)
                neg_loss = self.loss(out_neg, neg_target)

                loss = pos_loss * 0.2 + neg_loss * 0.8

                loss.backward()

                self.optimizer.step()



            torch.cuda.empty_cache()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

            # neg_loss = MSE(y_pred_neg, y_true_neg)

            print(f'Epoch {epoch + 1}/{self.epochs},Pos Loss: {pos_loss:.8f}')# ,Neg Loss: {neg_loss:.8f}')

            test_loss = self.eval()

            if self.early_stopping(test_loss, self.gcn):
                print(f'Early stopping at epoch {epoch + 1}')
                break

        return

    def eval(self):
        self.gcn.eval()
        true = []
        pred = []
        with torch.no_grad():
            for batch in self.val_loader:

                if self.device != None:
                    batch = batch.to(self.device)

                x = batch.x.float()  # 获取节点特征
                pos_edge_index = batch.edge_index

                edge_attr = batch.edge_attr.float()

                neg_edge_index = sample_negative_edges(batch)

                # pos_target = batch.edge_target[pos_edge_index[0]].float()  # 取出正样本边的目标值
                pos_target = batch.edge_target.float()
                neg_target = torch.zeros([neg_edge_index.shape[1], 1]).float()

                neg_edge_attr = self.all_edge_attr[neg_edge_index[0], neg_edge_index[1], :].reshape(neg_edge_index.shape[1], -1).float()
                out_pos = self.gcn(x, pos_edge_index, edge_attr)
                out_neg = self.gcn(x, neg_edge_index, neg_edge_attr)

                torch.cuda.empty_cache()

                out_pos = out_pos.cpu().numpy()
                out_neg = out_neg.cpu().numpy()
                pos_target = pos_target.cpu().numpy()
                if self.args.scale:
                    if self.args.scale_method == "Minmaxscaler":
                        out_pos = self.E_data_scaler.inverse_transform(out_pos)
                    elif self.args.scale_method == "log":
                        out_pos = np.power(10, out_pos) - 1
                        out_neg = np.power(10, out_neg) - 1
                        pos_target = np.power(10, pos_target) - 1
                        neg_target = np.power(10, neg_target) - 1

                    elif self.args.scale_method == "boxcox":
                        out_pos_ = special.inv_boxcox1p(out_pos.flatten(), self.lam_best_E)
                        out_pos = out_pos_.reshape(out_pos.shape)
                pred.append(out_pos)
                pred.append(out_neg)
                true.append(pos_target)
                true.append(neg_target)

        true = np.concatenate(true, axis=0)
        pred = np.concatenate(pred, axis=0)
        mae = MAE(true, pred)
        mse = MSE(true, pred)
        mape = MAPE(true, pred)
        print("val loss mae:", mae, "mape:", mape)
        return mse

    def test(self):
        self.gcn.load_state_dict(torch.load(os.path.join(self.args.save_path, "best_checkpoint_"+self.args.city+".pt"), weights_only=True))
        if self.device != None:
            self.gcn = self.gcn.to(self.device)
        self.gcn.eval()
        print("start testing")
        true = []
        pred = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                if self.device != None:
                    batch = batch.to(self.device)

                x = batch.x.float()  # 获取节点特征

                pos_edge_index = batch.edge_index

                edge_attr = batch.edge_attr.float()

                neg_edge_index = sample_negative_edges(batch)

                # pos_target = batch.edge_target[pos_edge_index[0]].float().cpu()  # 取出正样本边的目标值
                pos_target = batch.edge_target.float()
                neg_target = torch.zeros([neg_edge_index.shape[1], 1]).float()

                neg_edge_attr = self.all_edge_attr[neg_edge_index[0], neg_edge_index[1], :].reshape(neg_edge_index.shape[1], -1).float()
                out_pos = self.gcn(x, pos_edge_index, edge_attr)
                out_neg = self.gcn(x, neg_edge_index, neg_edge_attr)
                
                
                out_pos = out_pos.cpu().numpy()
                out_neg = out_neg.cpu().numpy()

                pred.append(out_pos)
                pred.append(out_neg)

                pos_target = pos_target.cpu().numpy()
                neg_target = neg_target.cpu().numpy()

                true.append(pos_target)
                true.append(neg_target)

        true = np.concatenate(true, axis=0)
        pred = np.concatenate(pred, axis=0)
        
        true = true.flatten()
        pred = pred.flatten()

        print(np.sum(true))
        mape = MAPE(true, pred)

        if np.all(true == 0):
            true = np.ones_like(true) * 1e-10
        if np.all(pred == 0):
            pred = np.ones_like(pred) * 1e-10


        cpc = CPC(true, pred)
        jsd = JSD(true, pred)
        
        if self.args.scale:

            if self.args.scale_method == "log":
                true = np.power(10, true) - 1
                pred = np.power(10, pred) - 1


        mae = MAE(true, pred)
        rmse = RMSE(true, pred)

        print("test loss rmse:", rmse, "mae:", mae, "mape:", mape, "CPC:", cpc, "JSD:", jsd)
        return mae

    def predict_no_negative(self):
        from shapely.wkt import loads
        from matplotlib import cm
        from matplotlib.colors import Normalize
        from shapely.geometry import LineString
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        self.gcn.load_state_dict(
            torch.load(os.path.join(self.args.save_path, "best_checkpoint_" + self.args.city + ".pt"),
                       weights_only=True))
        if self.device != None:
            self.gcn = self.gcn.to(self.device)
        self.gcn.eval()  # 设置模型为评估模式

        # 初始化一个与 E_data 相同大小的矩阵
        predicted_E = np.zeros((len(self.N_data), len(self.N_data)), dtype=float)
        true_E = np.zeros((len(self.N_data), len(self.N_data)), dtype=float)

        # 遍历 DataLoader（仅包含完整图的一批数据）
        with torch.no_grad():
            for batch in self.pred_loader:
                # 将图数据移动到设备
                if self.device is not None:
                    batch = batch.to(self.device)

                # 获取输入特征
                x = torch.tensor(batch.x[0]).float().to(self.device)
                edge_index = batch.edge_index
                edge_attr = batch.edge_attr.float().to(self.device)
                edge_target = batch.edge_target.float().to(self.device)

                # 模型推理
                out = self.gcn(x, edge_index, edge_attr).cpu().numpy()

                # 将预测结果填充到 E_data 矩阵
                for i, (src, dst) in enumerate(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())):
                    predicted_E[src, dst] = out[i, 0]  # 假设模型输出是 (num_edges, 1)
                    true_E[src, dst] = edge_target[i, 0]


        print(CPC(true_E, predicted_E))

    def predict(self):
        from shapely.wkt import loads
        from matplotlib import cm
        from matplotlib.colors import Normalize
        from shapely.geometry import LineString
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        self.gcn.load_state_dict(
            torch.load(os.path.join(self.args.save_path, "best_checkpoint_" + self.args.city + ".pt"),
                       weights_only=True))
        if self.device != None:
            self.gcn = self.gcn.to(self.device)
        self.gcn.eval()  # 设置模型为评估模式

        # 初始化一个与 E_data 相同大小的矩阵
        predicted_E = np.zeros((len(self.N_data), len(self.N_data)), dtype=float)
        true_E = np.zeros((len(self.N_data), len(self.N_data)), dtype=float)

        # 遍历 DataLoader（仅包含完整图的一批数据）
        with torch.no_grad():
            for batch in self.pred_loader:
                # 将图数据移动到设备
                if self.device is not None:
                    batch = batch.to(self.device)

                # 获取输入特征
                x = torch.tensor(batch.x[0]).float().to(self.device)
                edge_index = batch.edge_index
                edge_attr = batch.edge_attr.float().to(self.device)
                edge_target = batch.edge_target.float().to(self.device)

                neg_edge_index = negative_sampling(
                    edge_index=edge_index,
                    num_nodes=batch.num_nodes,
                    num_neg_samples= batch.num_nodes ** 2 - edge_index.size(1)
                )
                neg_edge_attr = self.all_edge_attr[neg_edge_index[0], neg_edge_index[1], :].reshape(
                    neg_edge_index.shape[1], -1).float()
                
                # 模型推理
                out = self.gcn(x, edge_index, edge_attr).cpu().numpy()
                out_neg = self.gcn(x, neg_edge_index, neg_edge_attr).cpu().numpy()

                # 将预测结果填充到 E_data 矩阵
                for i, (src, dst) in enumerate(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())):
                    predicted_E[src, dst] = out[i, 0]  # 假设模型输出是 (num_edges, 1)
                    true_E[src, dst] = edge_target[i, 0]
                for i, (src, dst) in enumerate(zip(neg_edge_index[0].cpu().numpy(), neg_edge_index[1].cpu().numpy())):
                    predicted_E[src, dst] = out_neg[i, 0]  # 假设模型输出是 (num_edges, 1)
        
        print("CPC", CPC(true_E, predicted_E))
        self.N_data_["geometry"] = self.N_data_["geometry"].apply(loads)
        self.N_data_ = gpd.GeoDataFrame(self.N_data_, geometry="geometry")
        self.N_data_["centroid"] = self.N_data_["geometry"].centroid

        self.N_data_all["geometry"] = self.N_data_all["geometry"].apply(loads)
        self.N_data_all = gpd.GeoDataFrame(self.N_data_all, geometry="geometry")


        data = {}
        data["slon"] = []; data["slat"] = []; data["elon"] = []; data["elat"] = []; data["count_true"] = []; data["count_pred"] = []
        from folium.plugins import HeatMap
        import folium
        from matplotlib import cm
        from matplotlib.colors import Normalize

        # 绘制流量（真值）和预测流量，跳过流量为零的部分
        for i in range(len(self.N_data_)):
            for j in range(len(self.N_data_)):
                data["slon"].append(self.N_data_.loc[i, "centroid"].x)
                data["slat"].append(self.N_data_.loc[i, "centroid"].y)
                data["elon"].append(self.N_data_.loc[j, "centroid"].x)
                data["elat"].append(self.N_data_.loc[j, "centroid"].y)
                data["count_true"].append(true_E[i,j])
                data["count_pred"].append(predicted_E[i,j])

        data = pd.DataFrame(data)
        # 创建一个地图对象，设置地图的初始位置和缩放级别
        center_lat = (data["slat"].mean() + data["elat"].mean()) / 2
        center_lon = (data["slon"].mean() + data["elon"].mean()) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='CartoDB Positron')

        # 定义颜色映射
        norm = Normalize(vmin=2.5, vmax=data['count_true'].max())  # 归一化 data['count_true'].min()
        colormap = cm.get_cmap('magma_r')  # 使用 'Yellow-Orange-Red' 颜色映射
    
        print("start drawing")
        # 绘制 OD 流量线
        for i, row in data.iterrows():
            if data.loc[i, "count_pred"] < 2.5:
                continue
            color = colormap(norm(row['count_pred']))  # 映射到归一化颜色
            color = f'rgba({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}, {color[3]})'  # 转换为 RGBA 格式
            folium.PolyLine(
                locations=[(row['slat'], row['slon']), (row['elat'], row['elon'])],
                weight=0.3,  # 根据流量来控制线的粗细
                color=color,
                opacity=0.1
            ).add_to(m)
            print("line drawing complete", i/len(data))

        # 添加颜色条（可选）
        from branca.colormap import LinearColormap
        linear_colormap = LinearColormap(
            [colormap(norm(v))[:3] for v in [data['count_true'].min(), data['count_true'].max()]],
            vmin=data['count_true'].min(),
            vmax=data['count_true'].max(),
            caption="Flow Count"
        )
        linear_colormap.add_to(m)

        # 将结果保存为 HTML 文件
        m.save('od_map_pred.html')

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles='CartoDB Positron')

        # 定义颜色映射
        norm = Normalize(vmin=2.5, vmax=data['count_true'].max())  # 归一化 data['count_true'].min()
        colormap = cm.get_cmap('magma_r')  # 使用 'Yellow-Orange-Red' 颜色映射

        print("start drawing")
        # 绘制 OD 流量线
        for i, row in data.iterrows():
            if data.loc[i, "count_true"] < 2.5:
                continue
            color = colormap(norm(row['count_true']))  # 映射到归一化颜色
            color = f'rgba({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}, {color[3]})'  # 转换为 RGBA 格式
            folium.PolyLine(
                locations=[(row['slat'], row['slon']), (row['elat'], row['elon'])],
                weight=0.3,  # 根据流量来控制线的粗细
                color=color,
                opacity=0.1
            ).add_to(m)
            print("line drawing complete", i / len(data))

        # 添加颜色条（可选）
        from branca.colormap import LinearColormap
        linear_colormap = LinearColormap(
            [colormap(norm(v))[:3] for v in [data['count_true'].min(), data['count_true'].max()]],
            vmin=data['count_true'].min(),
            vmax=data['count_true'].max(),
            caption="Flow Count"
        )
        linear_colormap.add_to(m)

        # 将结果保存为 HTML 文件
        m.save('od_map_true.html')

        return predicted_E

    def train_dis(self):
        print("Start training")
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        for epoch in range(self.epochs):
            self.gcn.train()
            for batch in tqdm(self.train_loader):

                self.optimizer.zero_grad()
                if self.device != None:
                    batch.x = batch.x.float()
                    batch.edge_attr = batch.edge_attr.float()
                    batch.edge_target = batch.edge_target.float()
                    batch = batch.to(self.device)

                x = batch.x  # 获取节点特征
                pos_edge_index = batch.edge_index

                edge_attr = batch.edge_attr

                neg_edge_index = sample_negative_edges(batch)

                # pos_target = batch.edge_target[pos_edge_index[0]].float() # 取出正样本边的目标值
                pos_target = batch.edge_target.float()
                neg_target = torch.tensor([self.E_data[i, j] for i, j in zip(neg_edge_index[0].cpu().numpy(),
                                                                             neg_edge_index[1].cpu().numpy())]).float()
                neg_target = neg_target.unsqueeze(-1).to(self.device)  # 将目标值调整为 (num_neg_edges, 1)
                neg_edge_attr = self.all_edge_attr[neg_edge_index[0], neg_edge_index[1], :].reshape(
                    neg_edge_index.shape[1], -1).float().to(self.device)

                if self.device != None:
                    neg_target = neg_target.to(self.device)

                neg_edge_index = neg_edge_index.to(self.device)

                try:
                    out_pos = self.gcn(x, pos_edge_index, edge_attr)
                    out_neg = self.gcn(x, neg_edge_index, neg_edge_attr)

                    pos_loss = self.loss(out_pos, pos_target)
                    neg_loss = self.loss(out_neg, neg_target)

                    loss = pos_loss + neg_loss

                    loss.backward()

                    self.optimizer.step()
                except:
                    continue

            torch.cuda.empty_cache()

            # neg_loss = MSE(y_pred_neg, y_true_neg)

            print(f'Epoch {epoch + 1}/{self.epochs},Pos Loss: {pos_loss:.8f}')  # ,Neg Loss: {neg_loss:.8f}')

            test_loss = self.eval_dis()

            if self.early_stopping(test_loss, self.gcn):
                print(f'Early stopping at epoch {epoch + 1}')
                break

        return

    def eval_dis(self):
        self.gcn.eval()
        true = []
        pred = []
        with torch.no_grad():
            for batch in self.val_loader:

                if self.device != None:
                    batch = batch.to(self.device)

                x = batch.x.float()  # 获取节点特征
                pos_edge_index = batch.edge_index

                edge_attr = batch.edge_attr.float()

                neg_edge_index = sample_negative_edges(batch)

                # pos_target = batch.edge_target[pos_edge_index[0]].float()  # 取出正样本边的目标值
                pos_target = batch.edge_target.float()
                neg_target = torch.tensor([self.E_data[i, j] for i, j in zip(neg_edge_index[0].cpu().numpy(),
                                                                             neg_edge_index[1].cpu().numpy())]).float()
                neg_target = neg_target.unsqueeze(-1)  # 将目标值调整为 (num_neg_edges, 1)

                neg_edge_attr = self.all_edge_attr[neg_edge_index[0], neg_edge_index[1], :].reshape(
                    neg_edge_index.shape[1], -1).float()
                out_pos = self.gcn(x, pos_edge_index, edge_attr)
                out_neg = self.gcn(x, neg_edge_index, neg_edge_attr)

                torch.cuda.empty_cache()

                out_pos = out_pos.cpu().numpy()
                out_neg = out_neg.cpu().numpy()
                pos_target = pos_target.cpu().numpy()
                if self.args.scale:
                    if self.args.scale_method == "Minmaxscaler":
                        out_pos = self.E_data_scaler.inverse_transform(out_pos)
                    elif self.args.scale_method == "log":
                        out_pos = np.power(10, out_pos) - 1
                        out_neg = np.power(10, out_neg) - 1
                        pos_target = np.power(10, pos_target) - 1
                        neg_target = np.power(10, neg_target) - 1

                    elif self.args.scale_method == "boxcox":
                        out_pos_ = special.inv_boxcox1p(out_pos.flatten(), self.lam_best_E)
                        out_pos = out_pos_.reshape(out_pos.shape)
                pred.append(out_pos)
                pred.append(out_neg)
                true.append(pos_target)
                true.append(neg_target)

        true = np.concatenate(true, axis=0)
        pred = np.concatenate(pred, axis=0)
        mae = MAE(true, pred)
        mse = MSE(true, pred)
        mape = MAPE(true, pred)
        print("val loss mae:", mae, "mape:", mape)
        return mse

    def test_dis(self):
        self.gcn.load_state_dict(
            torch.load(os.path.join(self.args.save_path, "best_checkpoint_" + self.args.city + ".pt"),
                       weights_only=True))
        if self.device != None:
            self.gcn = self.gcn.to(self.device)
        self.gcn.eval()
        print("start testing")
        true = []
        pred = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                if self.device != None:
                    batch = batch.to(self.device)

                x = batch.x.float()  # 获取节点特征

                pos_edge_index = batch.edge_index

                edge_attr = batch.edge_attr.float()

                neg_edge_index = sample_negative_edges(batch)

                # pos_target = batch.edge_target[pos_edge_index[0]].float().cpu()  # 取出正样本边的目标值
                pos_target = batch.edge_target.float()
                neg_target = torch.tensor([self.E_data[i, j] for i, j in zip(neg_edge_index[0].cpu().numpy(),
                                                                             neg_edge_index[1].cpu().numpy())]).float()
                neg_target = neg_target.unsqueeze(-1)  # 将目标值调整为 (num_neg_edges, 1)

                neg_edge_attr = self.all_edge_attr[neg_edge_index[0], neg_edge_index[1], :].reshape(
                    neg_edge_index.shape[1], -1).float()
                out_pos = self.gcn(x, pos_edge_index, edge_attr)
                out_neg = self.gcn(x, neg_edge_index, neg_edge_attr)

                out_pos = out_pos.cpu().numpy()
                out_neg = out_neg.cpu().numpy()

                pred.append(out_pos)
                pred.append(out_neg)

                pos_target = pos_target.cpu().numpy()
                neg_target = neg_target.cpu().numpy()

                true.append(pos_target)
                true.append(neg_target)

        true = np.concatenate(true, axis=0)
        pred = np.concatenate(pred, axis=0)

        true = true.flatten()
        pred = pred.flatten()

        print(np.sum(true))
        mape = MAPE(true, pred)

        if np.all(true == 0):
            true = np.ones_like(true) * 1e-10
        if np.all(pred == 0):
            pred = np.ones_like(pred) * 1e-10

        cpc = CPC(true, pred)
        jsd = JSD(true, pred)

        if self.args.scale:
            if self.args.scale_method == "Minmaxscaler":
                edge_attr_pos = self.E_data_scaler.inverse_transform(edge_attr_pos)
            elif self.args.scale_method == "log":
                true = np.power(10, true) - 1
                pred = np.power(10, pred) - 1

            elif self.args.scale_method == "boxcox":
                edge_target_ = special.inv_boxcox1p(edge_target.flatten(), self.lam_best_E)
                edge_target = edge_target_.reshape(edge_target.shape)
        mae = MAE(true, pred)
        rmse = RMSE(true, pred)

        print("test loss rmse:", rmse, "mae:", mae, "mape:", mape, "CPC:", cpc, "JSD:", jsd)
        return mae