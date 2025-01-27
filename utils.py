import numpy as np
import torch

from scipy.spatial import distance

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# def CPC(true, pred):
#     CPC_ = np.sum(np.abs(true - pred)) / np.sum(true)
#     return 1 - 0.5 * CPC_

def CPC(true, pred):
    epsilon = 1e-6  # 避免分母为零
    # 计算最小值的和，以度量两个向量之间的重合部分
    overlap = np.sum(np.minimum(true, pred))
    total = np.sum(true) + np.sum(pred) + epsilon
    CPC_ = 2 * overlap / total
    return CPC_

def MSE(true, pred):
    return np.mean((true - pred) ** 2)

def RMSE(true, pred):
    return np.sqrt(np.nanmean((true - pred) ** 2))

def MAE(true, pred):
    return np.nanmean(np.abs(true - pred))

def non_zero_MAPE(true, pred):
    non_zero_index = np.argwhere(true != 0)
    non_zero_pred = pred[non_zero_index[:, 0], non_zero_index[:, 1]]
    non_zero_true = true[non_zero_index[:, 0], non_zero_index[:, 1]]

    return np.mean(np.abs(non_zero_true - non_zero_pred) / non_zero_true) * 100

def r2(true, pred):
    return r2_score(true, pred)

def MAPE(true, pred, null_val=0, adjust_value=1e-2):
    true[true < adjust_value] = 0
    pred[pred < adjust_value] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(true)
        else:
            mask = np.not_equal(true, null_val)
        mask = mask.astype('float32')
        mask /= np.nanmean(mask)
        mape = np.abs(np.divide((pred - true).astype('float32'), true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def MAPE1(y_true, y_pred, epsilon=1):
    # 创建掩码，标记哪些 y_true 为 0
    mask = y_true != 0

    # 使用掩码分别计算真值为零和非零的情况
    mape_non_zero = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    mape_zero = np.abs((y_true[~mask] - y_pred[~mask]) / epsilon)

    # 合并结果并计算平均值
    mape = np.mean(np.concatenate((mape_non_zero, mape_zero))) * 100
    return mape

def JSD(true, pred):
    return distance.jensenshannon(true, pred)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.checkpoint_path = checkpoint_path
        self.val_loss_min = np.inf


    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            return False
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
            else:
                return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            return False

    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))