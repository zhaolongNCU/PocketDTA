
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_aupr_davis(Y, P):
    Y = np.where(Y >= 7, 1, 0)
    P = np.where(P >= 7, 1, 0)
    print(Y,P)
    prec, re, _ = metrics.precision_recall_curve(Y, P)
    aupr = metrics.auc(re, prec)
    return aupr

def get_aupr_kiba(Y, P):
    Y = np.where(Y >= 12.1, 1, 0)
    P = np.where(P >= 12.1, 1, 0)
    print(Y,P)
    prec, re, _ = metrics.precision_recall_curve(Y, P)
    aupr = metrics.auc(re, prec)
    return aupr

def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)
    return CI

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / (float(y_obs_sq * y_pred_sq) + 0.00000001)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / (float(sum(y_pred * y_pred)) + 0.00000001)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / (float(down) + 0.00000001))

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def calculate_metrics(Y, P):
    cindex = get_cindex(Y, P) 
    rm2 = get_rm2(Y, P)  
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)

    print('mse:', mse)
    print('cindex:', cindex)
    print('rm2:', rm2)
    print('pearson', pearson)
    print('spearman',spearman)

    return mse,cindex,rm2,pearson,spearman


def plot_train_val_metric(epochs, train_metric, val_metric, base_path, metric_name, dataset_name):
    plt.figure(figsize=(10,10),dpi=300)
    plt.plot(epochs, train_metric, '#3fc1fd', label='Training %s' % metric_name)
    plt.plot(epochs, val_metric, '#d09fff', label='Validation %s' % metric_name)
    plt.title(f'Training and validation {metric_name} on {dataset_name}')
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(os.path.join(base_path, dataset_name + '_' + metric_name +'.jpg'),dpi=300)
    plt.cla()

