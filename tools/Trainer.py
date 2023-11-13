import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tools.DataLoader import SummaryDataset
from tools.flows import get_flow_model
from tools.FCN import ANN_Classifier, ANN
from tools.MINE import batch_to_score,batch_to_score_with_r, get_MI_estimator
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import logging
import numpy as np
import os
from copy import deepcopy

# Weight initialization
def weight_init_fn(module):
    """ Initialize weights """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, std=0.01)
        torch.nn.init.zeros_(module.bias)


import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_MI(MIs, show_result=False):
    if len(MIs.shape) > 1:
        lineObjects = plt.plot(MIs.T)
        plt.legend(lineObjects, ('train', 'val', 'total'))
    else:
        plt.plot(MIs)

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.draw()
    plt.pause(0.001)

    if is_ipython:

        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def train_mine(dataset,
               para,
               MI_estimator_name: string,
               train_info: dict = {
                   'epochs': 1000,
                   'batch_size': 32,
                   'learning_rate': 1e-3
               },
               verbose=True):

    epochs = train_info['epochs']
    batch_size = train_info['batch_size']
    lr = train_info['learning_rate']

    MI_estimator = get_MI_estimator(MI_estimator_name)

    DS = SummaryDataset(dataset, para, norm=False)

    train_frac = 0.8
    val_frac = 0.2

    n_train = int(len(DS) * train_frac)
    n_val = len(DS) - n_train

    train, val = random_split(DS, [n_train, n_val])

    train_data = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val, batch_size=batch_size, shuffle=True)

    total_data = DataLoader(DS, batch_size=batch_size, shuffle=True)

    model = ANN_Classifier(data_dim=dataset.shape[1],
                           para_dim=para.shape[1],
                           hidden_dims=[256, 128, 64, 32],
                           apply_dropout=False)

    model.apply(weight_init_fn)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda')
    model.to(device=device)

    best_eval = -np.inf
    total_MI = 0
    train_MIs = []
    eval_MIs = []
    total_MIs = []
    for epoch in range(epochs):

        epoch_loss = 0

        epoch_MI = 0

        model.train()

        for batches in train_data:

            datas = (batches['data'])

            params = batches['param']

            datas = datas.to(device=device, dtype=torch.float32)
            params = params.to(device=device, dtype=torch.float32)

            model.zero_grad()
            score = batch_to_score(datas, params, model)

            loss = -MI_estimator(score)

            loss.backward()

            epoch_loss += loss.item()

            epoch_MI += -(loss.item()) * len(datas) / n_train

            optimizer.step()

        train_MIs.append(epoch_MI)

        eval_MI = eval_mine(model, MI_estimator, val_data)
        eval_MIs.append(eval_MI)

        if eval_MI > best_eval:

            best_eval = eval_MI
            best_epoch = epoch
            total_MI = eval_mine(model, MI_estimator, total_data)

        total_MIs.append(total_MI)

        if verbose:
            plot_MI(np.array(total_MIs))

        if verbose:
            plot_MI(np.array([train_MIs, eval_MIs, total_MIs]))

    if verbose:
        plot_MI(np.array([train_MIs, eval_MIs, total_MIs]), show_result=True)

    return train_MIs, eval_MIs, total_MIs


def eval_mine(model, MI_estimator, val_data, device='cuda'):
    model.eval()
    eval_MI = 0
    for batches in val_data:

        datas = batches['data']
        params = batches['param']

        datas = datas.to(device=device, dtype=torch.float32)
        params = params.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            score = batch_to_score(datas, params, model)
            loss = -MI_estimator(score)
            eval_MI += -(loss.item()) * len(datas) / len(val_data.dataset)

    return eval_MI



def train_mine_with_r(dataset,
               para,
               r,
               MI_estimator_name: string,
               train_info: dict = {
                   'epochs': 1000,
                   'batch_size': 32,
                   'learning_rate': 1e-3
               },
               verbose=True):

    epochs = train_info['epochs']
    batch_size = train_info['batch_size']
    lr = train_info['learning_rate']

    MI_estimator = get_MI_estimator(MI_estimator_name)

    DS = SummaryDataset(dataset, para, norm=False)

    train_frac = 0.8
    val_frac = 0.2

    n_train = int(len(DS) * train_frac)
    n_val = len(DS) - n_train

    train, val = random_split(DS, [n_train, n_val])

    train_data = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val, batch_size=batch_size, shuffle=True)

    total_data = DataLoader(DS, batch_size=batch_size, shuffle=True)

    model = ANN_Classifier(data_dim=dataset.shape[1],
                           para_dim=para.shape[1],
                           hidden_dims=[256, 128],
                           apply_dropout=False)

    # model.apply(weight_init_fn)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda')
    model.to(device=device)

    best_eval = -np.inf
    total_MI = 0
    train_MIs = []
    eval_MIs = []
    total_MIs = []
    for epoch in range(epochs):

        epoch_loss = 0

        epoch_MI = 0

        model.train()

        for batches in train_data:

            datas = (batches['data'])

            params = batches['param']

            datas = datas.to(device=device, dtype=torch.float32)
            params = params.to(device=device, dtype=torch.float32)

            model.zero_grad()
            score = batch_to_score_with_r(datas, params, model, r)

            loss = -MI_estimator(score)

            loss.backward()

            epoch_loss += loss.item()

            epoch_MI += -(loss.item()) * len(datas) / n_train

            optimizer.step()

        train_MIs.append(epoch_MI)

        eval_MI = eval_mine_with_r(model, MI_estimator, val_data,r)
        eval_MIs.append(eval_MI)

        if eval_MI > best_eval:

            best_eval = eval_MI
            best_epoch = epoch
            total_MI = eval_mine_with_r(model, MI_estimator, total_data, r)

        total_MIs.append(total_MI)

        if verbose:
            plot_MI(np.array(total_MIs))

        if verbose:
            plot_MI(np.array([train_MIs, eval_MIs, total_MIs]))

    if verbose:
        plot_MI(np.array([train_MIs, eval_MIs, total_MIs]), show_result=True)

    return train_MIs, eval_MIs, total_MIs


def eval_mine_with_r(model, MI_estimator, val_data,r, device='cuda'):
    model.eval()
    eval_MI = 0
    for batches in val_data:

        datas = batches['data']
        params = batches['param']

        datas = datas.to(device=device, dtype=torch.float32)
        params = params.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            score = batch_to_score_with_r(datas, params, model, r)
            loss = -MI_estimator(score)
            eval_MI += -(loss.item()) * len(datas) / len(val_data.dataset)

    return eval_MI


def train_ba(dataset,
             para,
             train_info: dict = {
                 'epochs': 1000,
                 'batch_size': 32,
                 'learning_rate': 1e-3
             },
             verbose=True):

    epochs = train_info['epochs']
    batch_size = train_info['batch_size']
    lr = train_info['learning_rate']
    wd = train_info['weigth_decay']

    DS = SummaryDataset(dataset, para, norm=False)
    train_frac = 0.8
    val_frac = 0.2

    n_train = int(len(DS) * train_frac)
    n_val = len(DS) - n_train

    train, val = random_split(DS, [n_train, n_val])

    train_data = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val, batch_size=len(val), shuffle=True)

    total_data = DataLoader(DS, batch_size=len(DS), shuffle=True)

    num_inputs = para.shape[1]
    num_cond_inputs = dataset.shape[1]
    num_hidden = 32
    num_blocks = 5

    Hx = np.log(4)

    model = get_flow_model(num_blocks=num_blocks,
                           num_inputs=num_inputs,
                           num_hidden=num_hidden,
                           num_cond_inputs=num_cond_inputs)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    device = torch.device('cuda')
    model.to(device=device)

    best_eval = -np.inf
    total_MI = 0
    train_MIs = []
    eval_MIs = []
    total_MIs = []
    for epoch in range(epochs):

        epoch_loss = 0
        epoch_MI = 0
        eval_MI = 0
        model.train()
        for batches in train_data:
            datas = batches['data']
            params = batches['param']

            datas = datas.to(device=device, dtype=torch.float32)
            params = params.to(device=device, dtype=torch.float32)

            model.zero_grad()

            loss = -model.log_probs(params, datas).mean()
            loss.backward()
            epoch_loss += loss.item()
            epoch_MI += -loss.item()

            optimizer.step()

        epoch_MI = epoch_MI / len(train_data) + Hx
        train_MIs.append(epoch_MI)

        eval_MI = eval_CE(model, val_data) + Hx
        eval_MIs.append(eval_MI)

        if eval_MI > best_eval:

            best_eval = eval_MI
            best_epoch = epoch
            best_model_dict = model.state_dict().copy()
            total_MI = eval_CE(model, total_data) + Hx

        total_MIs.append(total_MI)

        scheduler.step()
        if verbose:
            plot_MI(np.array([train_MIs, eval_MIs, total_MIs]))

    if verbose:
        plot_MI(np.array([train_MIs, eval_MIs, total_MIs]), show_result=True)

    #load best dict
    model.load_state_dict(best_model_dict)
    return model,train_MIs, eval_MIs, total_MIs


# def train_ba_with_test(dataset,para,train_info:dict={'epochs':1000,'batch_size':32,'learning_rate':1e-3},verbose=True):

#     epochs=train_info['epochs']
#     batch_size=train_info['batch_size']
#     lr=train_info['learning_rate']
#     wd=train_info['weigth_decay']

#     DS=SummaryDataset(dataset,para,norm=True)
#     train_frac=0.4
#     val_frac=0.1
#     test_frac=0.5

#     n_test = int(len(DS)*test_frac)
#     n_train=int(len(DS)*train_frac)
#     n_val=len(DS)-n_test-n_train

#     train, val, test = random_split(DS, [n_train, n_val,n_test])

#     train_data=DataLoader(train,batch_size=batch_size,shuffle=True)
#     val_data=DataLoader(val,batch_size=len(val),shuffle=True)
#     test_data=DataLoader(test,batch_size=len(test),shuffle=True)

#     num_inputs=para.shape[1]
#     num_cond_inputs=dataset.shape[1]
#     num_hidden=32
#     num_blocks=5

#     Hx=np.log(4)

#     model=get_flow_model(num_blocks=num_blocks,num_inputs=num_inputs,num_hidden=num_hidden,num_cond_inputs=num_cond_inputs)

#     optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
#     scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

#     device = torch.device('cuda')
#     model.to(device=device)

#     best_eval=-np.inf
#     test_MI=0
#     total_MIs=[]
#     total_eval_MIs=[]
#     total_test_MIs=[]
#     for epoch in range(epochs):

#         trained=0
#         epoch_loss=0
#         epoch_MI=0
#         eval_MI=0
#         model.train()

#         for batches in train_data:
#             datas=batches['data']
#             params=batches['param']

#             datas = datas.to(device=device, dtype=torch.float32)
#             params = params.to(device=device, dtype=torch.float32)

#             model.zero_grad()

#             loss = -model.log_probs(params,datas).mean()
#             loss.backward()
#             epoch_loss+=loss.item()
#             epoch_MI+=-loss.item()

#             optimizer.step()
#             trained+=batch_size

#         epoch_MI=epoch_MI/len(train_data)+Hx
#         total_MIs.append(epoch_MI)

#         eval_MI=eval_CE(model,val_data)+Hx
#         total_eval_MIs.append(eval_MI)

#         if eval_MI>best_eval:

#             best_eval=eval_MI
#             best_epoch=epoch
#             test_MI=eval_CE(model,test_data)+Hx

#         total_test_MIs.append(test_MI)

#         scheduler.step()
#         if verbose:
#             plot_MI(np.array([total_MIs,total_eval_MIs,total_test_MIs]))

#     return total_MIs,total_eval_MIs,total_test_MIs


def eval_CE(model, val_data, device='cuda'):
    model.eval()
    CE = 0

    for val_batches in val_data:

        datas = val_batches['data']
        params = val_batches['param']

        datas = datas.to(device=device, dtype=torch.float32)
        params = params.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            loss = -model.log_probs(params, datas).mean()
        CE += -loss.item()

    CE = CE / len(val_data)

    return CE


def add_random_dim(data, N_dim=10):
    N_data = data.shape[0]
    noise = np.random.normal(0, 1, (N_data, N_dim))

    return np.hstack((data, noise))


from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import early_stopping
import contextlib


def feature_selection(data_ori, para_ori):

    N_feature = data_ori.shape[1]
    N_noise = 10
    N_trail = 100

    selection_strength = 3

    ips = np.zeros((N_trail, N_feature))
    ips_ns = np.zeros((N_trail, N_noise))
    selection = np.zeros((N_feature))

    for idx_para in range(para_ori.shape[1]):
        for i in range(N_trail):
            data_ns = add_random_dim(data_ori, N_noise)
            xtrain, xtest, ytrain, ytest = train_test_split(data_ns,
                                                            para_ori,
                                                            test_size=.10,
                                                            random_state=1)
            with contextlib.redirect_stdout(None):
                dtr = lgb.LGBMRegressor(objective='regression',
                                        num_leaves=31,
                                        learning_rate=0.1,
                                        min_data_in_leaf=200,
                                        n_estimators=150,
                                        importance_type='gain',
                                        random_state=i)
                dtr.fit(xtrain,
                        ytrain[:, idx_para],
                        eval_set=[(xtest, ytest[:, idx_para])],
                        eval_metric=['l2'],
                        callbacks=[early_stopping(5)])

            ips[i] = dtr.feature_importances_[:N_feature]
            ips_ns[i] = dtr.feature_importances_[-N_noise:]

        mean_ips = np.mean(ips, axis=0)
        std_ips = np.std(ips, axis=0)
        mean_ns = np.mean(ips_ns)
        std_ns = np.std(ips_ns)
        selection = np.logical_or(
            mean_ips > (mean_ns + selection_strength * std_ns), selection)
    return selection


import copy


def train_SumNet(dataset,
                 para,
                 train_info: dict = {
                     'epochs': 1000,
                     'batch_size': 32,
                     'learning_rate': 1e-3
                 },
                 verbose=True):
    '''
    train a NN a compress input summaries into same dimension of parameters.
    '''

    epochs = train_info['epochs']
    batch_size = train_info['batch_size']
    lr = train_info['learning_rate']

    DS = SummaryDataset(dataset, para, norm=False)

    train_frac = 0.8
    val_frac = 0.2

    n_train = int(len(DS) * train_frac)
    n_val = len(DS) - n_train

    train, val = random_split(DS, [n_train, n_val])

    train_data = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val, batch_size=batch_size, shuffle=True)
    model = ANN(input_dim=dataset.shape[1],
                hidden_dims=[128, 64, 32],
                output_dim=para.shape[1],
                apply_dropout=False)

    # model.apply(weight_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    MSE = nn.MSELoss()
    device = torch.device('cuda')
    model.to(device=device)

    best_eval = np.inf
    best_model = None

    losses = []
    eval_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batches in train_data:

            datas = (batches['data'])
            params = batches['param']
            datas = datas.to(device=device, dtype=torch.float32)
            params = params.to(device=device, dtype=torch.float32)

            model.zero_grad()
            preds = model(datas)
            loss = MSE(preds, params)
            loss.backward()

            epoch_loss += loss.item()
            optimizer.step()

        model.eval()
        eval_loss = 0
        for batches in val_data:
            datas = (batches['data'])
            params = batches['param']
            datas = datas.to(device=device, dtype=torch.float32)
            params = params.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                preds = model(datas)
                loss = MSE(preds, params)

                eval_loss += loss.item()

        if eval_loss < best_eval:

            best_eval = epoch_loss
            best_model = copy.deepcopy(model)

        losses.append(epoch_loss / len(train_data))
        eval_losses.append(eval_loss / len(val_data))
        if verbose:
            plot_MI(np.array([losses, eval_losses]))

    if verbose:
        plot_MI(np.array([losses, eval_losses]), show_result=True)

    return best_model
