
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import early_stopping
import contextlib
import numpy as np


def add_random_dim(data, N_dim=10):
    N_data = data.shape[0]
    noise = np.random.normal(0, 1, (N_data, N_dim))

    return np.hstack((data, noise))


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