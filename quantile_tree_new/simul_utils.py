import numpy as np
from sklearn.metrics import mean_pinball_loss

# from mqboost import MQRegressor
# from ngboost import NGBRegressor

# import optuna
# from optuna import Trial

# import lightgbm as lgb 
# import xgboost as xgb

from scipy.stats import t, skewnorm, laplace_asymmetric, genextreme, norm
import scipy 

def gen_simul1(train_n, input_dim=1, scale=0.1, seed=42):
    np.random.seed(seed)
    # 학습 데이터 생성
    x_train = np.random.uniform(-1, 1, (train_n, input_dim))
    sincx = np.sin(np.pi * x_train[:, 0]) / (np.pi * x_train[:, 0])
    Z = np.reshape(sincx, (train_n, 1))
    
    ep = genextreme.rvs(c = -1/2, scale= scale * np.exp(1 - x_train[:, 0]), size=train_n)[:, np.newaxis]
    # ep = np.random.normal(0, scale * np.exp(1 - x_train), (train_n, input_dim))
    y_train = Z + ep

    # alpha = np.arange(0.1, 1.0, 0.2)
    # true_train_q = []
    # for a in alpha:
    #     true_train_q.append(Z + genextreme.ppf(a, c = -1/2, scale = scale * np.exp(1 - x_train[:, 0]))[:, np.newaxis])
    
    # 검증 데이터 생성
    valid_n = 500 
    x_valid = np.random.uniform(-1, 1, (valid_n, input_dim))
    sincx_valid = np.sin(np.pi * x_valid[:, 0]) / (np.pi * x_valid[:, 0])
    y_valid = np.reshape(sincx_valid, (valid_n, 1)) + genextreme.rvs(c = -1/2, scale= scale * np.exp(1 - x_valid[:, 0]), size=valid_n)[:, np.newaxis]

    # true_valid_q = []
    # for a in alpha:
    #     true_valid_q.append(sincx_valid + genextreme.ppf(a, c = -1/2, scale = scale * np.exp(1 - x_valid[:, 0]))[:, np.newaxis])
    
    # 테스트 데이터 생성
    test_n = 500
    x_test = np.random.uniform(-1, 1, (test_n, input_dim))
    sincx_test = np.sin(np.pi * x_test[:, 0]) / (np.pi * x_test[:, 0])
    y_test = np.reshape(sincx_test, (test_n, 1)) + genextreme.rvs(c = -1/2, scale= scale * np.exp(1 - x_test[:, 0]), size=test_n)[:, np.newaxis]

    # true_test_q = []
    # for a in alpha:
    #     true_test_q.append(sincx_test + genextreme.ppf(a, c = -1/2, scale = scale * np.exp(1 - x_test[:, 0]))[:, np.newaxis])
    
    return {"data": x_train, "label": y_train}, {"data": x_valid, "label": y_valid}, {"data": x_test, "label": y_test}
            #{"true_train_q": true_train_q}, {"true_valid_q": true_valid_q}, {"true_test_q": true_test_q}

def gen_simul2(train_n, input_dim=1, scale=0.1, seed=42):
    np.random.seed(seed)
    # 학습 데이터 생성
    x_train = np.random.uniform(-1, 1, (train_n, input_dim))
    sincx = np.sin(np.pi * x_train[:, 0]) / (np.pi * x_train[:, 0])
    Z = np.reshape(sincx, (train_n, 1))
     # ep = genextreme.rvs(c = -1/2, scale= scale * np.exp(1 - x_train[:, 0]), size=train_n)[:, np.newaxis]
    ep = np.random.normal(0, scale * np.exp(1 - x_train[:, 0])[:, np.newaxis], size=(train_n, 1))
    y_train = 10 * scale * Z + ep

    # alpha = np.arange(0.1, 1.0, 0.2)
    # true_train_q = []
    # for a in alpha:
    #     true_train_q.append(10 * scale * sincx + norm.ppf(a, scale = scale * np.exp(1 - x_train[:, 0])))
    
    # 검증 데이터 생성
    valid_n = 500 
    x_valid = np.random.uniform(-1, 1, (valid_n, input_dim))
    sincx_valid = np.sin(np.pi * x_valid[:, 0]) / (np.pi * x_valid[:, 0])
    y_valid = 10 * scale * np.reshape(sincx_valid, (valid_n, 1)) + np.random.normal(0, scale * np.exp(1 - x_valid[:, 0])[:, np.newaxis], (valid_n, 1))

    # true_valid_q = []
    # for a in alpha:
    #     true_valid_q.append(10 * scale * sincx_valid + norm.ppf(a, scale = scale * np.exp(1 - x_valid[:, 0])))
    
    # 테스트 데이터 생성
    test_n = 500
    x_test = np.random.uniform(-1, 1, (test_n, input_dim))
    sincx_test = np.sin(np.pi * x_test[:, 0]) / (np.pi * x_test[:, 0])
    y_test = 10 * scale * np.reshape(sincx_test, (test_n, 1)) + np.random.normal(0, scale * np.exp(1 - x_test[:, 0])[:, np.newaxis], (test_n, 1))

    # true_test_q = []
    # for a in alpha:
    #     true_test_q.append(10 * scale * sincx_test + norm.ppf(a, scale = scale * np.exp(1 - x_test[:, 0])))
    
    return {"data": x_train, "label": y_train}, {"data": x_valid, "label": y_valid}, {"data": x_test, "label": y_test}
# {"true_train_q": true_train_q}, {"true_valid_q": true_valid_q}, {"true_test_q": true_test_q}


def gen_simul3(train_n, input_dim=1, scale=0.3, seed=42):
    np.random.seed(seed)
    # 학습 데이터 생성
    
    x_train = np.random.uniform(-1, 1, (train_n, input_dim))
    def step_fun(x):
        x = x[:, 0]
        y = np.zeros_like(x)
        ind1 = (-1 <= x) & (x < -0.7)
        ind2 = (-0.7 <= x) & (x < -0.3)
        ind3 = (-0.3 <= x) & (x < -0.1)
        ind4 = (-0.1 <= x) & (x < 0.3)
        ind5 = (0.3 <= x) & (x < 0.7)
        ind6 = (0.7 <= x) & (x <= 1)
        for c, ind in zip([0, 0.4, 0.9, 1.5, 1.2, 1.3], [ind1, ind2, ind3, ind4, ind5, ind6]):
            y[ind] = c
        return y
    
    sincx = step_fun(x_train)
     # ep = genextreme.rvs(c = -1/2, scale= scale * np.exp(1 - x_train[:, 0]), size=train_n)[:, np.newaxis]
    ep = np.random.normal(0, scale, size = train_n)
    y_train = sincx + ep

    # alpha = np.arange(0.1, 1.0, 0.2)
    # true_train_q = []
    # for a in alpha:
    #     true_train_q.append(sincx + norm.ppf(a, scale = scale))
    
    # 검증 데이터 생성
    valid_n = 500 
    x_valid = np.random.uniform(-1, 1, (valid_n, input_dim))
    sincx_valid = step_fun(x_valid)
    y_valid = sincx_valid + np.random.normal(0, scale, size = valid_n)

    # true_valid_q = []
    # for a in alpha:
    #     true_valid_q.append(sincx_valid + norm.ppf(a, scale = scale))
    
    # 테스트 데이터 생성
    test_n = 500
    x_test = np.random.uniform(-1, 1, (test_n, input_dim))
    sincx_test = step_fun(x_test)
    y_test = sincx_test + np.random.normal(0, scale, size = test_n)

    # true_test_q = []
    # for a in alpha:
    #     true_test_q.append(sincx_test + norm.ppf(a, scale = scale))
    
    return {"data": x_train, "label": y_train}, {"data": x_valid, "label": y_valid}, {"data": x_test, "label": y_test}
# {"true_train_q": true_train_q}, {"true_valid_q": true_valid_q}, {"true_test_q": true_test_q}