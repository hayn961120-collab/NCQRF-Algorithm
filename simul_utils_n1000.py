import numpy as np
from sklearn.metrics import mean_pinball_loss
from scipy.stats import t, skewnorm, laplace_asymmetric, genextreme, norm
import scipy

def gen_simul1(train_n, input_dim=1, scale=0.1, seed=42):
    np.random.seed(seed)
    # 학습 데이터 생성
    x_train = np.random.uniform(-1, 1, (train_n, input_dim))
    sincx = np.sin(np.pi * x_train[:, 0]) / (np.pi * x_train[:, 0])
    Z = np.reshape(sincx, (train_n, 1))

    ep = genextreme.rvs(c=-1/2, scale=scale * np.exp(1 - x_train[:, 0]), size=train_n)[:, np.newaxis]
    y_train = Z + ep

    # 검증 데이터 생성
    valid_n = 500
    x_valid = np.random.uniform(-1, 1, (valid_n, input_dim))
    sincx_valid = np.sin(np.pi * x_valid[:, 0]) / (np.pi * x_valid[:, 0])
    y_valid = np.reshape(sincx_valid, (valid_n, 1)) + genextreme.rvs(
        c=-1/2, scale=scale * np.exp(1 - x_valid[:, 0]), size=valid_n
    )[:, np.newaxis]

    # 테스트 데이터 생성
    test_n = 500
    x_test = np.random.uniform(-1, 1, (test_n, input_dim))
    sincx_test = np.sin(np.pi * x_test[:, 0]) / (np.pi * x_test[:, 0])
    y_test = np.reshape(sincx_test, (test_n, 1)) + genextreme.rvs(
        c=-1/2, scale=scale * np.exp(1 - x_test[:, 0]), size=test_n
    )[:, np.newaxis]

    return {"data": x_train, "label": y_train}, {"data": x_valid, "label": y_valid}, {"data": x_test, "label": y_test}


def gen_simul2(train_n, input_dim=1, scale=0.1, seed=42):
    np.random.seed(seed)
    # 학습 데이터 생성
    x_train = np.random.uniform(-1, 1, (train_n, input_dim))
    sincx = np.sin(np.pi * x_train[:, 0]) / (np.pi * x_train[:, 0])
    Z = np.reshape(sincx, (train_n, 1))

    ep = np.random.normal(0, scale * np.exp(1 - x_train[:, 0])[:, np.newaxis], size=(train_n, 1))
    y_train = 10 * scale * Z + ep

    # 검증 데이터 생성
    valid_n = 500
    x_valid = np.random.uniform(-1, 1, (valid_n, input_dim))
    sincx_valid = np.sin(np.pi * x_valid[:, 0]) / (np.pi * x_valid[:, 0])
    y_valid = 10 * scale * np.reshape(sincx_valid, (valid_n, 1)) + np.random.normal(
        0, scale * np.exp(1 - x_valid[:, 0])[:, np.newaxis], (valid_n, 1)
    )

    # 테스트 데이터 생성
    test_n = 500
    x_test = np.random.uniform(-1, 1, (test_n, input_dim))
    sincx_test = np.sin(np.pi * x_test[:, 0]) / (np.pi * x_test[:, 0])
    y_test = 10 * scale * np.reshape(sincx_test, (test_n, 1)) + np.random.normal(
        0, scale * np.exp(1 - x_test[:, 0])[:, np.newaxis], (test_n, 1)
    )

    return {"data": x_train, "label": y_train}, {"data": x_valid, "label": y_valid}, {"data": x_test, "label": y_test}


def gen_simul4(train_n, input_dim=40, seed=42):
    np.random.seed(seed)
    # 학습 데이터 생성
    x_train = np.random.uniform(-1,1, (train_n, input_dim))
    sigma = 1 + (x_train[:, 0] > 0).astype(float)
    ep = np.random.normal(0, sigma[:, np.newaxis], size = (train_n, 1))
    
    Z = np.zeros((train_n, 1))
    y_train = Z + ep
    
    # 검증 데이터 생성
    valid_n = 500
    x_valid = np.random.uniform(-1, 1, (valid_n, input_dim))
    
    sigma_valid = 1 + (x_valid[:, 0] > 0).astype(float)
    y_valid = np.random.normal(0, sigma_valid[:, np.newaxis], size = (valid_n, 1))
    
    # 테스트 데이터 생성
    test_n = 500
    x_test = np.random.uniform(-1, 1, (test_n, input_dim))
    
    sigma_test = 1 + (x_test[:, 0] > 0).astype(float)
    y_test = np.random.normal(0, sigma_test[:, np.newaxis], size = (test_n, 1))
    
    return {"data": x_train, "label": y_train}, {"data": x_valid, "label": y_valid}, {"data": x_test, "label": y_test}
