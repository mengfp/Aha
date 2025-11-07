#!/usr/bin/env python3
#
# Copyright 2025 Meng, Fanping. All rights reserved.
#
import numpy as np
import pandas as pd
from codetiming import Timer
from sklearn.metrics import mean_squared_error
from aha import Model, Trainer

# 生成数据
# x: random.normal(0, 1)
# y: random.normal(0, 1)
# z: x * y + random.normal(0, 1)
# w: x + y + random.normal(0, 1)
def generate_data(size = 1000000):
    df = pd.DataFrame(np.random.normal(size=(size, 4)), columns=list('xyzw'))
    df['z'] += df['x'] * df['y']
    df['w'] += df['x'] + df['y']
    return df


# 训练模型
# rank: 圈数
# loop: 迭代次数
@Timer()
def train(df, rank, loop):
    print('Training model: rank =', rank, ', loop =', loop)
    model = Model(rank, 4)
    trainer = Trainer(model)
    for i in range(loop):
        trainer.Reset()
        trainer.BatchTrain(df.to_numpy())
        e = trainer.Update()
        print(i, ': entropy =', e)
    return model

# 测试模型性能，计算均方误差
@Timer()
def test(model, df):
    d = 0.0
    for row in df.to_numpy():
        r, z = model.Predict(row[:2])
        e1 = z[0] - row[2]
        e2 = z[1] - row[3]
        d += (e1 * e1 + e2 * e2) / 2
    d /= len(df)
    print('MSE =', d)
    return d

def test_ex(model, df):
    d = 0.0
    for row in df.to_numpy():
        r, z, cov = model.PredictEx(row[:2])
        e1 = z[0] - row[2]
        e2 = z[1] - row[3]
        d += (e1 * e1 + e2 * e2) / 2
    d /= len(df)
    print('MSE =', d)
    print('COV =', cov)
    return d

# 测试模型性能，计算均方误差
@Timer()
def batch_test(model, df):
    d = 0.0
    r, z = model.BatchPredict(df[['x', 'y']])
    d = mean_squared_error(df[['z', 'w']], z)
    print('MSE =', d)
    print('z[0].shape = ', z[0].shape)
    return d

# 测试模型性能，计算均方误差
@Timer()
def batch_test_ex(model, df):
    d = 0.0
    r, z, cov = model.BatchPredictEx(df[['x', 'y']])
    d = mean_squared_error(df[['z', 'w']], z)
    print('MSE =', d)
    print('z.shape = ', z.shape)
    print('cov.shape = ', cov.shape)
    print('cov =', cov)
    return d

# test = batch_test
# test = test_ex
test = batch_test_ex

# 主程序
if __name__ == "__main__":
    df = generate_data()

    # Rank 1
    model = train(df, 1, 5)
    test(model, df)
    print()

    # Rank 5
    model = train(df, 5, 50)
    test(model, df)
    print()

    # print('Model =', model.Export())