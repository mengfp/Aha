#!/usr/bin/env python3
#
# Copyright 2025 Meng, Fanping. All rights reserved.
#
import numpy as np
import pandas as pd
from codetiming import Timer
from sklearn.metrics import mean_squared_error
from aha import Model32 as Model, Trainer32 as Trainer

# 生成数据
# x: random.normal(0, 1)
# y: random.normal(0, 1)
# z: x * y + random.normal(0, 1)
def generate_data(size = 1000000):
    df = pd.DataFrame(np.random.normal(size=(size, 3)), columns=list('xyz'))
    df['z'] += df['x'] * df['y']
    return df


# 训练模型
# rank: 圈数
# loop: 迭代次数
@Timer()
def train(df, rank, loop):
    print('Training model: rank =', rank, ', loop =', loop)
    model = Model(rank, 3)
    trainer = Trainer(model)
    for i in range(loop):
        for row in df.to_numpy():
            trainer.Train(row)
        e = trainer.Update()
        print(i, ': entropy =', e)
    return model

# 训练模型
# rank: 圈数
# loop: 迭代次数
@Timer()
def batch_train(df, rank, loop):
    print('Training model: rank =', rank, ', loop =', loop)
    model = Model(rank, 3)
    trainer = Trainer(model)
    for i in range(loop):
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
        e = z[0] - row[2]
        d += e * e
    d /= len(df)
    print('MSE =', d)
    return d

# 测试模型性能，计算均方误差
@Timer()
def batch_test(model, df):
    d = 0.0
    r, z = model.BatchPredict(df[['x', 'y']])
    d = mean_squared_error(df['z'], z)
    print('MSE =', d)
    return d

train = batch_train
test = batch_test

# 主程序
if __name__ == "__main__":
    df = generate_data()

    # Rank 1
    model = train(df, 1, 10)
    test(model, df)
    print()

    # Rank 2
    model = train(df, 2, 20)
    test(model, df)
    print()

    # Rank 3
    model = train(df, 3, 20)
    test(model, df)
    print()

    # Rank 4
    model = train(df, 4, 20)
    test(model, df)
    print()
    
    # Rank 5
    model = train(df, 5, 20)
    test(model, df)
    print()

    print('Model =', model.Export())