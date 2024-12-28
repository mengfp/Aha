import numpy as np
import pandas as pd
from aha import Model, Trainer

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
def train(rank, loop):
    print('Training model: rank =', rank, ', loop =', loop)
    model = Model(rank, 3)
    trainer = Trainer(model)
    for i in range(loop):
        df = generate_data()
        trainer.Reset()
        for row in df.to_numpy():
            trainer.Train(row)
        trainer.Update()
        print(i, ': entropy =', trainer.Entropy())
    return model


# 测试模型性能，计算均方误差
def test(model):
    d = 0.0
    df = generate_data()
    for row in df.to_numpy():
        r, z = model.Predict(row[:2])
        e = z[0] - row[2]
        d += e * e
    d /= len(df)
    print('MSE =', d)
    return d


# 主程序
if __name__ == "__main__":
    # Rank 1
    model = train(1, 5)
    test(model)

    # Rank 2
    model = train(2, 20)
    test(model)

    # Rank 3
    model = train(3, 20)
    test(model)

    # Rank 4
    model = train(4, 20)
    test(model)
    
    # Rank 5
    model = train(5, 20)
    test(model)

    print('Model =', model.Export())