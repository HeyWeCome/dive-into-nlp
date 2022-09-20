import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化参数
        W1 = 0.01 * np.random.randn(input_size, hidden_size)
        b1 = np.random.randn(hidden_size)
        W2 = 0.01 * np.random.randn(hidden_size, output_size)
        b2 = np.random.randn(output_size)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 定义loss
        self.loss_layer = SoftmaxWithLoss()

        # 整理所有的权重和梯度到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # 推理
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # 正向传播
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        # 反向传播
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
