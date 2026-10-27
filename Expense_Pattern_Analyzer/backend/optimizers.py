"""SGD and Adam implemented from scratch. Both update the params dict in place."""
import numpy as np


class SGD:
    """Plain gradient descent step: w = w - lr * grad."""
    name = "sgd"

    def __init__(self, learning_rate=0.05):
        self.lr = float(learning_rate)

    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]


class Adam:
    """Adam: running means of gradient (m) and squared gradient (v) + bias correction."""
    name = "adam"

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = float(learning_rate)
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(name, learning_rate):
    name = str(name).lower()
    if name == "sgd":
        return SGD(learning_rate)
    if name == "adam":
        return Adam(learning_rate)
    raise ValueError("optimizer must be 'sgd' or 'adam'")
