"""
CS231n 模块化全连接神经网络
核心实现：层、网络、优化器、训练器
"""

import numpy as np


# ==================== 1. 基础层实现 ====================

def affine_forward(x, w, b):
    """全连接层前向：out = x·w + b"""
    out = np.dot(x.reshape(x.shape[0], -1), w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """全连接层反向：计算dx, dw, db"""
    x, w, b = cache
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(x.shape[0], -1).T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """ReLU激活前向：max(0, x)"""
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """ReLU激活反向：梯度只传给x>0的部分"""
    x = cache
    dx = dout * (x > 0)
    return dx


def affine_relu_forward(x, w, b):
    """组合层：Affine + ReLU"""
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """组合层反向"""
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def softmax_loss(x, y):
    """Softmax + 交叉熵损失"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(len(y)), y]))
    dx = probs
    dx[np.arange(len(y)), y] -= 1
    dx /= len(y)
    return loss, dx


# ==================== 2. 多层全连接网络 ====================

class FullyConnectedNet(object):
    
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 reg=0.0, weight_scale=1e-2, dtype=np.float32):
        self.reg = reg
        self.dtype = dtype
        
        # 网络结构：[input_dim] + hidden_dims + [num_classes]
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        self.params = {}
        
        # 初始化权重和偏置
        for i in range(len(layer_dims) - 1):
            self.params[f'W{i+1}'] = np.random.randn(layer_dims[i], layer_dims[i+1]) * weight_scale
            self.params[f'b{i+1}'] = np.zeros(layer_dims[i+1])
    
    def loss(self, X, y=None):
        # 前向传播
        caches = []
        out = X
        num_layers = len(self.params) // 2
        
        for i in range(1, num_layers):
            out, cache = affine_relu_forward(out, self.params[f'W{i}'], self.params[f'b{i}'])
            caches.append(cache)
        
        # 最后一层无ReLU
        scores, cache = affine_forward(out, self.params[f'W{num_layers}'], self.params[f'b{num_layers}'])
        caches.append(cache)
        
        # 测试模式：只返回得分
        if y is None:
            return scores
        
        # 训练模式：计算损失和梯度
        loss, dscores = softmax_loss(scores, y)
        
        # 添加正则化损失
        for i in range(1, num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params[f'W{i}'] ** 2)
        
        # 反向传播
        grads = {}
        dout = dscores
        dout, dw, db = affine_backward(dout, caches.pop())
        grads[f'W{num_layers}'] = dw + self.reg * self.params[f'W{num_layers}']
        grads[f'b{num_layers}'] = db
        
        for i in range(num_layers - 1, 0, -1):
            dout, dw, db = affine_relu_backward(dout, caches.pop())
            grads[f'W{i}'] = dw + self.reg * self.params[f'W{i}']
            grads[f'b{i}'] = db
        
        return loss, grads


# ==================== 3. 优化器 ====================

def sgd(w, dw, config):
    """随机梯度下降"""
    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config):
    """SGD + 动量"""
    v = config.get('velocity', np.zeros_like(w))
    mu = config.get('momentum', 0.9)
    lr = config['learning_rate']
    
    v = mu * v - lr * dw
    w += v
    
    config['velocity'] = v
    return w, config


def rmsprop(w, dw, config):
    """RMSProp：自适应学习率"""
    cache = config.get('cache', np.zeros_like(w))
    decay = config.get('decay_rate', 0.99)
    eps = config.get('epsilon', 1e-8)
    lr = config['learning_rate']
    
    cache = decay * cache + (1 - decay) * (dw ** 2)
    w -= lr * dw / (np.sqrt(cache) + eps)
    
    config['cache'] = cache
    return w, config


def adam(w, dw, config):
    """Adam：Momentum + RMSProp"""
    m = config.get('m', np.zeros_like(w))
    v = config.get('v', np.zeros_like(w))
    t = config.get('t', 0) + 1
    lr = config['learning_rate']
    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.999)
    eps = config.get('epsilon', 1e-8)
    
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    config['m'] = m
    config['v'] = v
    config['t'] = t
    return w, config


# ==================== 4. 训练器 ====================

class Solver(object):
    """封装训练循环"""
    
    def __init__(self, model, data, update_rule='sgd', optim_config={'learning_rate': 1e-3},
                 lr_decay=1.0, batch_size=100, num_epochs=10, print_every=100):
        self.model = model
        self.X_train, self.y_train = data['X_train'], data['y_train']
        self.X_val, self.y_val = data['X_val'], data['y_val']
        self.update_rule = globals()[update_rule]
        self.optim_configs = {k: optim_config.copy() for k in model.params}
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.print_every = print_every
        
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
    
    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        
        for t in range(num_iterations):
            # 采样batch
            batch_mask = np.random.choice(num_train, self.batch_size)
            X_batch = self.X_train[batch_mask]
            y_batch = self.y_train[batch_mask]
            
            # 前向+反向
            loss, grads = self.model.loss(X_batch, y_batch)
            self.loss_history.append(loss)
            
            # 更新参数
            for p, w in self.model.params.items():
                dw = grads[p]
                config = self.optim_configs[p]
                new_w, new_config = self.update_rule(w, dw, config)
                self.model.params[p] = new_w
                self.optim_configs[p] = new_config
            
            # Epoch结束：验证、学习率衰减
            if (t+1) % iterations_per_epoch == 0:
                epoch = (t+1) // iterations_per_epoch
                train_acc = self.check_accuracy(self.X_train, self.y_train)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                print(f'(Epoch {epoch} / {self.num_epochs}) train acc: {train_acc:.4f}; val_acc: {val_acc:.4f}')
                
                # 学习率衰减
                for config in self.optim_configs.values():
                    config['learning_rate'] *= self.lr_decay
    
    def check_accuracy(self, X, y, num_samples=None):
        if num_samples and X.shape[0] > num_samples:
            mask = np.random.choice(X.shape[0], num_samples)
            X, y = X[mask], y[mask]
        scores = self.model.loss(X)
        preds = np.argmax(scores, axis=1)
        return np.mean(preds == y)


# ==================== 5. 数据加载（CIFAR-10） ====================

def get_cifar10_data(num_training=49000, num_val=1000, num_test=1000):
    """加载并预处理CIFAR-10"""
    # 这里假设数据已下载，实际需调用load_cifar10()
    # 数据形状：(N, 3, 32, 32)  → 需展平为 (N, 3072)
    
    # 模拟数据结构
    class Data: pass
    data = Data()
    # 实际使用时替换为真实数据
    data.X_train = np.random.randn(num_training, 3*32*32)
    data.y_train = np.random.randint(0, 10, num_training)
    data.X_val = np.random.randn(num_val, 3*32*32)
    data.y_val = np.random.randint(0, 10, num_val)
    data.X_test = np.random.randn(num_test, 3*32*32)
    data.y_test = np.random.randint(0, 10, num_test)
    return data


# ==================== 6. 使用示例 ====================

def demo():
    #加载数据
    data = get_cifar10_data()
    
    #创建3层网络
    model = FullyConnectedNet([100, 100], weight_scale=5e-2, reg=1e-3)
    
    #创建训练器
    solver = Solver(model, data,
                    update_rule='adam',
                    optim_config={'learning_rate': 1e-3},
                    lr_decay=0.95,
                    batch_size=200,
                    num_epochs=10,
                    print_every=100)
    solver.train()
    
    print(f"\nBest validation accuracy: {max(solver.val_acc_history):.4f}")
    return solver


if __name__ == '__main__':
    solver = demo()