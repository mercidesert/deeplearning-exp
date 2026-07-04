import numpy as np
import matplotlib.pyplot as plt

class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # 维度参数初始化设置
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # 初始化权重和偏置（使用小随机数，避免梯度爆炸）
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
    
    def forward(self, x_sequence, h_prev=None):
        T = len(x_sequence)
        h_states = []
        y_preds = []
        if h_prev is None:
            h_prev = np.zeros((self.hidden_size, 1))#未传入前一隐藏状态时采用全0隐藏状态
        h = h_prev
        for t in range(T):
            x = x_sequence[t] 
            # RNN核心公式: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            a = np.dot(self.W_xh, x) + np.dot(self.W_hh, h) + self.b_h
            h = np.tanh(a)  #激活函数层
            # 输出层: y_t = softmax(W_hy * h_t + b_y)
            b = np.dot(self.W_hy, h) + self.b_y
            # 数值稳定性：减去最大值
            exp_b = np.exp(b - np.max(b))
            y_pred = exp_b / np.sum(exp_b)  # softmax
            
            h_states.append(h.copy())
            y_preds.append(y_pred)
        
        return h_states, y_preds
    
    def bptt(self, x_sequence, y_true_sequence, h_states):
        T = len(x_sequence)
        # 初始化梯度（与权重形状相同）
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # 初始化沿时间反向传播的梯度（dh_next）
        dh_next = np.zeros((self.hidden_size, 1))
        
        # 从最后一个时间步反向遍历
        for t in reversed(range(T)):
            x_t = x_sequence[t]
            h_t = h_states[t]
            y_true = y_true_sequence[t]
            dy = h_states[t]  
            b = np.dot(self.W_hy, h_t) + self.b_y
            exp_b = np.exp(b - np.max(b))
            y_pred = exp_b / np.sum(exp_b)
            dL_db = y_pred - y_true  
            # 输出层权重梯度
            dW_hy += np.dot(dL_db, h_t.T)
            db_y += dL_db
            #传播到隐藏层
            #dL/dh_t = W_hy^T * dL/db
            dh = np.dot(self.W_hy.T, dL_db) + dh_next
            da = dh * (1 - np.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_t) + self.b_h)**2)
            #权重梯度
            dW_xh += np.dot(da, x_t.T)
            dW_hh += np.dot(da, h_t.T)
            db_h += da
            #计算传递到上一时间步的 dh_next，dh_{t-1} = (W_hh^T * da)
            dh_next = np.dot(self.W_hh.T, da)
        grads = {
            'W_xh': dW_xh, 'W_hh': dW_hh, 'W_hy': dW_hy,
            'b_h': db_h, 'b_y': db_y
        }
        return grads
    
    def update_parameters(self, grads):
        """梯度下降更新参数"""
        self.W_xh -= self.lr * grads['W_xh']
        self.W_hh -= self.lr * grads['W_hh']
        self.W_hy -= self.lr * grads['W_hy']
        self.b_h -= self.lr * grads['b_h']
        self.b_y -= self.lr * grads['b_y']
    
    def train_step(self, x_seq, y_seq):
        #对于单个batch进行全部训练步骤
        #1.前向传播
        h_states, y_preds = self.forward(x_seq)
        #2。反向传播
        grads = self.bptt(x_seq, y_seq, h_states)
        #3.更新参数
        self.update_parameters(grads)
        #计算损失（交叉熵）
        loss = 0
        for t in range(len(y_seq)):
            loss -= np.log(y_preds[t][np.argmax(y_seq[t])] + 1e-8)
        return loss / len(y_seq)


# ===================== 示例：训练RNN做二进制加法 =====================
def binary_addition_example():
    """使用RNN学习二进制加法（经典的RNN教学示例）"""
    
    # 生成二进制加法数据
    def int_to_binary(n, bits=8):
        return [int(b) for b in format(n, f'0{bits}b')]
    
    def binary_to_int(bits):
        return int(''.join(map(str, bits)), 2)
    
    # 生成一个样本序列：两个8位二进制数相加，带进位
    def generate_sample(bits=8):
        a = np.random.randint(0, 2**bits // 2)
        b = np.random.randint(0, 2**bits // 2)
        c = a + b
        
        a_bits = int_to_binary(a, bits)
        b_bits = int_to_binary(b, bits)
        c_bits = int_to_binary(c, bits)
        
        # 输入序列：每个时间步输入 (a_bit, b_bit, carry_in)
        # 输出：预测当前位的和（0或1）
        x_seq = []
        y_seq = []
        carry = 0
        for i in range(bits-1, -1, -1):  # 从最低位开始
            bit_a = a_bits[i]
            bit_b = b_bits[i]
            total = bit_a + bit_b + carry
            out_bit = total % 2
            carry = total // 2
            
            # 输入: [bit_a, bit_b, carry_in]  -> 3维
            x_seq.append(np.array([[bit_a], [bit_b], [carry]]))
            # 输出: one-hot [0,1] 或 [1,0]
            y_onehot = np.zeros((2, 1))
            y_onehot[out_bit] = 1
            y_seq.append(y_onehot)
        
        return x_seq, y_seq, (a, b, a+b)
    
    # 创建RNN: 输入维度3，隐藏层10，输出维度2
    rnn = VanillaRNN(input_size=3, hidden_size=10, output_size=2, learning_rate=0.1)
    
    # 训练
    print("开始训练二进制加法...")
    losses = []
    for epoch in range(2000):
        total_loss = 0
        for _ in range(100):  # 每个epoch 100个样本
            x_seq, y_seq, _ = generate_sample(8)
            loss = rnn.train_step(x_seq, y_seq)
            total_loss += loss
        avg_loss = total_loss / 100
        losses.append(avg_loss)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    # 测试
    print("\n测试结果:")
    for _ in range(5):
        x_seq, y_seq, (a, b, true_sum) = generate_sample(8)
        _, preds = rnn.forward(x_seq)
        pred_bits = [np.argmax(p) for p in preds]
        pred_sum = binary_to_int(pred_bits[::-1])  # 反转回正常顺序
        print(f"{a} + {b} = {true_sum} (预测: {pred_sum})")
    
    # 绘制损失曲线
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Training on Binary Addition')
    plt.show()

if __name__ == "__main__":
    binary_addition_example()