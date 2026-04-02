import pickle
import numpy as np
import os
import urllib.request
import tarfile

def download_cifar10(dest_dir):
    """自动下载并解压 CIFAR-10 数据集"""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(dest_dir, "cifar-10-python.tar.gz")
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    if not os.path.exists(tar_path):
        print("正在从多伦多大学服务器下载 CIFAR-10 数据集 (~162MB)，请耐心等待...")
        urllib.request.urlretrieve(url, tar_path)
        print("下载完成！正在解压...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print("解压完毕！")

def load_CIFAR_batch(filename):
    """读取单个 CIFAR-10 批次文件"""
    with open(filename, 'rb') as f:
        # Python 3 需要加上 encoding='latin1'
        datadict = pickle.load(f, encoding='latin1') 
        X = datadict['data']
        Y = datadict['labels']
        # 把 (10000, 3072) 的数组还原成 (10000, 32, 32, 3) 的彩色图像格式
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """读取全部 CIFAR-10 数据"""
    download_cifar10(ROOT) # 检查并自动下载
    base_dir = os.path.join(ROOT, 'cifar-10-batches-py')
    
    xs, ys = [],[]
    # 读取 5 个训练批次
    for b in range(1, 6):
        f = os.path.join(base_dir, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    # 读取 1 个测试批次
    Xte, Yte = load_CIFAR_batch(os.path.join(base_dir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte