import numpy as np
import torch
import os
import pickle

NODE_NUMS = 500
DATA_PATH = f"data/nodes{NODE_NUMS}.pkl"
INSTANS_NUMS = 10000

def load_data(path, offset, num_samples):
    """加载数据集"""
    assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
    with open(path, 'rb') as f:
        data = pickle.load(f)[offset: offset + num_samples]
    node_xy_list = [sample[0] for sample in data]
    node_xy_tensor = torch.tensor(node_xy_list)
    return node_xy_tensor

def generate_synthetic_data(output_path, num_instances, num_nodes, x_range, y_range):
    """生成指定结构的合成数据并保存为.pkl文件"""
    data = []
    
    for _ in range(num_instances):
        # 随机生成num_nodes个节点的xy坐标
        node_xy = np.random.uniform(low=[x_range[0], y_range[0]], 
                                    high=[x_range[1], y_range[1]], 
                                    size=(num_nodes, 2)).astype(np.float32)  # 生成形状为 (num_nodes, 2) 的坐标数组
        # 将生成的xy坐标打包成元组，并添加到数据列表
        data.append((node_xy, ))  # 使用元组来保持原始数据格式

    # 保存生成的数据到指定路径
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

# 用法示例
generate_synthetic_data(DATA_PATH, INSTANS_NUMS, NODE_NUMS, x_range=(0, 50), y_range=(0, 50))
