import torch
import os
import pickle
import csv
import time
from agent import PPOAgent
from env import Environment
'''
加载已训练好的模型
读取公共数据集进行推理，并生成结果记录
'''
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DECISION_REGION = 4  # 决策区域长度
SUB_DECISION_REGION = 3  # 子决策区域

NODE_NUM = 150
DATASET = 'ch150'

COORD_MAX = 50
INSTANS_NUMS = 1
GREEDY = True
CHANGE_GRAPH = 1 if GREEDY else 10

DATA_PATH = f"data/{DATASET}.tsp"  # 数据集路径
CSV_PATH = f"SAPPO/result_info/{'Greedy' if GREEDY else 'Sample'}_{DATASET}_result.csv"  # CSV 输出路径

# 模型路径
MODEL_PATH = f"SAPPO/trained_models/100nodes_best_model_{DECISION_REGION}_{SUB_DECISION_REGION}.pth"

def load_tsp_coords(path):
    """从 .tsp 文件中提取坐标并转换为 (num_nodes, 2) 的 torch.Tensor"""
    coords = []
    with open(path, 'r') as f:
        lines = f.readlines()

    # 找到 NODE_COORD_SECTION 的起始行
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "NODE_COORD_SECTION":
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("NODE_COORD_SECTION not found in file.")

    # 从 NODE_COORD_SECTION 开始读取坐标，直到 EOF 或空行
    for line in lines[start_idx:]:
        line = line.strip()
        if line == "EOF" or line == "":
            break
        parts = line.split()
        if len(parts) >= 3:
            x = float(parts[1])
            y = float(parts[2])
            coords.append([x, y])

    return torch.tensor(coords, dtype=torch.float32)

def infer(visible=False):
    # 初始化环境和智能体
    env = Environment(coord_max=COORD_MAX, node_num=NODE_NUM, device=DEVICE, 
                      decision_region=DECISION_REGION, sub_decision_region=SUB_DECISION_REGION)
    agent = PPOAgent(node_feature_dim=3, hidden_dim=128, decision_region=DECISION_REGION, 
                     sub_decision_region=SUB_DECISION_REGION, num_heads=4, num_layers=2, device=DEVICE).to(DEVICE)
    
    # 加载训练好的模型
    agent.load_state_dict(torch.load(MODEL_PATH))
    agent.eval()  # 设置为评估模式

    # 初始化 CSV 文件
    csv_header = ["episode", "path_time", "original_time", "infer_time", "total_reward"]
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    max_episodes = INSTANS_NUMS * CHANGE_GRAPH

    for episode in range(max_episodes):
        s_t = time.time()

        # 使用当前数据重置环境
        if episode % CHANGE_GRAPH == 0:
            data = load_tsp_coords(DATA_PATH)
            obs, original_time = env.reset(data)
        else:
            obs, original_time = env.reset()

        total_reward = 0
        done = False
        episode_actions = []

        while not done:
            # 将单次观测扩展为批量维度
            batch_obs = {key: val.unsqueeze(0).to(DEVICE) for key, val in obs.items()}

            # 前向推理（通过注释切换贪婪或采样）
            with torch.no_grad():
                if GREEDY:
                    # 贪婪策略
                    action = agent.greedy_forward(batch_obs)
                else:
                    # 采样策略
                    action, _, _, _ = agent.forward(batch_obs)

            # 执行动作
            obs, reward, done, total_time = env.step(action.squeeze(0))
            total_reward += reward
            episode_actions.append(action)

            # 可视化（可选）
            if visible:
                env.draw()

        # 计算推理耗时
        infer_time = time.time() - s_t

        # 将结果写入 CSV
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, total_time.item(), original_time, infer_time, total_reward.item()])

        # 打印信息
        print(f"Episode {episode + 1}/{max_episodes}, 路径耗时: {total_time:.2f}, "
              f"原时间: {original_time:.2f}, 推理耗时: {infer_time:.2f}, 总奖励: {total_reward:.2f}")

    print(f"当前设置{NODE_NUM}个节点，决策区域长度为{DECISION_REGION}，子决策区域长度为{SUB_DECISION_REGION}。{'Greedy' if GREEDY else 'Sample'}模式推理完成！结果已保存至", CSV_PATH)

if __name__ == "__main__":
    infer(False)