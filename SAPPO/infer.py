import torch
import os
import pickle
import csv
import time
from agent import PPOAgent
from env import Environment
'''
加载已训练好的模型
读取数据进行推理，并生成结果记录
'''
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DECISION_REGION = 4  # 决策区域长度
SUB_DECISION_REGION = 3  # 子决策区域
NODE_NUM = 50
COORD_MAX = 50
INSTANS_NUMS = 100
GREEDY = False
CHANGE_GRAPH = 1 if GREEDY else 50

DATA_PATH = f"data/nodes{NODE_NUM}.pkl"  # 数据集路径
CSV_PATH = f"SAPPO/result_info/{'Greedy' if GREEDY else 'Sample'}_{NODE_NUM}nodes_{DECISION_REGION}_{SUB_DECISION_REGION}_result.csv"  # CSV 输出路径

# 模型路径
MODEL_PATH = f"SAPPO/trained_models/100nodes_best_model_{DECISION_REGION}_{SUB_DECISION_REGION}.pth"

result_save_path = "SAPPO/result.png"
def load_data(path, offset, num_samples):
    """加载数据集"""
    assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
    with open(path, 'rb') as f:
        data = pickle.load(f)[offset: offset + num_samples]
    node_xy_list = [sample[0] for sample in data]
    node_xy_tensor = torch.tensor(node_xy_list)
    return node_xy_tensor

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

    loaded_data = load_data(DATA_PATH, 0, INSTANS_NUMS)
    max_episodes = INSTANS_NUMS * CHANGE_GRAPH
    best_model = 1000

    for episode in range(max_episodes):
        s_t = time.time()

        # 使用当前数据重置环境
        if episode % CHANGE_GRAPH == 0:
            # data = COORD_MAX * torch.rand((NODE_NUM, 2), dtype=torch.float)
            data = loaded_data[episode // CHANGE_GRAPH]
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
        if total_time / original_time < best_model:
            best_model = total_time / original_time
            env.save_result_to_img(result_save_path)

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