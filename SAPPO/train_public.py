import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from agent import PPOAgent
from env import Environment
import time
'''
读取public数据集进行训练
训练时使用tensorboard记录训练数据，可导出为csv数据
'''
# PPO超参数
GAMMA = 0.7
LAMBDA = 0.95
EPOCHS = 3
CLIP_EPS = 0.1
MAX_EPISODES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INIT_ENTROPY_COEF = 0.05
CHANGE_GRAPH = 1000
LR=0.003

DECISION_REGION = 4 # 决策区域长度
SUB_DECISION_REGION = 3 # 子决策区域
NODE_NUM = 101
COORD_MAX = 50

result_save_path = "SAPPO/result.png"
DATA_PATH = "data/eil101.tsp"

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

def compute_gae(rewards, values, next_value, dones):
    """计算GAE（Generalized Advantage Estimation）"""
    advantages = []
    gae = 0
    for reward, value, next_v, done in zip(reversed(rewards), reversed(values), reversed(next_value), reversed(dones)):
        delta = reward + GAMMA * next_v * (1 - done) - value
        gae = delta + GAMMA * LAMBDA * (1 - done) * gae
        advantages.insert(0, gae)
    return advantages

def get_entropy_coef(episode):
    # 计算线性递减部分
    ratio = (MAX_EPISODES - episode) / (MAX_EPISODES - 100)
    entropy_coef = ratio * (INIT_ENTROPY_COEF - 0.01) + 0.05
    return max(0.001, entropy_coef)

def train(visible=False):
    # 初始化环境和智能体
    timestamp = str(time.time())  # 获取时间戳并转为字符串
    short_name = timestamp[:9]    # 截取前9位
    writer = SummaryWriter(f'logs/SAPPO/{short_name}')  # 创建TensorBoard日志记录器
    env = Environment(coord_max=COORD_MAX, node_num=NODE_NUM, device=DEVICE, decision_region=DECISION_REGION, sub_decision_region=SUB_DECISION_REGION)
    agent = PPOAgent(node_feature_dim=3, hidden_dim=128, decision_region=DECISION_REGION, sub_decision_region=SUB_DECISION_REGION, num_heads=4, num_layers=2, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LR)
    best_model = 1000
    for episode in range(MAX_EPISODES):
        s_t = time.time()
        # 重置环境，获取初始观测和图数据
        if episode % CHANGE_GRAPH == 0:
            data = load_tsp_coords(DATA_PATH)
            obs, original_time = env.reset(data)
        else:
            obs, original_time = env.reset()
        total_reward = 0
        done = False
            
        # 单次交互数据存储
        episode_obs = []
        episode_actions = []
        episode_log_probs = []
        episode_rewards = []
        episode_dones = []
        episode_values = []

        while not done:
            # 将单次观测扩展为批量维度
            batch_obs = {key: val.unsqueeze(0).to(DEVICE) for key, val in obs.items()}

            # 前向推理
            with torch.no_grad():
                action, log_prob, value, entropy = agent(batch_obs)

            # 执行动作
            obs, reward, done, total_time = env.step(action.squeeze(0))
            total_reward += reward
            # 可视化
            if visible and episode > 100:
                env.draw()

            # 存储交互数据
            episode_obs.append({key: val.clone() for key, val in batch_obs.items()})
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_values.append(value)
        if total_time < 300: # 绘制论文中所需的图
            env.reset()
            env.save_instance_to_img()
            env.save_step_to_img()
            for action in episode_actions:
                env.step(action.squeeze(0))
                env.save_step_to_img()

        if total_time / original_time < best_model:
            best_model = total_time / original_time
            torch.save(agent.state_dict(), f'SAPPO/trained_models/{NODE_NUM}nodes_best_model_{DECISION_REGION}_{SUB_DECISION_REGION}.pth')
            env.save_result_to_img(result_save_path)

        # 计算GAE和回报
        with torch.no_grad():
            next_value = torch.tensor([0.0], device=DEVICE) if done else agent(batch_obs)[2]  # 取value
        advantages = compute_gae(episode_rewards, episode_values, [next_value] * len(episode_rewards), episode_dones)
        returns = [adv + val for adv, val in zip(advantages, episode_values)]

        # 转换为批量张量
        batch_obs = {key: torch.cat([m[key] for m in episode_obs], dim=0) for key in episode_obs[0]}
        batch_actions = torch.cat(episode_actions, dim=0)
        batch_log_probs = torch.cat(episode_log_probs, dim=0)
        batch_returns = torch.tensor(returns, dtype=torch.float, device=DEVICE)
        batch_values = torch.cat(episode_values, dim=0)
        batch_advantages = torch.tensor(advantages, dtype=torch.float, device=DEVICE)
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

        # PPO更新
        for _ in range(EPOCHS):
            new_log_probs, new_values, new_entropy = agent.evaluate(batch_obs, batch_actions)

            # 添加熵正则化项
            entropy_loss = new_entropy.mean()

            ratio = torch.exp(new_log_probs - batch_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (batch_returns - new_values).pow(2).mean()

            # entropy_coef = get_entropy_coef(episode)
            entropy_coef = INIT_ENTROPY_COEF
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        # 记录训练指标到TensorBoard
        writer.add_scalar('Coef/entropy', entropy_coef, episode)
        writer.add_scalar('Loss/critic', critic_loss.item(), episode)
        writer.add_scalar('Loss/entropy', entropy_loss.item(), episode)
        writer.add_scalar('Reward/total', total_reward, episode)
        writer.add_scalar('Time/total', total_time, episode)
        
        e_t = time.time() - s_t
        print(f"Episode {episode + 1}/{MAX_EPISODES}，总奖励：{total_reward:.2f}, 路径耗时: {total_time:.2f}, 原时间：{original_time:.2f}，训练一轮耗时：{e_t:.2f}，熵{entropy_loss:.2f}")

    writer.close()  # 关闭TensorBoard日志记录器
    print(f"基于{NODE_NUM}节点环境，决策区域为{DECISION_REGION}，子决策区域为{SUB_DECISION_REGION}，训练完成！")

if __name__ == "__main__":
    train(False)
