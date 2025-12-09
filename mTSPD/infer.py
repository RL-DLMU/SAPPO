import torch
import os
import pickle
import csv
import time
from agent import PPOAgent
from env import Environment
from VRP import vrp_clarke_wright
'''
加载已训练好的模型
读取数据进行推理，并生成结果记录
现在整合VRP算法：使用VRP生成多个路由，每条路由视为一个TSP路径，然后逐个应用TSPD修改。由于仅作扩展性测试，因此VRP算法较为简单，各条路线生成可能不太完美，实际使用中用更好的算法替换掉clarke-wright即可。
'''
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DECISION_REGION = 4  # 决策区域长度
SUB_DECISION_REGION = 3  # 子决策区域
NODE_NUM = 50
COORD_MAX = 50
INSTANS_NUMS = 100
GREEDY = False
CHANGE_GRAPH = 1
NUM_VEHICLES = 3  # VRP车辆数量（也就是多少条路线），可根据需要调整

DATA_PATH = f"data/nodes{NODE_NUM}.pkl"  # 数据集路径
CSV_PATH = f"mTSPD/result_info/{'Greedy' if GREEDY else 'Sample'}_{NODE_NUM}nodes_{DECISION_REGION}_{SUB_DECISION_REGION}_vrp_result.csv"  # CSV 输出路径

# 模型路径
MODEL_PATH = f"trained_models/best_model_{DECISION_REGION}_{SUB_DECISION_REGION}.pth"

result_save_path = "mTSPD/result_vrp.svg"

def load_data(path, offset, num_samples):
    """加载数据集"""
    assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
    with open(path, 'rb') as f:
        data = pickle.load(f)[offset: offset + num_samples]
    node_xy_list = [sample[0] for sample in data]
    node_xy_tensor = torch.tensor(node_xy_list)
    return node_xy_tensor

def infer(visible=False):
    # 初始化智能体
    agent = PPOAgent(node_feature_dim=3, hidden_dim=128, decision_region=DECISION_REGION, 
                     sub_decision_region=SUB_DECISION_REGION, num_heads=4, num_layers=2, device=DEVICE).to(DEVICE)
    
    # 加载训练好的模型
    agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    agent.eval()  # 设置为评估模式

    # 初始化 CSV 文件
    csv_header = ["episode", "vehicle", "path_time", "original_time", "infer_time", "total_reward"]
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    loaded_data = load_data(DATA_PATH, 0, INSTANS_NUMS)
    max_episodes = INSTANS_NUMS * CHANGE_GRAPH
    best_model = 1000

    for episode in range(max_episodes):
        s_t = time.time()

        # 加载当前实例的数据
        if episode % CHANGE_GRAPH == 0:
            # data_np = COORD_MAX * torch.rand((NODE_NUM, 2), dtype=torch.float).numpy() # 使用随机生成的数据
            data_np = loaded_data[14].numpy()  # 转换为numpy以兼容VRP
            # 使用VRP算法生成多个路由
            vrp_routes = vrp_clarke_wright(data_np, NUM_VEHICLES)
            # 转换为torch张量
            coordinates = torch.cat([torch.tensor(data_np), torch.zeros((NODE_NUM, 1))], dim=1).to(DEVICE)

        total_reward_sum = 0
        episode_infer_time = 0
        route_original_times = []
        route_modified_times = []
        all_edge_index = []
        all_edge_attr = []

        for vehicle_idx, vrp_route_info in enumerate(vrp_routes):
            route_dist, route = vrp_route_info
            node_num = len(route)
            # 创建环境实例，仅处理当前路由
            env = Environment(coord_max=COORD_MAX, node_num=node_num, device=DEVICE, 
                              decision_region=DECISION_REGION, sub_decision_region=SUB_DECISION_REGION)
            
            # 重置环境，提供coordinates和当前route
            obs, original_time = env.reset(coordinates=coordinates, route=route)
            route_original_times.append(original_time)

            total_reward = 0
            done = False
            route_s_t = time.time()

            while not done:
                # 将单次观测扩展为批量维度
                batch_obs = {key: val.unsqueeze(0).to(DEVICE) for key, val in obs.items()}

                # 前向推理
                with torch.no_grad():
                    if GREEDY:
                        action = agent.forward(batch_obs)
                    else:
                        action, _, _, _ = agent.forward(batch_obs)

                # 执行动作
                obs, reward, done, total_time = env.step(action.squeeze(0))
                total_reward += reward

            route_infer_time = time.time() - route_s_t
            episode_infer_time += route_infer_time
            route_modified_times.append(env.time_to_now)
            total_reward_sum += total_reward

            # 收集边信息用于最终绘图
            all_edge_index.append(env.edge_index)
            all_edge_attr.append(env.edge_attr)

            # 写入CSV（每条路由一行）
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode + 1, vehicle_idx + 1, env.time_to_now.item(), original_time, route_infer_time, total_reward.item()])

        # 计算总时间：VRP中总时间为最大路径时间
        total_original_time = max(route_original_times)
        total_modified_time = max(route_modified_times)
        total_infer_time = time.time() - s_t

        # 合并所有边的index和attr
        combined_edge_index = torch.cat(all_edge_index, dim=1)
        combined_edge_attr = torch.cat(all_edge_attr, dim=0)
        save_combined_result_to_img(coordinates.cpu(), combined_edge_index.cpu(), combined_edge_attr.cpu(), vrp_routes, result_save_path)
        time.sleep(1)

        # 打印信息（总计）
        print(f"Episode {episode + 1}/{max_episodes}, 总路径耗时: {total_modified_time:.2f}, "
              f"原总时间: {total_original_time:.2f}, 总推理耗时: {total_infer_time:.2f}, 总奖励: {total_reward_sum:.2f}")

    print(f"当前设置{NODE_NUM}个节点，决策区域长度为{DECISION_REGION}，子决策区域长度为{SUB_DECISION_REGION}。{'Greedy' if GREEDY else 'Sample'}模式推理完成！结果已保存至", CSV_PATH)

def save_combined_result_to_img(coordinates, all_edge_index, all_edge_attr, vrp_routes,
                                file_path='mTSPD/result_vrp.svg'):
    """
    基于所有路径的合并边信息保存最终结果图片
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from env import COMMON_ROUTE, DRONE_ROUTE, TRUCK_ROUTE

    # 基础检查
    if all_edge_attr is None or all_edge_index is None:
        raise ValueError("all_edge_index/all_edge_attr 不能为空")

    fig, ax = plt.subplots()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 12})

    city_x = coordinates[:, 0]
    city_y = coordinates[:, 1]

    # 构建每个 vehicle 的节点集合
    vehicle_node_sets = []
    for vr in vrp_routes:
        if isinstance(vr, (list, tuple)) and len(vr) >= 2:
            route = vr[1]
        else:
            route = vr
        vehicle_node_sets.append(set(route))

    num_vehicles = len(vehicle_node_sets)
    # 准备容器
    vehicle_lines = [{"truck": [], "drone": []} for _ in range(num_vehicles)]

    E = all_edge_index.shape[1]

    for i in range(0, all_edge_attr.size(0), 2):
        u = int(all_edge_index[0, i].item())
        v = int(all_edge_index[1, i].item())
        # 路径类型判断
        route_type = int(all_edge_attr[i][1].item())

        matched_v = -1
        for vidx, nset in enumerate(vehicle_node_sets):
            if u in nset and v in nset:
                matched_v = vidx
                break  # 一定唯一，找到即停止

        edge_coords = [(city_x[u], city_y[u]), (city_x[v], city_y[v])]

        if route_type == DRONE_ROUTE:
            vehicle_lines[matched_v]["drone"].append(edge_coords)
        else:
            vehicle_lines[matched_v]["truck"].append(edge_coords)

    # 颜色列表（可扩充）
    base_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                   'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    # 绘制每个 vehicle（truck 用实线，drone 用虚线）
    for v in range(num_vehicles):
        color = base_colors[v % len(base_colors)]
        # truck (含 common)
        if vehicle_lines[v]["truck"]:
            lc_truck = LineCollection(vehicle_lines[v]["truck"],
                                      colors=color,
                                      linewidths=1.6,
                                      linestyles='solid',
                                      label=f'Truck {v+1}')
            ax.add_collection(lc_truck)
        # drone
        if vehicle_lines[v]["drone"]:
            lc_drone = LineCollection(vehicle_lines[v]["drone"],
                                      colors=color,
                                      linewidths=1.6,
                                      linestyles='dashed',
                                      label=f'Drone {v+1}')
            ax.add_collection(lc_drone)

    ax.legend(loc='upper right')

    ax.scatter(city_x, city_y, color='red', linewidths=0.5, zorder=3)
    try:
        ax.scatter(city_x[0], city_y[0], color='yellow', linewidths=0.5, zorder=3)
    except Exception:
        pass

    # 设定坐标范围（与原代码一致）
    ax.set_xlim(0, COORD_MAX+15)
    ax.set_ylim(0, COORD_MAX)
    ax.set_title('mTSP-D')

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    infer(False)