import torch
from lkh_solver import LKH_Solver
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

TRUCK_ROUTE = 0
DRONE_ROUTE = 1
COMMON_ROUTE = 2

TRUCK_SPEED = 0.5
DRONE_SPEED = 1.0
UN_VISITED = 0
DRONE_VISITED = 1
TRUCK_VISITED = 2
COMMON_VISITED = 3
"""
self.now和xxx_idx记录的是节点在self.result_seq中的索引
self.result_seq记录的是节点在self.coordinates中的索引，这个索引可以理解为节点的id
decision_node记录的也是节点在self.coordinates中的索引
"""
class Environment():
    def __init__(self, coord_max, node_num, device, decision_region, sub_decision_region):
        assert decision_region>=3, "决策区域大小至少为3"
        assert node_num>=3, "节点数至少为3"
        assert sub_decision_region>=3 and sub_decision_region<=decision_region, "子决策区域需合理"
        self.coord_max = coord_max
        self.node_num = node_num
        self.device = device
        self.decision_region = decision_region # 决策区间的长度
        self.sub_decision_region = sub_decision_region # 子决策区间的长度
        self.coordinates = None
        self.result = None
        self.result_seq = None
        self.now = None # 用来存当前出发点在result_seq中的索引，以便从result_seq中取到节点的id
        self.edge_index = None
        self.edge_attr = None
        self.done = 0
        self.time = 0
        self.time_to_now = 0 # 用来在step时记录到达当前now的时间，在修改过程中顺便记录更容易。如果在self.done后再统计会非常麻烦
        self.fig = None # 画图用的
        self.ax = None
        self.time_step = 0

    def reset(self, data=None):
        if data != None:
            zeros = torch.zeros(self.node_num, 1)   # 创建 (self.node_num,1) 的全零张量
            self.coordinates = torch.cat([data, zeros], dim=1)
            lkh = LKH_Solver(self.coordinates)
            self.result = lkh.solve()
            self.result_seq = self.result['result']
        self.process_result(self.result)
        self.now = 0 # 表示上次决策的终点的索引/本次决策的起点在result_seq中的索引
        self.done = 0
        self.time_to_now = 0
        decision_node = []
        decision_mask = []
        for i in range(self.decision_region): # 使用决策索引和决策mask来处理邻近终止情况下的观测数据
            if self.now + i >= len(self.result_seq)-1:
                decision_node.append(self.result_seq[self.now+i-self.node_num])
                decision_mask.append(0)
            else:
                decision_node.append(self.result_seq[self.now+i])
                if i==0 or i == self.decision_region-1:
                    decision_mask.append(0)
                else:
                    decision_mask.append(1)
        obs = {'decision_node': torch.tensor(decision_node).to(self.device),
               'decision_mask': torch.tensor(decision_mask).to(self.device),
               'now_node': torch.tensor([self.result_seq[self.now]]).to(self.device),
               'coord':self.coordinates.to(self.device)}
        # decision_node:(decision_region, ), decision_mask:(decision_region, ), now_node:(1,), start_node:(1,)
        self.time = self.result['total_distance']/TRUCK_SPEED
        self.time_step = 0
        return obs, self.time
    
    def get_reward(self, original_time, path_time):
        if self.done == 1:
            reward =  (self.time - self.time_to_now)/10 + (original_time - path_time)
        else:
            reward =  (original_time - path_time)
        return reward
    # def get_reward(self, original_time, time):
    #     return original_time-time
        
    def step(self, action):
        self.time_step += 1
        sub_action = action[1:] # 剩余部分是子决策区域的分配
        fork_start_idx = min(self.now + action[0], len(self.result_seq)-1) # 分叉点索引
        fork_end_idx = min(fork_start_idx + self.sub_decision_region - 1, len(self.result_seq)-1)# 子决策区域的汇合点
        drone_path = [self.result_seq[fork_start_idx]]  # 用于记录动作为1的节点
        truck_path = [self.result_seq[fork_start_idx]]  # 用于记录动作为0的节点
        for idx in range(self.now, fork_start_idx+1):
            self.coordinates[self.result_seq[idx]][2] = COMMON_VISITED

        if fork_end_idx-fork_start_idx>=2: # 当子决策区域内至少有3个点时，才有划分的意义
            # 1. 移除旧边
            original_distance = self.remove_old_edges(fork_start_idx, fork_end_idx) # 将移除范围的起点与终点传入，并得到被移除边的路径长度

            # 2. 遍历动作，构建路径的序列
            for i in range(fork_end_idx-fork_start_idx-1):
                current_node = self.result_seq[fork_start_idx + i + 1] # # 动作控制中间节点，从fork_start_idx+1开始
                if sub_action[i] == 1:
                    drone_path.append(current_node)  # 记录分叉路径中的节点
                    self.coordinates[current_node][2] = DRONE_VISITED
                else:
                    truck_path.append(current_node)  # 记录主路径节点
                    self.coordinates[current_node][2] = TRUCK_VISITED
            drone_path.append(self.result_seq[fork_end_idx]) # 添加汇合点
            truck_path.append(self.result_seq[fork_end_idx]) # 添加汇合点
            self.coordinates[self.result_seq[fork_end_idx]][2] = COMMON_VISITED

            total_time = 0 # 先统计没有分叉时的时间，然后到了分叉时加上max(truck_time, drone_time)即可
            truck_time = 0 # 统计分叉时货车路径所需时间
            drone_time = 0 # 统计分叉时无人机路径所需时间
            
            # 3. 统计主路径的距离和花费时长（用货车速度）
            main_distance = 0
            nodes = self.result_seq[self.now:fork_start_idx+1]
            for i in range(len(nodes)-1):
                main_distance += self.calculate_distance(nodes[i], nodes[i+1])
            total_time = main_distance/TRUCK_SPEED
            original_distance += main_distance
            original_time = original_distance/TRUCK_SPEED

            # 4. 构建货车路径的边和属性
            truck_edge_count = len(truck_path) - 1
            truck_edges = torch.zeros((2, truck_edge_count * 2), dtype=torch.long)  # 预分配边的张量
            truck_edge_attr = torch.zeros((truck_edge_count * 2, 2), dtype=torch.float)  # 预分配边属性的张量
            for i in range(truck_edge_count):
                truck_edges[0][i * 2] = truck_path[i]
                truck_edges[1][i * 2] = truck_path[i + 1]  # 正向边
                truck_edges[0][i * 2 + 1] = truck_path[i + 1]
                truck_edges[1][i * 2 + 1] = truck_path[i]  # 反向边
                distance = self.calculate_distance(truck_path[i], truck_path[i + 1])
                truck_time += distance/TRUCK_SPEED
                truck_edge_attr[i * 2] = torch.tensor([distance, TRUCK_ROUTE], dtype=torch.float)
                truck_edge_attr[i * 2 + 1] = torch.tensor([distance, TRUCK_ROUTE], dtype=torch.float)

            # 5. 构建无人机路径的边和属性
            drone_edge_count = len(drone_path) - 1
            drone_edges = torch.zeros((2, drone_edge_count * 2), dtype=torch.long)  # 预分配边的张量
            drone_edge_attr = torch.zeros((drone_edge_count * 2, 2), dtype=torch.float)  # 预分配边属性的张量
            for i in range(drone_edge_count):
                drone_edges[0][i * 2] = drone_path[i]
                drone_edges[1][i * 2] = drone_path[i + 1]  # 正向边
                drone_edges[0][i * 2 + 1] = drone_path[i + 1]
                drone_edges[1][i * 2 + 1] = drone_path[i]  # 反向边
                distance = self.calculate_distance(drone_path[i], drone_path[i + 1])
                drone_time += distance/DRONE_SPEED
                drone_edge_attr[i * 2] = torch.tensor([distance, DRONE_ROUTE], dtype=torch.float)
                drone_edge_attr[i * 2 + 1] = torch.tensor([distance, DRONE_ROUTE], dtype=torch.float)
            
            # 主路+无人机和货车其中较长的时间，即为从起点到汇合点的总时间，可以视作先完成访问的在汇合点等待后完成的。然后将这段时间合并到已处理部分的时间
            total_time += max(truck_time, drone_time)
            # 6. 拼接原始边与新生成的边
            self.edge_index = torch.cat([self.edge_index, truck_edges, drone_edges], dim=1)  # 拼接边
            self.edge_attr = torch.cat([self.edge_attr, truck_edge_attr, drone_edge_attr], dim=0)  # 拼接边属性
        else:
            original_distance = 0
            if fork_end_idx<len(self.result_seq):
                nodes = self.result_seq[self.now:fork_end_idx+1]
            else:
                nodes = self.result_seq[self.now:]
            for i in range(len(nodes)-1):
                original_distance += self.calculate_distance(nodes[i], nodes[i+1])
            original_time = original_distance/TRUCK_SPEED
            total_time = original_time
            
        self.time_to_now += total_time

        self.now = fork_end_idx # 更新当前节点为汇合点
        decision_node = []
        decision_mask = []
        for i in range(self.decision_region): # 使用决策区域索引和决策mask来处理邻近终止情况下的观测数据
            if self.now + i >= len(self.result_seq)-1: # 最后一个点必须是汇合状态，所以条件里用的是len(self.result_seq)-1
                decision_node.append(self.result_seq[self.now+i-self.node_num])
                decision_mask.append(0)
            else:
                decision_node.append(self.result_seq[self.now+i])
                if i==0 or i == self.decision_region-1:
                    decision_mask.append(0)
                else:
                    decision_mask.append(1)
        obs = {'decision_node':torch.tensor(decision_node).to(self.device),
               'decision_mask':torch.tensor(decision_mask).to(self.device),
               'now_node':torch.tensor([self.result_seq[self.now]]).to(self.device),
               'coord':self.coordinates.to(self.device)}

        if len(self.result_seq) - self.now < 3: # 如果无法保证决策区域内至少有三个点可用，则当前场景结束，剩下部分相当于货车拉着无人机走完
            self.done = 1
            for i in range(len(self.result_seq)-self.now-1):
                time = self.calculate_distance(self.result_seq[self.now+i], self.result_seq[self.now+i+1])
                total_time += time
                self.time_to_now += time
                original_time += time
        else:
            self.done = 0
        reward = self.get_reward(original_time, total_time)

        return obs, reward, self.done, self.time_to_now

    def process_result(self, result):
        '''
            根据当前动作的处理，重现构建边结构以及节点的访问状态
        '''
        node_num = len(self.coordinates)
        edge_index = torch.zeros((2, node_num*2), dtype=torch.long) # 由于是TSP的解，所以有多少个点就有多少条边，并且是无向图所以边是双向的，所以添加的边是实际的两倍。
        edge_attr = torch.zeros((node_num*2, 2), dtype=torch.float) # 属性1为距离，属性2为边的种类，表示当前边由谁走，货车是1，无人机是2
        for i in range(node_num): # 构建边
            edge_attr[i*2] = torch.tensor([result['distances'][i], COMMON_ROUTE], dtype=torch.float)
            edge_attr[i*2+1] = torch.tensor([result['distances'][i], COMMON_ROUTE], dtype=torch.float)
            edge_index[0][i*2] = result['result'][i] # 这两行加入了i指向i+1的边
            edge_index[1][i*2] = result['result'][i+1]
            edge_index[0][i*2+1] = result['result'][i+1] # 这两行加入了i+1指向i的边
            edge_index[1][i*2+1] = result['result'][i]
        self.edge_index = edge_index
        self.edge_attr = edge_attr
    
    def calculate_distance(self, start, end):
        return ((self.coordinates[start][0]-self.coordinates[end][0])**2 + 
                (self.coordinates[start][1]-self.coordinates[end][1])**2)**0.5
    
    def remove_old_edges(self, fork_start_idx, fork_end_idx):
        '''
            移除范围内的边
        '''
        # 获取要删除的节点范围
        if(fork_end_idx)>=len(self.result_seq):
            remove_node_list = self.result_seq[fork_start_idx:]
        else:
            remove_node_list = self.result_seq[fork_start_idx:fork_end_idx+1]
        n = self.edge_index.shape[1]  # 边的数量
        remove_edge_mask = torch.ones((n,), dtype=torch.bool)  # 初始化为True，表示默认保留所有边
        distance = 0
        for i in range(n):
            # 检查边的两端是否都在删除节点列表中
            if (self.edge_index[0][i] in remove_node_list) and (self.edge_index[1][i] in remove_node_list):
                remove_edge_mask[i] = False  # 设为False，表示该边需要被删除
                distance += self.calculate_distance(self.edge_index[0][i], self.edge_index[1][i]) # 将边的距离纳入统计

        # 根据mask保留需要的边
        self.edge_index = self.edge_index[:, remove_edge_mask]  # 只保留不需要删除的边
        self.edge_attr = self.edge_attr[remove_edge_mask, :]  # 保留对应的边属性
        return distance/2

    def draw(self):
        '''
            逐步调用以实时显示解路径的当前结构
        '''
        assert self.edge_attr.size(0)%2==0
        assert self.result is not None
        # 创建绘图
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            # plt.show()
        # 清除当前绘图内容
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        self.ax.cla()

        city_x=self.coordinates[:,0]
        city_y=self.coordinates[:,1]
        line_common = []
        line_drone = []
        line_truck = []
        for i in range(0, self.edge_attr.size(0), 2):
            edge = [(city_x[self.edge_index[0][i]], city_y[self.edge_index[0][i]]), (city_x[self.edge_index[1][i]], city_y[self.edge_index[1][i]])]
            if self.edge_attr[i][1] == COMMON_ROUTE:
                line_common.append(edge)
            elif self.edge_attr[i][1] == DRONE_ROUTE:
                line_drone.append(edge)
            elif self.edge_attr[i][1] == TRUCK_ROUTE:
                line_truck.append(edge)
            else:
                raise ValueError("应该也不会有其他值了吧")
        line_segments1 = LineCollection(line_common, colors='blue', linewidths=1,label='共同路线')
        line_segments2 = LineCollection(line_drone, colors='red', linewidths=1,label='无人机路线')
        line_segments3 = LineCollection(line_truck, colors='green', linewidths=1,label='货车路线')
        
        # 添加线段到绘图中
        self.ax.add_collection(line_segments1)
        self.ax.add_collection(line_segments2)
        self.ax.add_collection(line_segments3)
        # 添加图例
        self.ax.legend(loc='upper right')
        # 添加点到绘图中
        self.ax.scatter(city_x, city_y, color='red', linewidths=0.5)
        self.ax.scatter(city_x[0], city_y[0], color='yellow', linewidths=0.5)
        # 添加每个点的ID作为标签
        # for i in range(len(self.coordinates)):
        #     self.ax.text(city_x[i], city_y[i], str(i), fontsize=9, ha='right', color='black')
        # 设置坐标范围
        self.ax.set_xlim(0, self.coord_max)
        self.ax.set_ylim(0, self.coord_max)
        # 添加标题和标签
        self.ax.set_title(f'TSPD')
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        # 显示绘图
        plt.draw()
        plt.pause(0.3)

    def save_result_to_img(self, file_path='SAPPO/result.svg'):
        '''
            调用时会保存当前的解路径结构为一个矢量图
        '''
        assert self.edge_attr.size(0) % 2 == 0
        assert self.result is not None
        
        # 创建一个新的绘图对象以保存图片
        fig, ax = plt.subplots()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        city_x = self.coordinates[:, 0]
        city_y = self.coordinates[:, 1]
        line_common = []
        line_drone = []
        line_truck = []
        
        # 按照边的类型分类，准备绘图数据
        for i in range(0, self.edge_attr.size(0), 2):
            edge = [(city_x[self.edge_index[0][i]], city_y[self.edge_index[0][i]]), 
                    (city_x[self.edge_index[1][i]], city_y[self.edge_index[1][i]])]
            if self.edge_attr[i][1] == COMMON_ROUTE:
                line_common.append(edge)
            elif self.edge_attr[i][1] == DRONE_ROUTE:
                line_drone.append(edge)
            elif self.edge_attr[i][1] == TRUCK_ROUTE:
                line_truck.append(edge)
            else:
                raise ValueError("应该也不会有其他值了吧")
        
        # 创建线段集合
        line_segments1 = LineCollection(line_common, colors='blue', linewidths=1,label='common route')
        line_segments2 = LineCollection(line_drone, colors='red', linewidths=1,label='drone route')
        line_segments3 = LineCollection(line_truck, colors='green', linewidths=1,label='truck route')
        
        # 添加线段到绘图中
        ax.add_collection(line_segments1)
        ax.add_collection(line_segments2)
        ax.add_collection(line_segments3)

        # 添加图例
        ax.legend(loc='upper right')
        
        # 添加点到绘图中
        ax.scatter(city_x, city_y, color='red', linewidths=0.5)
        ax.scatter(city_x[0], city_y[0], color='yellow', linewidths=0.5)
        
        # 设置坐标范围
        ax.set_xlim(0, self.coord_max)
        ax.set_ylim(0, self.coord_max)
        
        print(f'TSPD time={self.time_to_now:.2f} original={self.time:.2f}')
        
        # 保存图片到指定路径
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        
        # 关闭绘图以释放内存
        plt.close(fig)

    def save_instance_to_img(self, file_path='SAPPO/instance.svg'):
        """
        生成并保存仅包含节点的图片
        参数:
            file_path (str): 保存图片的文件路径，默认为 'SAPPO/instance.svg'
        """
        assert self.coordinates is not None, "坐标数据未初始化"

        # 创建新的绘图对象
        fig, ax = plt.subplots()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 提取节点坐标
        city_x = self.coordinates[:, 0]
        city_y = self.coordinates[:, 1]

        # 绘制所有节点，起点用黄色，其他用红色
        ax.scatter(city_x, city_y, color='red', linewidths=0.5)
        ax.scatter(city_x[0], city_y[0], color='yellow', linewidths=0.5)

        # 设置坐标范围
        ax.set_xlim(0, self.coord_max)
        ax.set_ylim(0, self.coord_max)

        # 添加标题和标签
        ax.set_title('Instance')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

        # 保存图片
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 关闭绘图以释放内存

    def save_step_to_img(self, base_dir='SAPPO/step_images'):
        """
        类似 draw 函数，但将当前步骤的路径图保存到本地，文件名与 self.time_step 相关
        参数:
            base_dir (str): 保存图片的目录，默认为 'SAPPO/step_images'
        """
        import os
        assert self.edge_attr.size(0) % 2 == 0, "边属性数量必须为偶数"
        assert self.result is not None, "结果数据未初始化"

        # 创建保存目录（如果不存在）
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # 生成文件名，基于 self.time_step
        file_path = os.path.join(base_dir, f'step_{self.time_step}.svg')

        # 创建新的绘图对象
        fig, ax = plt.subplots()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 提取坐标
        city_x = self.coordinates[:, 0]
        city_y = self.coordinates[:, 1]

        # 分类路径
        line_common = []
        line_drone = []
        line_truck = []
        for i in range(0, self.edge_attr.size(0), 2):
            edge = [(city_x[self.edge_index[0][i]], city_y[self.edge_index[0][i]]), 
                    (city_x[self.edge_index[1][i]], city_y[self.edge_index[1][i]])]
            if self.edge_attr[i][1] == COMMON_ROUTE:
                line_common.append(edge)
            elif self.edge_attr[i][1] == DRONE_ROUTE:
                line_drone.append(edge)
            elif self.edge_attr[i][1] == TRUCK_ROUTE:
                line_truck.append(edge)
            else:
                raise ValueError("未知的路径类型")

        # 创建线段集合
        line_segments1 = LineCollection(line_common, colors='blue', linewidths=1, label='common route')
        line_segments2 = LineCollection(line_drone, colors='red', linewidths=1, label='drone route')
        line_segments3 = LineCollection(line_truck, colors='green', linewidths=1, label='truck route')

        # 添加线段到绘图中
        ax.add_collection(line_segments1)
        ax.add_collection(line_segments2)
        ax.add_collection(line_segments3)

        # 添加图例
        ax.legend(loc='upper right')

        # 添加节点
        ax.scatter(city_x, city_y, color='red', linewidths=0.5)
        ax.scatter(city_x[0], city_y[0], color='yellow', linewidths=0.5)

        # 设置坐标范围
        ax.set_xlim(0, self.coord_max)
        ax.set_ylim(0, self.coord_max)

        # 添加标题和标签
        ax.set_title(f'TSP-D')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

        # 保存图片
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 关闭绘图以释放内存