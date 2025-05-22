# 文件名: agent.py
import torch
import torch.nn as nn
import math

class PPOAgent(nn.Module):
    def __init__(self, node_feature_dim=2, hidden_dim=128, decision_region=5, sub_decision_region=4, num_heads=4, num_layers=2, device='cuda'):
        super(PPOAgent, self).__init__()
        self.device = device
        self.decision_region = decision_region
        self.sub_decision_region = sub_decision_region
        self.fork_range = decision_region - sub_decision_region + 1  # action[0]的取值范围
        self.hidden_dim = hidden_dim

        # 节点特征嵌入
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Actor部分：注意力机制处理决策区域
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads)
        self.attn_fc = nn.Linear(hidden_dim, hidden_dim)

        # Actor解码器：选择分叉点
        self.fork_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.fork_range)  # 输出分叉点的logits
        )

        # Actor解码器：子决策区域动作生成
        self.actor_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出动作的logits
        )

        # Critic部分
        self.critic_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出单一价值估计
        )

        # 位置编码
        self.pos_encoding = self._generate_pos_encoding(decision_region, hidden_dim).to(device)

    def _generate_pos_encoding(self, max_len, d_model):
        """生成位置编码矩阵，基于正弦和余弦函数"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model)

    def forward(self, obs):
        """
        输入：
        - obs: 观测数据，包含 decision_node (batch_size, decision_region), decision_mask (batch_size, decision_region),
               now_node (batch_size, 1), coord (batch_size, node_num, 2)
        输出：
        - action: 动作，(batch_size, sub_decision_region-1)，第一个是分叉点偏移，后续是子决策区域分配
        - log_prob: 动作的对数概率，(batch_size,)
        - value: 状态价值，(batch_size,)
        """
        batch_size = obs['decision_node'].shape[0]

        # 处理输入维度
        x = obs['coord']

        # 节点特征嵌入
        x = self.node_embedding(x)  # (batch_size, node_num, hidden_dim)

        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, node_num, hidden_dim)

        # 获取 now_node 的嵌入
        now_node_idx = obs['now_node']  # (batch_size, 1)
        now_node_embedding = torch.cat([x[i, now_node_idx[i]] for i in range(batch_size)], dim=0)  # (batch_size, hidden_dim)

        # 全局特征（平均池化）
        graph_embedding = x.mean(dim=1)  # (batch_size, hidden_dim)

        # 获取决策区域的节点嵌入
        decision_node = obs['decision_node']  # (batch_size, decision_region)
        decision_nodes = torch.stack([x[i, decision_node[i]] for i in range(batch_size)], dim=0)  # (batch_size, decision_region, hidden_dim)

        # 添加位置编码
        decision_nodes = decision_nodes + self.pos_encoding.unsqueeze(0)  # (batch_size, decision_region, hidden_dim)

        # 注意力机制处理决策区域
        decision_nodes = decision_nodes.permute(1, 0, 2)  # (decision_region, batch_size, hidden_dim)
        attn_output, _ = self.attn(decision_nodes, decision_nodes, decision_nodes)  # (decision_region, batch_size, hidden_dim)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, decision_region, hidden_dim)
        attn_output = torch.relu(self.attn_fc(attn_output))  # (batch_size, decision_region, hidden_dim)

        # 选择分叉点
        decision_mean = attn_output.mean(dim=1)  # (batch_size, hidden_dim)
        fork_combined = torch.cat([decision_mean, graph_embedding, now_node_embedding], dim=-1)  # (batch_size, hidden_dim*3)
        fork_logits = self.fork_decoder(fork_combined)  # (batch_size, fork_range)
        fork_probs = torch.softmax(fork_logits, dim=-1)  # (batch_size, fork_range)
        fork_dist = torch.distributions.Categorical(probs=fork_probs)
        fork_idx = fork_dist.sample()  # (batch_size,)
        fork_log_prob = fork_dist.log_prob(fork_idx)  # (batch_size,)

        # 获取子决策区域的节点嵌入
        sub_start_idx = fork_idx.unsqueeze(-1)  # (batch_size, 1)
        sub_indices = sub_start_idx + torch.arange(self.sub_decision_region, device=self.device).unsqueeze(0)  # (batch_size, sub_decision_region)
        sub_nodes = torch.stack([attn_output[i, sub_indices[i]] for i in range(batch_size)], dim=0)  # (batch_size, sub_decision_region, hidden_dim)

        # 只取子决策区域中间部分（去掉分叉点和汇合点）
        sub_middle = sub_nodes[:, 1:-1, :]  # (batch_size, sub_decision_region-2, hidden_dim)

        # 融合全局图特征、子决策区域特征和 now_node_embedding
        graph_embedding_expanded = graph_embedding.unsqueeze(1).expand(-1, self.sub_decision_region - 2, -1)  # (batch_size, sub_decision_region-2, hidden_dim)
        now_node_embedding_expanded = now_node_embedding.unsqueeze(1).expand(-1, self.sub_decision_region - 2, -1)  # (batch_size, sub_decision_region-2, hidden_dim)
        combined = torch.cat([sub_middle, graph_embedding_expanded, now_node_embedding_expanded], dim=-1)  # (batch_size, sub_decision_region-2, hidden_dim*3)

        # Actor: 生成子决策区域动作概率
        logits = self.actor_decoder(combined).squeeze(dim=2)  # (batch_size, sub_decision_region-2)
        probs = torch.sigmoid(logits)  # (batch_size, sub_decision_region-2)

        # 使用 decision_mask 屏蔽无效动作（动态提取子决策区域中间部分的掩码）
        sub_mask = torch.stack([obs['decision_mask'][i, fork_idx[i]+1:fork_idx[i]+self.sub_decision_region-1] for i in range(batch_size)], dim=0)  # (batch_size, sub_decision_region-2)
        probs = probs * sub_mask  # 屏蔽无效动作

        # 使用伯努利分布采样动作
        dist = torch.distributions.Bernoulli(probs=probs)
        sub_action = dist.sample()  # (batch_size, sub_decision_region-2)
        
        # 计算采样动作的对数概率
        sub_log_prob = dist.log_prob(sub_action).sum(dim=-1)  # (batch_size,)

        # 合并动作和对数概率
        action = torch.cat([fork_idx.unsqueeze(-1), sub_action.to(torch.int64)], dim=-1)  # (batch_size, sub_decision_region-1)
        log_prob = fork_log_prob + sub_log_prob  # (batch_size,)

        # 计算熵
        fork_entropy = fork_dist.entropy()  # (batch_size,)
        sub_entropy = -(probs * torch.log(probs + 1e-10) + (1 - probs) * torch.log(1 - probs + 1e-10)).sum(dim=-1) # (batch_size,)
        entropy = fork_entropy + sub_entropy  # (batch_size,)

        # Critic: 估计状态价值
        critic_combined = torch.cat([decision_mean, graph_embedding, now_node_embedding], dim=-1)  # (batch_size, hidden_dim*2)

        value = self.critic_decoder(critic_combined)  # (batch_size, 1)
        return action, log_prob, value.squeeze(-1), entropy

        # _, value, _ = self.evaluate(obs, action)
        # return action, log_prob, value, entropy

    def evaluate(self, obs, action):
        """
        用于计算给定动作的对数概率和状态价值
        输入：
        - obs: 观测数据
        - action: 已选择的动作，(batch_size, sub_decision_region-1)
        输出：
        - log_prob: 动作的对数概率，(batch_size,)
        - value: 状态价值，(batch_size,)
        """
        batch_size = obs['decision_node'].shape[0]

        x = obs['coord']

        x = self.node_embedding(x)
        x = self.transformer_encoder(x)

        # 获取 now_node 的嵌入
        now_node_idx = obs['now_node']  # (batch_size, 1)
        now_node_embedding = torch.cat([x[i, now_node_idx[i]] for i in range(batch_size)], dim=0)  # (batch_size, hidden_dim)

        graph_embedding = x.mean(dim=1)

        decision_node = obs['decision_node']
        decision_nodes = torch.stack([x[i, decision_node[i]] for i in range(batch_size)], dim=0)

        # 添加位置编码
        decision_nodes = decision_nodes + self.pos_encoding.unsqueeze(0)  # (batch_size, decision_region, hidden_dim)

        decision_nodes = decision_nodes.permute(1, 0, 2)
        attn_output, _ = self.attn(decision_nodes, decision_nodes, decision_nodes)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = torch.relu(self.attn_fc(attn_output))

        # 分叉点概率
        decision_mean = attn_output.mean(dim=1)
        fork_combined = torch.cat([decision_mean, graph_embedding, now_node_embedding], dim=-1)  # (batch_size, hidden_dim*3)
        fork_logits = self.fork_decoder(fork_combined)
        fork_probs = torch.softmax(fork_logits, dim=-1)
        fork_dist = torch.distributions.Categorical(probs=fork_probs)
        fork_idx = action[:, 0]  # (batch_size,)
        fork_log_prob = fork_dist.log_prob(fork_idx)  # (batch_size,)

        # 子决策区域动作概率
        sub_start_idx = fork_idx.unsqueeze(-1)
        sub_indices = sub_start_idx + torch.arange(self.sub_decision_region, device=self.device).unsqueeze(0)
        sub_nodes = torch.stack([attn_output[i, sub_indices[i]] for i in range(batch_size)], dim=0)
        sub_middle = sub_nodes[:, 1:-1, :]

        graph_embedding_expanded = graph_embedding.unsqueeze(1).expand(-1, self.sub_decision_region - 2, -1)
        now_node_embedding_expanded = now_node_embedding.unsqueeze(1).expand(-1, self.sub_decision_region - 2, -1)  # (batch_size, sub_decision_region-2, hidden_dim)
        combined = torch.cat([sub_middle, graph_embedding_expanded, now_node_embedding_expanded], dim=-1)  # (batch_size, sub_decision_region-2, hidden_dim*3)

        logits = self.actor_decoder(combined).squeeze(2)
        probs = torch.sigmoid(logits)

        # 使用 decision_mask 屏蔽无效动作（动态提取子决策区域中间部分的掩码）
        sub_mask = torch.stack([obs['decision_mask'][i, fork_idx[i]+1:fork_idx[i]+self.sub_decision_region-1] for i in range(batch_size)], dim=0)  # (batch_size, sub_decision_region-2)
        probs = probs * sub_mask  # 屏蔽无效动作

        # 计算子决策区域动作的对数概率
        sub_action = action[:, 1:]  # (batch_size, sub_decision_region-2)
        sub_log_prob = torch.log(probs + 1e-10) * sub_action + torch.log(1 - probs + 1e-10) * (1 - sub_action)
        sub_log_prob = sub_log_prob.sum(dim=-1)  # (batch_size,)

        # 合并对数概率
        action_log_prob = fork_log_prob + sub_log_prob  # (batch_size,)

        # 计算熵
        fork_entropy = fork_dist.entropy()  # (batch_size,)
        sub_entropy = -(probs * torch.log(probs + 1e-10) + (1 - probs) * torch.log(1 - probs + 1e-10)).sum(dim=-1)  # (batch_size,)
        entropy = fork_entropy + sub_entropy  # (batch_size,)

        # Critic: 估计状态价值
        critic_combined = torch.cat([decision_mean, graph_embedding, now_node_embedding], dim=-1)  # (batch_size, hidden_dim*2)
        value = self.critic_decoder(critic_combined).squeeze(-1)
        return action_log_prob, value, entropy
    
    def greedy_forward(self, obs):
        """
        推理阶段的贪婪策略前向传播。
        输入：
        - obs: 观测数据，包含 decision_node (batch_size, decision_region), decision_mask (batch_size, decision_region),
            now_node (batch_size, 1), coord (batch_size, node_num, 2)
        输出：
        - action: 动作，(batch_size, sub_decision_region-1)，第一个是分叉点偏移，后续是子决策区域分配
        """
        batch_size = obs['decision_node'].shape[0]

        # 处理输入维度
        x = obs['coord']

        # 节点特征嵌入
        x = self.node_embedding(x)  # (batch_size, node_num, hidden_dim)

        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, node_num, hidden_dim)

        # 获取 now_node 的嵌入
        now_node_idx = obs['now_node']  # (batch_size, 1)
        now_node_embedding = torch.cat([x[i, now_node_idx[i]] for i in range(batch_size)], dim=0)  # (batch_size, hidden_dim)

        # 全局特征（平均池化）
        graph_embedding = x.mean(dim=1)  # (batch_size, hidden_dim)

        # 获取决策区域的节点嵌入
        decision_node = obs['decision_node']  # (batch_size, decision_region)
        decision_nodes = torch.stack([x[i, decision_node[i]] for i in range(batch_size)], dim=0)  # (batch_size, decision_region, hidden_dim)

        # 添加位置编码
        decision_nodes = decision_nodes + self.pos_encoding.unsqueeze(0)  # (batch_size, decision_region, hidden_dim)

        # 注意力机制处理决策区域
        decision_nodes = decision_nodes.permute(1, 0, 2)  # (decision_region, batch_size, hidden_dim)
        attn_output, _ = self.attn(decision_nodes, decision_nodes, decision_nodes)  # (decision_region, batch_size, hidden_dim)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, decision_region, hidden_dim)
        attn_output = torch.relu(self.attn_fc(attn_output))  # (batch_size, decision_region, hidden_dim)

        # 选择分叉点（贪婪策略）
        decision_mean = attn_output.mean(dim=1)  # (batch_size, hidden_dim)
        fork_combined = torch.cat([decision_mean, graph_embedding, now_node_embedding], dim=-1)  # (batch_size, hidden_dim*3)
        fork_logits = self.fork_decoder(fork_combined)  # (batch_size, fork_range)
        fork_probs = torch.softmax(fork_logits, dim=-1)  # (batch_size, fork_range)
        fork_idx = torch.argmax(fork_probs, dim=-1)  # (batch_size,)，选择概率最大的分叉点

        # 获取子决策区域的节点嵌入
        sub_start_idx = fork_idx.unsqueeze(-1)  # (batch_size, 1)
        sub_indices = sub_start_idx + torch.arange(self.sub_decision_region, device=self.device).unsqueeze(0)  # (batch_size, sub_decision_region)
        sub_nodes = torch.stack([attn_output[i, sub_indices[i]] for i in range(batch_size)], dim=0)  # (batch_size, sub_decision_region, hidden_dim)

        # 只取子决策区域中间部分（去掉分叉点和汇合点）
        sub_middle = sub_nodes[:, 1:-1, :]  # (batch_size, sub_decision_region-2, hidden_dim)

        # 融合全局图特征、子决策区域特征和 now_node_embedding
        graph_embedding_expanded = graph_embedding.unsqueeze(1).expand(-1, self.sub_decision_region - 2, -1)  # (batch_size, sub_decision_region-2, hidden_dim)
        now_node_embedding_expanded = now_node_embedding.unsqueeze(1).expand(-1, self.sub_decision_region - 2, -1)  # (batch_size, sub_decision_region-2, hidden_dim)
        combined = torch.cat([sub_middle, graph_embedding_expanded, now_node_embedding_expanded], dim=-1)  # (batch_size, sub_decision_region-2, hidden_dim*3)

        # 生成子决策区域动作（贪婪策略）
        logits = self.actor_decoder(combined).squeeze(dim=2)  # (batch_size, sub_decision_region-2)
        probs = torch.sigmoid(logits)  # (batch_size, sub_decision_region-2)

        # 使用 decision_mask 屏蔽无效动作
        sub_mask = torch.stack([obs['decision_mask'][i, fork_idx[i]+1:fork_idx[i]+self.sub_decision_region-1] for i in range(batch_size)], dim=0)  # (batch_size, sub_decision_region-2)
        probs = probs * sub_mask  # 屏蔽无效动作

        # 贪婪选择子决策区域动作
        sub_action = (probs > 0.5).to(torch.int64)  # (batch_size, sub_decision_region-2)

        # 合并动作
        action = torch.cat([fork_idx.unsqueeze(-1), sub_action], dim=-1)  # (batch_size, sub_decision_region-1)

        return action