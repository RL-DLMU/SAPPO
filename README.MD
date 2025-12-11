# TSP-D强化学习两阶段框架

本项目使用强化学习解决TSP-D（带无人机的旅行商问题）。以下内容介绍了项目的环境要求、配置方法以及使用方式。

---

## 项目环境与配置

### 环境要求
- **操作系统**：Windows（已在Windows 11上测试）。
- **Python版本**：推荐使用Python 3.10及以上版本。
- **依赖库**：请参考`requirements.txt`安装所需的依赖库，主要依赖包括：
  - PyTorch
  - NumPy
  - Matplotlib
  - TensorBoard
  - lkh

### 配置方法
1. 确保已安装Python。
2. 安装依赖库：
   ```bash
   pip install -r requirements.txt
   ```
3. 准备数据集：
   - 数据集存储在`data/`目录中，支持`.pkl`和`.tsp`格式。
   - 使用`generate_data.py`生成随机数据集，或参考其代码将自定义问题实例转换为兼容格式。

---

## 项目使用方式

### 1. 生成数据集
- 使用`generate_data.py`生成随机数据集：
  ```bash
  python generate_data.py
  ```
- 如果您有自定义问题实例，请参考`generate_data.py`中的代码，将其转换为适合算法处理的数据文件。

### 2. 训练模型
- **使用随机数据或已有数据进行训练**：
  直接运行`train.py`即可，可在代码中修改随机生成数据或从`.pkl`文件中读取数据：
  ```bash
  python SAPPO/train.py
  ```
  训练过程中会记录指标到TensorBoard日志中，训练完成后模型会保存在`trained_models/`目录中。

- **读取`.tsp`（TSPLIB数据集）文件进行训练**：
  运行`train_public.py`：
  ```bash
  python SAPPO/train_public.py
  ```
  该脚本会读取TSPLIB数据集（`.tsp`文件）并进行训练。

### 3. 推理过程
- 在完成训练后，`trained_models/`目录中会保存训练好的模型。
- **推理前请注意**：
  请确认推理时设置的决策区域长度与子决策区域长度参数，与模型名称后两位数字是否保持一致。若参数不匹配，加载过程将报错。
- **对`.pkl`格式数据进行推理**：
  修改`SAPPO/infer.py`中的参数（如数据路径、节点数量、决策区域长度与子决策区域长度等），然后运行：
  ```bash
  python SAPPO/infer.py
  ```
- **对TSPLIB格式数据进行推理**：
  修改`SAPPO/infer_public.py`中的参数（如数据路径、节点数量、决策区域长度与子决策区域长度等），然后运行：
  ```bash
  python SAPPO/infer_public.py
  ```

### 注意事项
- 文件名中带有`public`的脚本（如`train_public.py`和`infer_public.py`）是针对TSPLIB格式数据兼容的版本，可以直接读取`.tsp`文件。
- 推理结果会保存在`SAPPO/result_info/`目录中，包括路径时间、奖励等统计信息。

---

## 扩展功能

为了证明该算法经过简单修改即可解决其他问题，我们提供了两个额外的功能模块：

- **mTSPD**：此文件夹包含了解决多车辆每辆车配备一架无人机问题的代码。用法及所需数据与SAPPO一致，可直接复用SAPPO训练出的模型。
- **no_fly**：此文件夹包含了解决带有禁飞区约束的TSPD问题的代码。同样，其用法也与SAPPO一致，可直接复用SAPPO训练出的模型，但其数据需要额外提供一个包含禁飞节点索引的列表。

这两个模块展示了算法的扩展性和灵活性，能够适应不同类型的问题。

---

## 数据集说明

本仓库提供了算法代码及其运行环境，以及生成随机数据的相关代码以供参考输入数据格式。
本项目为论文 [A two-stage framework based on RL for Truck-Drone Collaborative Delivery Problem]([https://ieeexplore.ieee.org/document/11288973]) 的官方开源实现，如有引用请使用以下 BibTeX：
```bibtex
@ARTICLE{11288973,
  author={Li, Yuanbo and Zhang, Chengwei and Liu, Wanting and Li, Chao and An, Dou and Wang, Qi},
  journal={IEEE Internet of Things Journal}, 
  title={A two-stage framework based on RL for Truck-Drone Collaborative Delivery Problem}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Drones;Optimization;Collaboration;Training;Scalability;Vehicle dynamics;Urban areas;Traveling salesman problems;Payloads;Logistics;Traveling salesman problem;Vehicle routing;Drones;Reinforcement learning},
  doi={10.1109/JIOT.2025.3641887}}
