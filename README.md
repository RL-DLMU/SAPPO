# TSP-D强化学习项目使用指南

本项目使用强化学习解决TSP-D（带无人机的旅行商问题）。以下内容介绍了项目的环境要求、配置方法以及使用方式。

---

## 项目环境与配置

### 环境要求
- **操作系统**：Windows/Linux/MacOS（已在Windows 11上测试）。
- **Python版本**：推荐使用Python 3.8及以上版本。
- **依赖库**：请参考`requirements.txt`安装所需的依赖库，主要依赖包括：
  - PyTorch
  - NumPy
  - Matplotlib
  - TensorBoard
  - 其他工具库（如`lkh`用于求解TSP问题）

### 配置方法
1. 确保已安装Python和相关依赖库。
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
- 如果您有自定义问题实例，请参考`generate_data.py`中的代码，将其转换为适合算法处理的格式（`.pkl`）。

### 2. 训练模型
- **使用随机数据或已有进行训练**：
  直接运行`train.py`即可，可在代码中更改数据由随机生成或从`pkl`文件中读取：
  ```bash
  python SAPPO/train.py
  ```
  训练过程中会记录指标到TensorBoard日志中，训练完成后模型会保存在`SAPPO/trained_models/`目录中。

- **读取公共数据集（TSPLIB数据集）格式进行训练**：
  运行`train_public.py`：
  ```bash
  python SAPPO/train_public.py
  ```
  该脚本会读取TSPLIB数据集（`.tsp`文件）并进行训练。

### 3. 推理过程
- 在完成训练后，`SAPPO/trained_models/`目录中会保存训练好的模型。
- **对`.pkl`格式数据进行推理**：
  修改`SAPPO/infer.py`中的参数（如数据路径、节点数量等），然后运行：
  ```bash
  python SAPPO/infer.py
  ```
- **对TSPLIB格式数据进行推理**：
  修改`SAPPO/infer_public.py`中的参数（如数据路径、节点数量等），然后运行：
  ```bash
  python SAPPO/infer_public.py
  ```

### 注意事项
- 文件名中带有`public`的脚本（如`train_public.py`和`infer_public.py`）是针对TSPLIB格式数据兼容的版本，可以直接读取`.tsp`文件。
- 推理结果会保存在`SAPPO/result_info/`目录中，包括路径时间、奖励等统计信息。

---

## 结果分析

- 使用`SAPPO/process_result.py`可以读取推理生成的CSV文件，计算每组数据的最小值，并输出统计信息（如平均值）。

---