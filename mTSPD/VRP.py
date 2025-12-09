from scipy.spatial.distance import cdist

def vrp_clarke_wright(nodes, m):
    """
    使用Clarke-Wright节约算法解决VRP问题。

    参数：
    - nodes: numpy数组，形状为(n, 2)，n为节点数量，列为x和y坐标。
             节点0为仓库。
    - m: 整数，车辆数量。

    返回：
    - 列表的列表：每个子列表包含两部分，第一部分为该路由的总距离，第二部分为路由列表，从仓库（节点0）开始并结束。
      例如 [[15.25, [0, 1, 2, 0]], [21.34, [0, 3, 4, 0]]]
    """
    n = nodes.shape[0]
    if n < 2:
        raise ValueError("至少需要一个仓库和一个客户节点。")
    if m < 1:
        raise ValueError("车辆数量m必须至少为1。")

    # 计算距离矩阵
    dist = cdist(nodes, nodes, metric='euclidean')

    # 初始化每个客户为单独路由：[0, i, 0]
    routes = [[0, i, 0] for i in range(1, n)]
    num_vehicles = len(routes)

    # 计算节约值列表：s_ij = d(0,i) + d(0,j) - d(i,j)，其中 i < j
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = dist[i, 0] + dist[0, j] - dist[i, j]
            savings.append((s, i, j))

    # 按节约值降序排序
    savings.sort(reverse=True, key=lambda x: x[0])

    # 合并路由直到达到m辆车或无法继续合并
    while num_vehicles > m:
        merged = False
        for idx in range(len(savings)):
            s, i, j = savings[idx]
            # 找到包含i和j的路由
            route_i = next((r for r in routes if i in r), None)
            route_j = next((r for r in routes if j in r), None)
            if route_i is None or route_j is None or route_i is route_j:
                continue

            # 检查是否可以合并：i在route_i末尾，j在route_j开头
            if route_i[-2] == i and route_j[1] == j:
                # 合并route_i + route_j（移除i的末尾0和j的开头0）
                new_route = route_i[:-1] + route_j[1:]
                routes.remove(route_i)
                routes.remove(route_j)
                routes.append(new_route)
                merged = True
                num_vehicles -= 1
                # 删除已使用的节约值
                del savings[idx]
                break
            # 或反向：j在route_j末尾，i在route_i开头
            elif route_j[-2] == j and route_i[1] == i:
                # 合并route_j + route_i
                new_route = route_j[:-1] + route_i[1:]
                routes.remove(route_j)
                routes.remove(route_i)
                routes.append(new_route)
                merged = True
                num_vehicles -= 1
                del savings[idx]
                break

        if not merged:
            # 无法继续合并
            break

    # 计算每个路由的总距离
    result = []
    for route in routes:
        total_distance = 0.0
        for k in range(len(route) - 1):
            total_distance += dist[route[k], route[k+1]]
        result.append([round(total_distance, 2), route])  # 保留两位小数

    return result