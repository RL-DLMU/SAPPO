import os
import torch
import lkh
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

TSP_NAME = "CityWalk"
# DATA_PATH = "data.txt"
CITY_NUM = 10

class LKH_Solver():
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.result = None

    def reset(self, coordinates):
        self.coordinates = coordinates
        self.result = None

    def clac_distance(self, X, Y):
        """
        计算两个城市之间的欧氏距离，二范数
        :param X: 城市X的坐标.np.array数组
        :param Y: 城市Y的坐标.np.array数组
        :return:
        """
        city_num = X.shape[0]
        distance_matrix = torch.zeros((city_num, city_num))
        for i in range(city_num):
            for j in range(city_num):
                if i == j:
                    continue

                distance = torch.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
                distance_matrix[i][j] = distance

        return distance_matrix

    def writeTSPLIBfile_FE(self, fname_tsp,CostMatrix,user_comment, tsplib_dir, pwd):
        dims_tsp = len(CostMatrix)
        name_line = 'NAME : ' + fname_tsp + '\n'
        type_line = 'TYPE: TSP' + '\n'
        comment_line = 'COMMENT : ' + user_comment + '\n'
        tsp_line = 'TYPE : ' + 'TSP' + '\n'
        dimension_line = 'DIMENSION : ' + str(dims_tsp) + '\n'
        edge_weight_type_line = 'EDGE_WEIGHT_TYPE : ' + 'EXPLICIT' + '\n' # explicit only
        edge_weight_format_line = 'EDGE_WEIGHT_FORMAT: ' + 'FULL_MATRIX' + '\n'
        display_data_type_line ='DISPLAY_DATA_TYPE: ' + 'NO_DISPLAY' + '\n' # 'NO_DISPLAY'
        edge_weight_section_line = 'EDGE_WEIGHT_SECTION' + '\n'
        eof_line = 'EOF\n'
        Cost_Matrix_STRline = []
        for i in range(0,dims_tsp):
            cost_matrix_strline = ''
            for j in range(0,dims_tsp-1):
                cost_matrix_strline = cost_matrix_strline + str(int(CostMatrix[i][j])) + ' '

            j = dims_tsp-1
            cost_matrix_strline = cost_matrix_strline + str(int(CostMatrix[i][j]))
            cost_matrix_strline = cost_matrix_strline + '\n'
            Cost_Matrix_STRline.append(cost_matrix_strline)
        
        fileID = open((pwd + tsplib_dir + fname_tsp + '.tsp'), "w")
        fileID.write(name_line)
        fileID.write(comment_line)
        fileID.write(tsp_line)
        fileID.write(dimension_line)
        fileID.write(edge_weight_type_line)
        fileID.write(edge_weight_format_line)
        fileID.write(edge_weight_section_line)
        for i in range(0,len(Cost_Matrix_STRline)):
            fileID.write(Cost_Matrix_STRline[i])

        fileID.write(eof_line)
        fileID.close()

        parameters = dict()
        parameters['problem_file'] = pwd + tsplib_dir + fname_tsp + '.tsp'
        parameters['optimum'] = 378032
        parameters['move_type'] = 5
        parameters['patching_c'] = 3
        parameters['patching_a'] = 2
        parameters['runs'] = 10
        # parameters['tour_file'] = fname_tsp + '.txt' # 这一项有的话就会生成输出文件
        return parameters

    def get_distance(self, x, distance):
        distances = torch.zeros((len(x)-1, ))
        for i in range(len(x)-1):
            distances[i] = distance[x[i]][x[i + 1]]
        return distances

    def solve(self):
        """
        coordinates 参数可以接收一个表示点坐标的NumPy数组（即由调用者提供的TSP问题），
        或者只传递一个整数，由函数随机生成相应规模的TSP问题。
        """
        
        city_x=self.coordinates[:,0]
        city_y=self.coordinates[:,1]

        #城市数量
        CostMatrix = self.clac_distance(city_x, city_y)*1000    #将距离矩阵放大1000倍（LKH算法只能处理整数）
        distance = self.clac_distance(city_x, city_y)

        user_comment = "a comment by the user"

        tsplib_dir = '\\TSPLIB\\'
        pwd=os.getcwd()
        parameters = self.writeTSPLIBfile_FE("tsp",CostMatrix,user_comment, tsplib_dir, pwd)
        solver_path = 'LKH-3.0.11/LKH.exe'
        result = [x-1 for x in lkh.solve(solver_path, **parameters)[0]]
        result.append(result[0])
        self.result = result
        distances = self.get_distance(result, distance)
        return {"total_distance":torch.sum(distances).item(), "result": result, "distances": distances}

    def draw(self):
        assert self.result is not None
        city_x=self.coordinates[:,0]
        city_y=self.coordinates[:,1]
        lines = []
        for i in range(len(self.result)-1):
            line_coord = [(city_x[self.result[i]], city_y[self.result[i]]), (city_x[self.result[i+1]], city_y[self.result[i+1]])]
            lines.append(line_coord)
        
        line_segments = LineCollection(lines, colors='blue', linewidths=0.5)
        # 创建绘图
        fig, ax = plt.subplots()
        # 添加线段到绘图中
        ax.add_collection(line_segments)
        # 添加点到绘图中
        ax.scatter(city_x, city_y, color='red', linewidths=0.5)
        ax.scatter(city_x[0], city_y[0], color='yellow', linewidths=0.5)
        # 添加每个点的ID作为标签
        for i in range(len(self.coordinates)):
            ax.text(city_x[i], city_y[i], str(i), fontsize=9, ha='right', color='black')
        # 设置坐标范围
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        # 添加标题和标签
        ax.set_title(f'{TSP_NAME}')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        # 显示绘图
        plt.show(block=True)

if __name__ == "__main__":
    coordinates = 50 * torch.rand((CITY_NUM, 2), dtype=torch.float)
    # torch.save(coordinates, "data.pt")
    # coordinates = torch.load("data.pt")
    lkh_solver = LKH_Solver(coordinates)
    result = lkh_solver.solve()
    print(result)
    lkh_solver.draw()
