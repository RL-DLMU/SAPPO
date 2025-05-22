import csv
'''
读取生成的csv结果记录，并计算统计信息，用来填论文中的表
'''
def get_min_values_from_csv(file_path):
    min_values = []
    second_column_data = []

    # 读取CSV文件，跳过表头
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过表头

        for row in reader:
            if len(row) >= 2:  # 确保至少有两列
                try:
                    value = float(row[1])  # 将第二列转换为float
                    second_column_data.append(value)
                except ValueError:
                    continue  # 如果转换失败（比如遇到非数字），就跳过该行

    # 每十个为一组，取每组的最小值
    for i in range(0, len(second_column_data), 10):
        group = second_column_data[i:i+10]
        if group:
            min_values.append(min(group))

    return min_values

file_path = 'SAPPO/result_info/Sample_500nodes_4_3_result.csv'
min_values = get_min_values_from_csv(file_path)
mean = sum(min_values)/len(min_values)
print(mean)
