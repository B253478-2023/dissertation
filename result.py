import pandas as pd


def extract_nmi_to_csv(input_file, output_file):
    """
    从输入的txt文件中提取方法名和NMI值，转换为百分数并保留两位小数，然后保存为CSV文件。

    参数:
    input_file (str): 输入txt文件的路径
    output_file (str): 输出csv文件的路径
    """
    # 初始化一个空字典来存储方法名和NMI
    data = {'Method': [], 'NMI (%)': []}

    # 读取文件内容并处理
    with open(input_file, 'r') as file:
        lines = file.readlines()
        header = lines[0].split()
        nmi_index = header.index('Normalized') + 1  # 找到NMI列的索引

        for line in lines[1:]:
            values = line.split()
            method = values[0]
            nmi = float(values[nmi_index]) * 100  # 转换为百分数
            data['Method'].append(method)
            data['NMI (%)'].append(f'{nmi:.2f}')  # 保留两位小数

    # 转换为DataFrame
    df = pd.DataFrame(data)

    # 保存为CSV文件
    df.to_csv(output_file, index=False)

    print(f'已保存为 {output_file}')


# 使用示例
input_file = 'n100_c3_it20_disp20.txt'
output_file = 'n100_c3_it20_disp20.csv'
extract_nmi_to_csv(input_file, output_file)
