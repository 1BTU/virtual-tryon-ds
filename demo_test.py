import os
import yaml

file_path = 'config.yaml'

# 判断文件是否存在
if os.path.exists(file_path):
    # 读取 YAML 文件
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
else:
    # 创建新的数据结构
    data = {}

# 判断变量是否存在，不存在则添加
if 'new_parameter' not in data:
    data['new_parameter'] = 'new_value'

# 修改变量
data['parameter1'] = 'new_value1'
data['parameter2'] = 'new_value2'

# 将数据写回 YAML 文件
with open(file_path, 'w') as file:
    yaml.dump(data, file)