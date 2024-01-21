import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, yaml.Loader)
    return config


if __name__ == '__main__':
    # 指定你的YAML配置文件路径
    config_path = 'evalute1.yaml'

    # 载入配置
    config = load_config(config_path)

    # 获取参数
    name = config['name']
    batch_size = config['batch_size']
    lr = config['lr']
    # 按照需要获取其他参数

    # 打印参数
    print(f'name: {name}')
    print(f'batch_size: {batch_size}')
    print(f'lr: {lr}')
    # 输出其他参数