import yaml
from train import train_model
from test import test_model

if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if config['mode'] == 'train':
        train_model(config)
    elif config['mode'] == 'test':
        test_model(config)
