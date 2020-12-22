from configparser import ConfigParser

config = ConfigParser()

config['preprocess'] = {
        'filter_size': '5',
        # should be the same
        # < 10 not much effect
        # > 150 strong effect image look "cartoonish"
        'sigma_color': '75',
        'sigma_space': '75'
        }

config['postprocess'] = {
        'kernelx': '5',
        'kernely': '5',
        }

with open('config.conf', 'w') as f:
    config.write(f)
