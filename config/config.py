from configparser import ConfigParser

config = ConfigParser()

config['preprocess'] = {
        # work donde in filtering d=5 for real time
        'filter_size': '3',
        # should be the same
        # < 10 not much effect
        # > 150 strong effect image look "cartoonish"
        'sigma_color': '75',
        'sigma_space': '75'
        }

config['postprocess'] = {
        'kernelx': '3',
        'kernely': '3',
        }

with open('config.conf', 'w') as f:
    config.write(f)
