import argparse

def get_softrules_argparser():
    parser = argparse.ArgumentParser(description='Read paths to config files (last takes precedence). Can also update parameters with command-line parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', nargs='+', type=str, default = ["config/default_config.yaml"], help='Path(s) to config file(s)')
    parser.add_argument('--basepath', type=str, required=True)
    return parser