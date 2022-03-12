import yaml

from src.argparser import get_softrules_argparser
from typing import Union, List

class Config:
    def __init__(self, config, basepath = ""):
        self.config   = config
        self.basepath = basepath

    """
    The reason behind accessing the config dictionary through this method is to discourage
     accessing it directly wherever needed in code. This is done because accessing it like:
     >>> config_object.config.get("param_name", default_value)
    can hide errors. "param_name" might not be in the config (or might be mispelled)
    """
    def get(self, param):
        if param in self.config:
            result = self.config[param]
            if isinstance(result, dict):
                return Config(result, self.basepath)
            else:
                return result
        else:
            error_str = f"The parameter {param} is not in the config, which contains the following keys: {list(self.config.keys())}.\nThe full config is: {self.config}"
            raise ValueError(error_str)

    """
    Use for path parameters. Useful because it can append a basepath
    The idea is to specify relative paths as much as possible, and only
    a single basepath. Then, only the basepath needs to be changed on other
    systems
    """
    def get_path(self, param):
        path = self.get(param)
        if isinstance(path, str):
            return self.basepath + "/" + path
        else:
            raise ValueError(f"Is {path} a string? Only strings are supported as paths.")

    def contains(self, param):
        return param in self.config

    """

    """
    @staticmethod
    def get_config(paths: Union[str, List[str]] = "config/default_config.yaml"):
        if isinstance(paths, List):
            config = {}
            for path in paths:
                with open(path) as f:
                    new_config = yaml.load(f, Loader=yaml.FullLoader)
                    config.update(new_config)
        else:
            with open(paths) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

        return Config(config)

    @staticmethod
    def parse_args_and_get_config():
        import sys
        parser = get_softrules_argparser()
        args   = parser.parse_args(sys.argv[1:])
        args   = vars(args)
        args   = {k:v for (k, v) in args.items() if v is not None}

        config = {}
        for path in args['path']:
            with open(path) as f:
                new_config = yaml.load(f, Loader=yaml.FullLoader)
                config.update(new_config)

        config.update(args)

        return Config(config, config['basepath'])


# python -m src.config
if __name__ == "__main__":
    c = Config.parse_args_and_get_config()
    print(c.config)