import yaml


def get_config(path="configs/config.yaml"):
    with open(path) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg
