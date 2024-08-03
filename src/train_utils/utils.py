import yaml

def save_yaml_ordered(data, filename):
    class OrderedDumper(yaml.Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    OrderedDumper.add_representer(dict, _dict_representer)

    with open(filename, 'w') as file:
        yaml.dump(data, file, Dumper=OrderedDumper, sort_keys=False)