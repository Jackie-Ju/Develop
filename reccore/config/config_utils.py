import os
import os.path as osp
from pathlib import Path
import ast
import yaml
import importlib.util
import copy
import os
from dotmap import DotMap
from reccore.utils.utils import update_dict


def is_str(x):
    """Whether the input is an string instance.
    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def is_filepath(x):
    return is_str(x) or isinstance(x, Path)


def fopen(filepath, *args, **kwargs):
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)
    raise ValueError('`filepath` should be a string or a Path')


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    else:
        dir_name = osp.expanduser(dir_name)
        os.makedirs(dir_name, mode=mode, exist_ok=True)


def _validate_py_syntax(filename):
    with open(filename, 'r') as f:
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError('There are syntax errors in config '
                          f'file {filename}: {e}')


def load_config_data(global_path, selection_path=None, model_path=None):
    global_cfg_path = global_path
    global_cfg_name = osp.abspath(osp.expanduser(global_cfg_path))
    global_spec = importlib.util.spec_from_file_location("config", global_cfg_name)
    global_mod = importlib.util.module_from_spec(global_spec)
    global_spec.loader.exec_module(global_mod)
    global_config = copy.deepcopy(global_mod.config)
    final_config = global_config
    for path in [selection_path, model_path]:
        if path is None:
            continue
        config_file_name = osp.abspath(osp.expanduser(path))
        check_file_exist(config_file_name)
        fileExtname = osp.splitext(config_file_name)[1]
        if fileExtname not in ['.py', '.yaml', '.yml']:
            raise IOError('Only py/yml/yaml type are supported now!')
        """
        Parsing Config file
        """
        if config_file_name.endswith('.yaml'):
            with open(config_file_name, 'r') as config_file:
                config_data = yaml.load(config_file, Loader=yaml.FullLoader)
        elif config_file_name.endswith('.py'):
            spec = importlib.util.spec_from_file_location("config", config_file_name)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            config_data = copy.deepcopy(mod.config)
            final_config = update_dict(final_config, config_data)
    return DotMap(final_config)
