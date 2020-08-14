import os
import yaml
import warnings
import logging
from os import path
from inspect import signature


log = logging.getLogger(__name__)

# Global configuration registry, used as 2nd source after env vars.
_file_cfg = {}


class Param():
    "Type annotator indicating argument is configurable"

    def __init__(self, cfg):
        self.cfg = cfg


def format_envvar(modname, parname):
    return f"{modname.upper().replace('.', '_')}_{parname.upper()}"


class Module():
    def __init__(self, name):
        if name == '__main__':
            warnings.warn('cannot use parame from __main__ module')
        self.name = name
        self.param = Param(self)

    @property
    def file_config(self):
        return _file_cfg.get(self.name, {})

    def __contains__(self, name):
        envvar = format_envvar(self.name, name)
        return envvar in os.environ or name in self.file_config

    def __getitem__(self, name):
        envvar = format_envvar(self.name, name)
        if envvar in os.environ:
            return eval(os.environ[envvar])
        if name in self.file_config:
            return self.file_config[name]
        raise KeyError(name)

    def get(self, name, default):
        envvar = format_envvar(self.name, name)
        if envvar in os.environ:
            return eval(os.environ[envvar])
        return self.file_config.get(name, default)


def configurable(f):
    sig = signature(f)
    for name, fparam in sig.parameters.items():
        if isinstance(fparam.annotation, Param):
            par = fparam.annotation
            val = par.cfg.get(name, fparam.default)
            f.__kwdefaults__[name] = val
    return f


cfg = Module(__name__)


@configurable
def _load_file_cfg(*, load: cfg.param = 'parame.yaml'):
    if path.exists(load):
        with open(load) as f:
            return yaml.safe_load(f)
    return {}


_file_cfg.update(_load_file_cfg())


@configurable
def set_profile(*, profile: cfg.param = 'default'):
    for modname, d in cfg.get(profile, {}).items():
        _file_cfg.setdefault(modname, {}).update(d)


set_profile()
