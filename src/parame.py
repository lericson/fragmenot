import sys
import yaml
import warnings
from functools import wraps
from os import path, environ
from inspect import signature


# Environment configuration registry. Indirect so we can replace it.
_environ = environ

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
        return envvar in _environ or name in self.file_config

    def __getitem__(self, name):
        envvar = format_envvar(self.name, name)
        if envvar in _environ:
            return yaml.safe_load(_environ[envvar])
        if name in self.file_config:
            return self.file_config[name]
        raise KeyError(name)

    def get(self, name, default=None):
        envvar = format_envvar(self.name, name)
        if envvar in _environ:
            return yaml.safe_load(_environ[envvar])
        return self.file_config.get(name, default)


def get(modname, varname, default=None):
    return Module(modname).get(varname, default)


def configurable(f):
    """Decorator to replace default keyword arguments with config values.

    This means that if you change the configuration during runtime (somehow),
    this will NOT be reflected. See *reconfigurable*.
    """
    sig = signature(f)
    f.cfg = {}
    for name, fparam in sig.parameters.items():
        if isinstance(fparam.annotation, Param):
            par = fparam.annotation
            val = par.cfg.get(name, fparam.default)
            f.__kwdefaults__[name] = val
    return f


def reconfigurable(f):
    "Decorator to replace default keyword arguments _on invocation_."
    @wraps(f)
    def wrap(*a, **k):
        return configurable(f)(*a, **k)
    return wrap


cfg = Module(__name__)


@configurable
def _load_file_cfg(*, load: cfg.param = 'parame.yaml'):
    if path.exists(load):
        with open(load) as f:
            return yaml.safe_load(f)
    return {}


_file_cfg.update(_load_file_cfg())


@configurable
def set_profile(*,
                profile: cfg.param = 'default',
                verbose: cfg.param = True):
    if isinstance(profile, str):
        profile = [profile]
    for prof in profile:
        if verbose:
            print('loading profile', prof, file=sys.stderr)
        for modname, d in cfg[prof].items():
            if verbose:
                print(' ', modname, d, file=sys.stderr)
            _file_cfg.setdefault(modname, {}).update(d)


set_profile()
