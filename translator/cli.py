import sys as _sys

from .app import cli as _module

_sys.modules[__name__] = _module
