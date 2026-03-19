import sys as _sys

from .markdown import normalize as _module

_sys.modules[__name__] = _module
