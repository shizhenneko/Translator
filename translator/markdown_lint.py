import sys as _sys

from .markdown import lint as _module

_sys.modules[__name__] = _module
