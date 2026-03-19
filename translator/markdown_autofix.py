import sys as _sys

from .markdown import autofix as _module

_sys.modules[__name__] = _module
