import sys as _sys

from .markdown import sanitize as _module

_sys.modules[__name__] = _module
