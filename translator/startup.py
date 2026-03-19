import sys as _sys

from .runtime import startup as _module

_sys.modules[__name__] = _module
