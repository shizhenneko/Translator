import sys as _sys

from .core import pipeline as _module

_sys.modules[__name__] = _module
