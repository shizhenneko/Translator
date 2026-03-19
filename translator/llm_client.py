import sys as _sys

from .llm import client as _module

_sys.modules[__name__] = _module
