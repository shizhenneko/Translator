import sys as _sys

from .services import translation_runner as _module

_sys.modules[__name__] = _module
