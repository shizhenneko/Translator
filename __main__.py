import os
import runpy
import sys


def _run_src_package() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    _ = sys.modules.pop("translator", None)
    _ = runpy.run_module("translator", run_name="__main__")


if __name__ == "__main__":
    _run_src_package()
