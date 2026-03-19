from __future__ import annotations

from typing import Dict, List, Type, cast


def require_dict(
    value: object,
    label: str,
    error_type: Type[RuntimeError],
    *,
    expected: str = "a dict",
) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise error_type(f"{label} must be {expected}")
    return cast(Dict[str, object], value)


def require_list(
    value: object,
    label: str,
    error_type: Type[RuntimeError],
    *,
    expected: str = "a list",
) -> List[object]:
    if not isinstance(value, list):
        raise error_type(f"{label} must be {expected}")
    return cast(List[object], value)


def require_str(value: object, label: str, error_type: Type[RuntimeError]) -> str:
    if not isinstance(value, str):
        raise error_type(f"{label} must be a string")
    return value


def require_int(value: object, label: str, error_type: Type[RuntimeError]) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise error_type(f"{label} must be an integer")
    return value


def require_bool(value: object, label: str, error_type: Type[RuntimeError]) -> bool:
    if not isinstance(value, bool):
        raise error_type(f"{label} must be a boolean")
    return value


def require_str_list(
    value: object,
    label: str,
    error_type: Type[RuntimeError],
    *,
    allow_none: bool = False,
    allow_str: bool = False,
    expected: str = "a list of strings",
) -> List[str]:
    if allow_none and value is None:
        return []
    if allow_str and isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        raise error_type(f"{label} must be {expected}")
    items = cast(List[object], value)
    values: List[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise error_type(f"{label}[{index}] must be a string")
        values.append(item)
    return values
