"""Discover drop-in format adapters; orchestration has no format branches."""
from dataclasses import dataclass, field
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import Any


@dataclass
class InputState:
    context: Any = None
    cache_config: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)


def available_formats() -> dict[str, ModuleType]:
    from . import formats
    result = {}
    for entry in sorted(iter_modules(formats.__path__), key=lambda item: item.name):
        if entry.ispkg or entry.name.startswith('_'):
            continue
        adapter = import_module(f'{formats.__name__}.{entry.name}')
        required = ('FORMAT_NAME', 'FILENAME_REGEX', 'DISPLAY_RANGE', 'prepare', 'options', 'load_frame')
        if not all(hasattr(adapter, name) for name in required):
            raise ValueError(f'Input adapter {entry.name} is missing required interface members')
        if adapter.FORMAT_NAME in result:
            raise ValueError(f'Duplicate input format: {adapter.FORMAT_NAME}')
        result[adapter.FORMAT_NAME] = adapter
    return result


def get_format(name: str) -> ModuleType:
    try:
        return available_formats()[name]
    except KeyError as error:
        raise ValueError(f'Unknown input format: {name}') from error
