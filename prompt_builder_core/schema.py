from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class LayoutElement:
    type: str
    name: Optional[str] = None
    children: List['LayoutElement'] = field(default_factory=list)
    # Additional props stored in data
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Layout:
    title: str = 'Prompt Builder'
    lists: Dict[str, List[str]] = field(default_factory=dict)
    tabs: List[LayoutElement] = field(default_factory=list)
