from __future__ import annotations
import json
import os
from typing import Dict, Any, List, Optional
from .schema import Layout, LayoutElement
import yaml


# Source discovery order when both formats exist.
LAYOUT_FILENAMES = ('layout.json', 'layout.yaml', 'layout.yml')


def find_layout_file(search_dir: str) -> Optional[str]:
    for name in LAYOUT_FILENAMES:
        path = os.path.join(search_dir, name)
        if os.path.isfile(path):
            return path
    return None


def parse_layout_file(path: str) -> Layout:
    with open(path, 'r', encoding='utf-8') as f:
        if path.lower().endswith('.json'):
            data = json.load(f) or {}
        else:
            data = yaml.safe_load(f) or {}
    return _from_dict(data)


def _from_dict(data: Dict[str, Any]) -> Layout:
    layout = Layout(
        title=data.get('title', 'Prompt Builder'),
        lists=data.get('lists', {}) or {},
    )
    layout.tabs = [_element_from_dict(t) for t in (data.get('tabs') or [])]
    return layout


def _element_from_dict(d: Dict[str, Any]) -> LayoutElement:
    elem = LayoutElement(
        type=d.get('type', ''),
        name=d.get('name'),
        data={k: v for k, v in d.items() if k not in ('type', 'name', 'children')},
    )
    elem.children = [_element_from_dict(c) for c in (d.get('children') or [])]
    return elem
