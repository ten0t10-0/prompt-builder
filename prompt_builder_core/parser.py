from __future__ import annotations
import json
import os
from typing import Dict, Any, List, Optional
from .schema import Layout, LayoutElement
import yaml


# Source discovery order when both formats exist.
LAYOUT_FILENAMES = ('layout.yaml', 'layout.yml', 'layout.json')


def find_layout_file(search_dir: str) -> Optional[str]:
    for name in LAYOUT_FILENAMES:
        path = os.path.join(search_dir, name)
        if os.path.isfile(path):
            return path
    return None


# Simple mtime-keyed cache so repeated evals / layout requests don't re-parse
# the (rarely changing) layout file on every call.
_layout_cache: dict = {}

def parse_layout_file(path: str) -> Layout:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    cache_key = (os.path.abspath(path), mtime)
    cached = _layout_cache.get(cache_key)
    if cached is not None:
        return cached
    with open(path, 'r', encoding='utf-8') as f:
        if path.lower().endswith('.json'):
            data = json.load(f) or {}
        else:
            data = yaml.safe_load(f) or {}
    layout = _from_dict(data)
    # Drop stale entries for the same path to avoid unbounded growth.
    for k in [k for k in _layout_cache if k[0] == os.path.abspath(path) and k != cache_key]:
        del _layout_cache[k]
    _layout_cache[cache_key] = layout
    return layout


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
