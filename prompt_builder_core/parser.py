from __future__ import annotations
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from .schema import Layout, LayoutElement
import yaml


# Source discovery order when both formats exist.
LAYOUT_FILENAMES = ('layout.yaml', 'layout.yml', 'layout.json', 'layout.xml')


def find_layout_file(search_dir: str) -> Optional[str]:
    for name in LAYOUT_FILENAMES:
        path = os.path.join(search_dir, name)
        if os.path.isfile(path):
            return path
    return None


# Simple mtime-keyed cache so repeated evals / layout requests don't re-parse
# the (rarely changing) layout file on every call.
_layout_cache: dict = {}

def _coerce_xml_value(v: str) -> Any:
    v = v.strip()
    if v.lower() in ('true', 'yes', 'on'):
        return True
    if v.lower() in ('false', 'no', 'off'):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _xml_list_items(el: ET.Element) -> List[Any]:
    items = []
    # Child <item> elements
    for child in el:
        if child.tag.lower() in ('item', 'value', 'entry'):
            text = (child.text or '').strip()
            if text:
                items.append(_coerce_xml_value(text))
            # allow nested attributes? fall back to 'value' attr
            if not text and 'value' in child.attrib:
                items.append(_coerce_xml_value(child.attrib['value']))
    # Comma-separated items attribute
    if not items and 'items' in el.attrib:
        for part in el.attrib['items'].split(','):
            part = part.strip()
            if part:
                items.append(_coerce_xml_value(part))
    # Fallback: split text content by lines/commas
    if not items and (el.text or '').strip():
        for part in (el.text or '').replace(',', '\n').splitlines():
            part = part.strip()
            if part:
                items.append(_coerce_xml_value(part))
    return items


def _xml_element_to_dict(el: ET.Element) -> Any:
    tag = el.tag.split('}')[-1].lower()
    # Wrapper elements handled by callers; but be tolerant.
    if tag in ('lists', 'listcontainer'):
        result_lists = {}
        for child in el:
            name = child.attrib.get('name') or child.attrib.get('id')
            if not name:
                continue
            result_lists[name] = _xml_list_items(child)
        return {'_lists': result_lists}
    if tag in ('tabs', 'tabcontainer'):
        return [_xml_element_to_dict(c) for c in el]
    # Generic element
    data: Dict[str, Any] = {}
    # type: tag name unless overridden
    if 'type' in el.attrib:
        data['type'] = _coerce_xml_value(el.attrib['type'])
    else:
        data['type'] = tag
    if 'name' in el.attrib:
        data['name'] = _coerce_xml_value(el.attrib['name'])
    if 'i' in el.attrib and 'name' not in el.attrib:
        data['name'] = _coerce_xml_value(el.attrib['i'])
    # Other attributes -> data (excluding structural wrappers handled above)
    for k, v in el.attrib.items():
        if k.lower() in ('type', 'name', 'i'):
            continue
        data[k] = _coerce_xml_value(v)
    # Children -> children list, unless inside lists/tabs wrappers (handled above)
    children = []
    for child in el:
        ctag = child.tag.split('}')[-1].lower()
        if ctag in ('lists', 'tabs'):
            # merge their content into this element's level
            converted = _xml_element_to_dict(child)
            if isinstance(converted, dict) and '_lists' in converted:
                data.setdefault('_lists', {}).update(converted['_lists'])
            elif isinstance(converted, list):
                children.extend(converted)
            continue
        children.append(_xml_element_to_dict(child))
    if children:
        data['children'] = children
    # If this element itself is a list definition (tag 'list') return handled by parent;
    # but if encountered standalone, convert to lists dict marker.
    if tag == 'list':
        return {'_list': (data.get('name'), _xml_list_items(el))}
    return data


def _xml_to_dict(path: str) -> Dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    # Determine lists
    lists: Dict[str, Any] = {}
    tabs: List[Any] = []
    # If root has <lists> / <tabs>
    for child in root:
        ctag = child.tag.split('}')[-1].lower()
        if ctag in ('lists', 'listcontainer'):
            converted = _xml_element_to_dict(child)
            if isinstance(converted, dict) and '_lists' in converted:
                lists.update(converted['_lists'])
        elif ctag in ('tabs', 'tabcontainer'):
            tabs.extend(_xml_element_to_dict(child) if isinstance(_xml_element_to_dict(child), list) else [_xml_element_to_dict(child)])
        elif ctag == 'list':
            name = child.attrib.get('name') or child.attrib.get('id')
            if name:
                lists[name] = _xml_list_items(child)
        elif ctag in ('tab',):
            tabs.append(_xml_element_to_dict(child))
        else:
            # Could be a top-level element; wrap into a default tab later
            tabs.append(_xml_element_to_dict(child))
    # Root attributes
    result: Dict[str, Any] = {}
    if root.attrib.get('title'):
        result['title'] = _coerce_xml_value(root.attrib['title'])
    # Allow lists/tabs as attributes? no.
    # If no tabs wrapper but root contains tab children handled above.
    # If lists found via nested _lists inside tabs/elements, collect
    # (Walk a copy to extract any _lists markers)
    def collect_lists(node):
        if isinstance(node, dict):
            if '_lists' in node:
                lists.update(node.pop('_lists'))
            if '_list' in node:
                name, items = node.pop('_list')
                if name:
                    lists[name] = items
            for v in list(node.values()):
                collect_lists(v)
        elif isinstance(node, list):
            for v in node:
                collect_lists(v)
    collect_lists(tabs)
    # Remove any leftover _lists/_list markers
    def clean(node):
        if isinstance(node, dict):
            node.pop('_lists', None)
            node.pop('_list', None)
            for v in node.values():
                clean(v)
        elif isinstance(node, list):
            for v in node:
                clean(v)
    clean(tabs)
    result['lists'] = lists
    # Ensure tabs have type tab
    for t in tabs:
        if isinstance(t, dict) and 'type' not in t:
            t['type'] = 'tab'
        if isinstance(t, dict):
            t.setdefault('type', 'tab')
    result['tabs'] = tabs
    if not result.get('tabs'):
        result['tabs'] = []
    return result


def parse_layout_file(path: str) -> Layout:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    cache_key = (os.path.abspath(path), mtime)
    cached = _layout_cache.get(cache_key)
    if cached is not None:
        return cached
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f) or {}
    elif path.lower().endswith('.xml'):
        data = _xml_to_dict(path)
    else:
        with open(path, 'r', encoding='utf-8') as f:
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
