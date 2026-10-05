#!/usr/bin/env python3
"""
Convert legacy layout.txt format to new layout.yaml format.

Legacy format (from A1111 script):
  TYPE --arg value --arg2 value2
  
Types:
  SINGLE, DUAL, EDIT, EDIT_LINK, SELECT, PRESET, SET, VALUE,
  GROUP, TAB, ROW, COLUMN, ACCORDION, SEPARATOR, LIST, ENTRY (LIST_ITEM), FROM_LIST, END

Arguments:
  --i: name
  --a: activated (1/0)
  --scale: scale factor
  --sort: sort (1/0)
  --open: open (1/0)
  --p: prompt
  --s: emphasis
  --pp: prompt_pos
  --sp: emphasis_pos
  --pn: prompt_neg
  --sn: emphasis_neg
  --r: edit (0-1)
  --pa: prompt_a
  --pb: prompt_b
  --prefix: prefix
  --postfix: postfix
  --link: linked element name
  --n: is_negative (1/0)
  --add: is_additive (1/0)
  --reset: is_reset_visible (1/0)
  --x: ignore (1/0)

Usage:
  python convert_layout.py layout.txt layout.yaml
"""

import sys
import re
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class LayoutElement:
    type: str
    name: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    children: List['LayoutElement'] = field(default_factory=list)


# Legacy type -> YAML type mapping
TYPE_MAP = {
    'SINGLE': 'single',
    'DUAL': 'dual',
    'EDIT': 'edit',
    'EDIT_LINK': 'edit_link',
    'SELECT': 'select',
    'PRESET': 'preset',
    'SET': 'set',  # handled specially - child of preset/select choice
    'VALUE': 'value',  # handled specially - child of SET
    'GROUP': 'group',
    'TAB': 'tab',
    'ROW': 'row',
    'COLUMN': 'column',
    'ACCORDION': 'accordion',
    'SEPARATOR': 'separator',
    'LIST': 'list',  # becomes lists dict
    'ENTRY': 'from_list',  # becomes from_list items
    'FROM_LIST': 'from_list',
    'END': 'end',  # structural
}

# Argument name mapping
ARG_MAP = {
    'i': 'name',
    'a': 'activated',
    'scale': 'scale',
    'sort': 'sort',
    'open': 'open',
    'p': 'prompt',
    's': 'emphasis',
    'pp': 'prompt_pos',
    'sp': 'emphasis_pos',
    'pn': 'prompt_neg',
    'sn': 'emphasis_neg',
    'r': 'edit',
    'pa': 'prompt_a',
    'pb': 'prompt_b',
    'prefix': 'prefix',
    'postfix': 'postfix',
    'link': 'link',
    'n': 'is_negative',
    'add': 'is_additive',
    'reset': 'is_reset_visible',
    'x': 'ignore',
}


def parse_args(arg_str: str) -> Dict[str, str]:
    """Parse --arg value --arg2 value2 string into dict."""
    args = {}
    # Split by -- but keep the -- prefix
    parts = re.split(r'\s*--(\w+)', arg_str)
    # parts[0] is before first --, then alternating: arg_name, value, arg_name, value...
    current_arg = None
    for part in parts[1:]:
        part = part.strip()
        if current_arg is None:
            current_arg = part
        else:
            args[current_arg] = part
            current_arg = None
    return args


def convert_value(key: str, value: str) -> Any:
    """Convert string value to appropriate type."""
    # Boolean args
    if key in ('a', 'sort', 'open', 'n', 'add', 'reset', 'x'):
        return value.lower() in ('true', '1', 'yes', 'on')
    # Try int first (handles numeric emphasis, scale, etc.)
    try:
        return int(value)
    except ValueError:
        pass
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    # Boolean-like strings
    if value.lower() in ('true', 'yes', 'on'):
        return True
    if value.lower() in ('false', 'no', 'off'):
        return False
    return value


def parse_line(line: str) -> Optional[LayoutElement]:
    """Parse a single legacy format line into LayoutElement."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    # Find first -- for args
    idx = line.find('--')
    if idx > 0:
        type_str = line[:idx].strip().upper()
        arg_str = line[idx:]
    elif idx == 0:
        type_str = ''
        arg_str = line
    else:
        type_str = line.upper()
        arg_str = ''
    
    if type_str not in TYPE_MAP:
        print(f"Warning: Unknown type '{type_str}' in line: {line}")
        return None
    
    yaml_type = TYPE_MAP[type_str]
    # 'end' elements are still returned, handled by stack logic
    
    # Parse arguments
    raw_args = parse_args(arg_str)
    data = {}
    for k, v in raw_args.items():
        if k in ARG_MAP:
            data[ARG_MAP[k]] = convert_value(k, v)
        else:
            data[k] = convert_value(k, v)
    
    # Special handling for certain types
    if yaml_type in ('list',):
        # LIST defines a list in the lists dict
        # We'll handle this separately
        pass
    
    name = data.pop('name', None)
    if name == '':
        name = None
    
    return LayoutElement(type=yaml_type, name=name, data=data)


def convert_layout(input_path: str, output_path: str):
    """Convert legacy layout.txt to YAML format."""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # First pass: parse all lines into elements with line numbers
    elements = []
    for line in lines:
        elem = parse_line(line)
        if elem:
            elements.append(elem)
    
    # Second pass: build hierarchy using stack with container tracking
    root = LayoutElement(type='root', name=None)
    stack = [(root, 'root')]  # (element, container_type)
    lists = {}  # name -> list of strings
    current_list = None
    
    for elem in elements:
        # Handle END - pop appropriate container
        if elem.type == 'end':
            # Pop until we close a container (not root)
            while len(stack) > 1:
                _, container_type = stack.pop()
                if container_type in ('tab', 'select', 'group', 'row', 'column', 'accordion'):
                    break
            continue
        
        # Handle LIST
        if elem.type == 'list':
            if elem.name:
                current_list = elem.name
                lists[current_list] = []
            continue
        
        # Handle FROM_LIST as list entry (ENTRY)
        if elem.type == 'from_list' and elem.data.get('prompt'):
            if current_list and current_list in lists:
                lists[current_list].append(elem.data['prompt'])
            continue
        
        # Normal element - add to current container
        parent, _ = stack[-1]
        parent.children.append(elem)
        
        # Container types push to stack
        if elem.type in ('tab', 'select', 'group', 'row', 'column', 'accordion'):
            stack.append((elem, elem.type))
    
    # Build final YAML structure
    tabs = []
    for child in root.children:
        if child.type == 'tab':
            tabs.append(elem_to_dict(child))
        else:
            if not tabs:
                tabs.append({
                    'name': 'Main',
                    'type': 'tab',
                    'children': []
                })
            tabs[-1]['children'].append(elem_to_dict(child))
    
    # Ensure all tabs have 'type': 'tab'
    for tab in tabs:
        tab['type'] = 'tab'
    
    output = {
        'title': 'Prompt Builder',
        'lists': lists,
        'tabs': tabs
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Converted {input_path} -> {output_path}")
    print(f"  Lists: {list(lists.keys())}")
    print(f"  Tabs: {[t['name'] for t in tabs]}")


def elem_to_dict(elem: LayoutElement) -> Dict[str, Any]:
    """Convert LayoutElement to dict for YAML."""
    result = {'type': elem.type}
    if elem.name:
        result['name'] = elem.name
    if elem.data:
        result.update(elem.data)
    if elem.children:
        result['children'] = [elem_to_dict(c) for c in elem.children]
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_layout.py input.txt [output.yaml]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.txt', '.yaml')
    convert_layout(input_path, output_path)