from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class PromptPair:
    pos: str
    neg: str


class PromptElement:
    def __init__(self, name: str = '', activated: bool = False, **kwargs):
        self.name = name
        self.activated = activated
        self.data = kwargs

    def build_prompt(self) -> PromptPair:
        ptype = self.data.get('type', 'single')
        if ptype == 'single':
            prompt = self.data.get('prompt', '')
            emphasis = self.data.get('emphasis', 1) or 1
            neg = self.data.get('is_negative', False)
            prefix = self.data.get('prefix', '') or ''
            postfix = self.data.get('postfix', '') or ''
            if prompt:
                if prefix:
                    prompt = prefix.strip() + ' ' + prompt.strip()
                if postfix:
                    prompt = prompt.strip() + ' ' + postfix.strip()
            # If emphasis is 0, omit the prompt
            if emphasis <= 0:
                prompt = ''
            elif emphasis != 1 and prompt:
                prompt = '(' + prompt + ':' + str(emphasis) + ')'
            if neg:
                return PromptPair('', prompt)
            return PromptPair(prompt, '')
        elif ptype == 'dual':
            pp = self.data.get('prompt_pos', '')
            pn = self.data.get('prompt_neg', '')
            sp = self.data.get('emphasis_pos', 1) or 1
            sn = self.data.get('emphasis_neg', 1) or 1
            prefix = self.data.get('prefix', '') or ''
            postfix = self.data.get('postfix', '') or ''
            if pp:
                if prefix:
                    pp = prefix.strip() + ' ' + pp.strip()
                if postfix:
                    pp = pp.strip() + ' ' + postfix.strip()
            if pn:
                if prefix:
                    pn = prefix.strip() + ' ' + pn.strip()
                if postfix:
                    pn = pn.strip() + ' ' + postfix.strip()
            # If emphasis is 0, omit the prompt
            if sp <= 0:
                pp = ''
            elif sp != 1 and pp:
                pp = '(' + pp + ':' + str(sp) + ')'
            if sn <= 0:
                pn = ''
            elif sn != 1 and pn:
                pn = '(' + pn + ':' + str(sn) + ')'
            return PromptPair(pp, pn)
        elif ptype == 'edit' or ptype == 'edit_link':
            pa = self.data.get('prompt_a', '')
            pb = self.data.get('prompt_b', '')
            e_raw = self.data.get('edit', 0.5)
            try:
                e = float(e_raw) if e_raw is not None else 0.5
            except (ValueError, TypeError):
                e = 0.5
            neg = self.data.get('is_negative', False)
            prefix = self.data.get('prefix', '') or ''
            postfix = self.data.get('postfix', '') or ''
            if pa:
                if prefix:
                    pa = prefix.strip() + ' ' + pa.strip()
                if postfix:
                    pa = pa.strip() + ' ' + postfix.strip()
            if pb:
                if prefix:
                    pb = prefix.strip() + ' ' + pb.strip()
                if postfix:
                    pb = pb.strip() + ' ' + postfix.strip()
            # Static blend: weighted prompts (ComfyUI native syntax)
            # At extremes (0 or 1), only show one prompt
            if e <= 0.001:
                prompt = pa
            elif e >= 0.999:
                prompt = pb
            else:
                wa = round(1 - e, 2)
                wb = round(e, 2)
                parts = []
                if pa and wa > 0:
                    parts.append('(' + pa + ':' + str(wa) + ')')
                if pb and wb > 0:
                    parts.append('(' + pb + ':' + str(wb) + ')')
                prompt = ', '.join(parts)
            if neg:
                return PromptPair('', prompt)
            return PromptPair(prompt, '')
        return PromptPair('', '')


class PromptEvaluator:
    def __init__(self, state=None, active_prompts=None):
        self.state = state
        self.active_prompts = active_prompts or []

    def build_prompts(self) -> PromptPair:
        pos_parts = []
        neg_parts = []
        for p in self.active_prompts:
            try:
                pr = p.build_prompt()
                if pr.pos:
                    pos_parts.append(pr.pos)
                if pr.neg:
                    neg_parts.append(pr.neg)
            except Exception:
                continue
        return PromptPair(', '.join(pos_parts), ', '.join(neg_parts))




class ElementBuilder:
    @staticmethod
    def from_element(elem, state=None, lists=None):
        data = dict(getattr(elem, 'data', {}) or {})
        name = str(getattr(elem, 'name', ''))
        activated = data.pop('activated', False)
        if state is not None and hasattr(state, 'activated') and name:
            # Panel state is authoritative: a missing key means deactivated
            # (e.g. after Clear All empties the map), not "use layout default".
            get = getattr(state.activated, 'get', None)
            if callable(get):
                activated = bool(get(name, False))
            elif name in state.activated:
                activated = bool(state.activated[name])
            else:
                activated = False
        etype = None
        if hasattr(elem, 'type') and elem.type:
            etype = elem.type
        if etype is None:
            etype = data.get('type')
        if etype is not None:
            data['type'] = etype
        # apply state values
        if state is not None and hasattr(state, 'values') and name and name in state.values:
            v = state.values[name]
            # for edit/edit_link, set 'edit' if numeric
            if etype in ('edit', 'edit_link'):
                try:
                    data['edit'] = float(v)
                except Exception:
                    data['edit'] = v
            else:
                data['value'] = v
        # For dual, read pos/neg prompts and emphases from state
        if etype == 'dual' and state is not None and hasattr(state, 'values'):
            for k in ('prompt_pos', 'prompt_neg', 'emphasis_pos', 'emphasis_neg'):
                state_key = name + '_' + k
                if state_key in state.values:
                    data[k] = state.values[state_key]
        # For single, read emphasis, is_negative, and prompt from state
        if etype == 'single' and state is not None and hasattr(state, 'values'):
            for k in ('emphasis', 'is_negative', 'prompt'):
                state_key = name + '_' + k
                if state_key in state.values:
                    data[k] = state.values[state_key]
        # For edit, read the negative flag from state (e.g. set by presets;
        # the panel row itself has no N toggle). Slider value handled above.
        if etype == 'edit' and state is not None and hasattr(state, 'values'):
            state_key = name + '_is_negative'
            if state_key in state.values:
                data['is_negative'] = state.values[state_key]
        # For edit_link, read linked element's edit value from state OR from element's own data
        if etype == 'edit_link':
            link_name = data.get('link')
            # First check state (for runtime changes)
            if link_name and state is not None and hasattr(state, 'values') and link_name in state.values:
                try:
                    data['edit'] = float(state.values[link_name])
                except Exception:
                    pass
            # Fallback: if still no edit value, check if element has its own edit in data
            elif 'edit' not in data and link_name:
                # Could look up the linked element's data here if needed
                pass
            # Negative follows the linked source (read-through, like the value):
            # linked entries have no independent negative flag.
            if link_name and state is not None and hasattr(state, 'values'):
                src_neg_key = link_name + '_is_negative'
                if src_neg_key in state.values:
                    data['is_negative'] = state.values[src_neg_key]
        return PromptElement(name=name, activated=activated, **data)


class ListExpander:
    @staticmethod
    def expand_from_list(elem, lists, current_select_name=None):
        # returns list of PromptElement-like data
        res = []
        data = dict(getattr(elem, 'data', {}) or {})
        etype = getattr(elem, 'type', data.get('type'))
        if etype != 'from_list':
            return res
        list_name = getattr(elem, 'name', None) or data.get('name') or data.get('i')
        postfix = data.get('postfix') or ''
        target_list = None
        if list_name:
            if list_name in lists:
                target_list = lists[list_name]
            else:
                # try case-insensitive
                for k,v in lists.items():
                    if k.lower() == str(list_name).lower():
                        target_list = v
                        break
        if target_list is None:
            return res
        # Build safe postfix for naming
        safe_postfix = '_' + postfix.replace(' ', '_') if postfix else ''
        # Use current_select_name if available, fallback to list_name
        select_name = current_select_name or list_name
        for item in target_list:
            # create single-like element with correct naming pattern
            # Prompt should be lowercase, display name keeps original case
            d = {
                'type': 'single',
                'prompt': item.lower(),  # lowercase for prompt
                'postfix': postfix or ''
            }
            item_name = f"{select_name}_{list_name}{safe_postfix}_{item}"
            res.append(type('E', (), {'name': item_name, 'data': d, 'type': 'single'})())
        return res
