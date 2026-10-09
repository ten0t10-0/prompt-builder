from __future__ import annotations
import json
import os
from typing import Dict, Any

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from server import PromptServer
except Exception:
    PromptServer = None

try:
    from .prompt_builder_core import parser, evaluator, state as state_mod
except ImportError:
    from prompt_builder_core import parser, evaluator, state as state_mod

WEB_DIRECTORY = './web'
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Global UI state shared between frontend and node
_shared_ui_state = {'values': {}, 'activated': {}, 'expanded': {}, 'active_tab': 0}


class PromptBuilderNode:
    CATEGORY = 'B Prompt Builder'
    OUTPUT_NODE = False
    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('positive', 'negative')
    FUNCTION = 'execute'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {},
            'hidden': {
                'state': ('STRING', {'default': '{}'}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, state: str = '{}'):
        # Always re-run when shared UI state changes
        import hashlib
        state_hash = hashlib.md5(str(_shared_ui_state).encode()).hexdigest()[:8]
        return state_hash

    def __init__(self):
        self._state = state_mod.UiState()
        self._layout_path = os.path.join(os.path.dirname(__file__), 'layout.yaml')

    def _get_shared_state(self):
        return state_mod.UiState.from_dict(_shared_ui_state)

    def _collect_active_prompts(self, st):
        try:
            ly = parser.parse_yaml(self._layout_path)
        except Exception:
            return []
        active = []
        lists = getattr(ly, 'lists', {}) or {}
        
        # Build a map of element name -> element data for edit_link resolution
        elem_data_map = {}
        def index_elements(elem):
            if hasattr(elem, 'name') and elem.name:
                elem_data_map[elem.name] = elem.data if hasattr(elem, 'data') else {}
            if hasattr(elem, 'children'):
                for child in elem.children:
                    index_elements(child)
        for tab in ly.tabs:
            index_elements(tab)
        
        def traverse(elem, inherited_prefix='', inherited_postfix='', current_select_name=None):
            etype = getattr(elem, 'type', None) or (elem.data.get('type') if hasattr(elem, 'data') and elem.data else None)
            if not etype:
                return
            
            # Update current_select_name when we enter a select
            if etype == 'select' and elem.name:
                current_select_name = elem.name
            
            # Merge inherited prefix/postfix with element's own
            elem_prefix = (elem.data.get('prefix', '') if hasattr(elem, 'data') and elem.data else '') or inherited_prefix
            elem_postfix = (elem.data.get('postfix', '') if hasattr(elem, 'data') and elem.data else '') or inherited_postfix
            
            if etype == 'from_list':
                for exp in evaluator.ListExpander.expand_from_list(elem, lists, current_select_name):
                    # Apply inherited prefix/postfix to expanded items
                    if hasattr(exp, 'data') and exp.data:
                        if elem_prefix and not exp.data.get('prefix'):
                            exp.data['prefix'] = elem_prefix
                        if elem_postfix and not exp.data.get('postfix'):
                            exp.data['postfix'] = elem_postfix
                    pe = evaluator.ElementBuilder.from_element(exp, st, lists=lists)
                    if pe.activated:
                        active.append(pe)
                return
            
            if etype in ('single', 'dual', 'edit', 'edit_link'):
                # Apply inherited prefix/postfix
                if hasattr(elem, 'data') and elem.data:
                    if elem_prefix and not elem.data.get('prefix'):
                        elem.data['prefix'] = elem_prefix
                    if elem_postfix and not elem.data.get('postfix'):
                        elem.data['postfix'] = elem_postfix
                # For edit_link without own edit value, inherit from linked element's data
                if etype == 'edit_link' and hasattr(elem, 'data') and elem.data:
                    link_name = elem.data.get('link')
                    if link_name and link_name in elem_data_map:
                        linked_data = elem_data_map[link_name]
                        if linked_data and 'edit' in linked_data and 'edit' not in elem.data:
                            elem.data['edit'] = linked_data['edit']
                pe = evaluator.ElementBuilder.from_element(elem, st, lists=lists)
                if pe.activated:
                    active.append(pe)
                return
            
            # Container types - recurse into children with inherited prefix/postfix
            if hasattr(elem, 'children'):
                for child in elem.children:
                    traverse(child, elem_prefix, elem_postfix, current_select_name)
        
        for tab in ly.tabs:
            traverse(tab)
        
        return active

    def execute(self, state: str = '{}') -> tuple[str, str]:
        # Always use shared UI state for workflow runs
        # The hidden 'state' input is just for the /eval endpoint
        st = self._get_shared_state()
        active = self._collect_active_prompts(st)
        ev = evaluator.PromptEvaluator(st, active)
        pair = ev.build_prompts()
        return (pair.pos, pair.neg)



try:
    if PromptServer is not None:
        @PromptServer.instance.routes.get('/extensions/prompt-builder/layout.yaml')
        async def get_layout(request):
            import aiofiles
            layout_path = os.path.join(os.path.dirname(__file__), 'layout.yaml')
            async with aiofiles.open(layout_path, 'rb') as f:
                data = await f.read()
            from aiohttp import web
            return web.Response(body=data, content_type='text/yaml')

        @PromptServer.instance.routes.get('/extensions/prompt-builder/state/{node_id}')
        async def get_state(request):
            from aiohttp import web
            return web.json_response({})

        @PromptServer.instance.routes.get('/extensions/prompt-builder/layout.json')
        async def get_layout_json(request):
            from aiohttp import web
            from .prompt_builder_core import parser, layout as layout_mod
            try:
                ly = parser.parse_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'layout.yaml'))
                
                def elem_to_dict(elem):
                    result = {'type': elem.type, 'name': elem.name}
                    # Flatten data properties to top level
                    if elem.data:
                        result.update(elem.data)
                    if elem.children:
                        result['children'] = [elem_to_dict(c) for c in elem.children]
                    return result
                
                data = {'tabs': [], 'lists': getattr(ly, 'lists', {}) or {}}
                for tab in (getattr(ly, 'tabs', []) or []):
                    tab_dict = {'name': getattr(tab, 'name', '')}
                    # Flatten tab-level props (e.g. no_reset) like elements do
                    if getattr(tab, 'data', None):
                        tab_dict.update(tab.data)
                    if tab.children:
                        tab_dict['children'] = [elem_to_dict(c) for c in tab.children]
                    data['tabs'].append(tab_dict)
            except Exception as e:
                data = {'error': str(e)}
            return web.json_response(data)

        @PromptServer.instance.routes.post('/extensions/prompt-builder/eval')
        async def pb_eval(request):
            from aiohttp import web
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            state = payload.get('state', '{}')
            try:
                n = PromptBuilderNode()
                res = n.execute(state=state)
                return web.json_response({'positive': res[0], 'negative': res[1]})
            except Exception as e:
                return web.json_response({'error': str(e)}, status=500)

        @PromptServer.instance.routes.post('/extensions/prompt-builder/state')
        async def pb_set_state(request):
            from aiohttp import web
            global _shared_ui_state
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            _shared_ui_state = payload.get('state', _shared_ui_state)
            return web.json_response({'ok': True})

        @PromptServer.instance.routes.get('/extensions/prompt-builder/state')
        async def pb_get_state(request):
            from aiohttp import web
            return web.json_response(_shared_ui_state)
except Exception:
    pass

NODE_CLASS_MAPPINGS['PromptBuilderNode'] = PromptBuilderNode
NODE_DISPLAY_NAME_MAPPINGS['PromptBuilderNode'] = 'B Prompt Builder'


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

