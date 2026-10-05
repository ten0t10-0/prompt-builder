from __future__ import annotations
from typing import Dict, List, Any, Optional
from .schema import Layout, LayoutElement


class CompiledLayout:
    def __init__(self, layout: Layout):
        self.layout = layout
        self.name_to_element: Dict[str, Any] = {}
        self._index(layout)

    def _index(self, layout: Layout):
        for tab in layout.tabs:
            self._index_element(tab)

    def _index_element(self, elem: LayoutElement):
        if elem.name:
            self.name_to_element[elem.name] = elem
        for child in elem.children:
            self._index_element(child)

    def get_element(self, name: str) -> Optional[LayoutElement]:
        return self.name_to_element.get(name)
