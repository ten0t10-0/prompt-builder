from __future__ import annotations
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class UiState:
    values: Dict[str, Any] = field(default_factory=dict)
    activated: Dict[str, bool] = field(default_factory=dict)
    expanded: Dict[str, bool] = field(default_factory=dict)
    active_tab: Optional[str] = None
    schema_version: str = '1.0.0'
    dirty: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'values': self.values,
            'activated': self.activated,
            'expanded': self.expanded,
            'active_tab': self.active_tab,
            'schema_version': self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'UiState':
        return cls(
            values=d.get('values', {}) or {},
            activated=d.get('activated', {}) or {},
            expanded=d.get('expanded', {}) or {},
            active_tab=d.get('active_tab'),
            schema_version=d.get('schema_version', '1.0.0'),
        )

    def mark_dirty(self, name: str):
        self.dirty.add(name)

    def clear_dirty(self):
        self.dirty.clear()
