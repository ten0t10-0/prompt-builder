from __future__ import annotations
from typing import Dict, Set, List


class DependencyGraph:
    def __init__(self):
        self.deps: Dict[str, Set[str]] = {}  # source -> targets
        self.rev_deps: Dict[str, Set[str]] = {}  # target -> sources

    def add_edge(self, source: str, target: str):
        if not source or not target:
            return
        self.deps.setdefault(source, set()).add(target)
        self.rev_deps.setdefault(target, set()).add(source)

    def get_dependents(self, node: str) -> Set[str]:
        return self.rev_deps.get(node, set())

    def get_dependencies(self, node: str) -> Set[str]:
        return self.deps.get(node, set())

    def clear(self):
        self.deps.clear()
        self.rev_deps.clear()
