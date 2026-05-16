from __future__ import annotations
import json
from dataclasses import asdict


def dump_node_state(node_state) -> str:
    return json.dumps(asdict(node_state), ensure_ascii=False)
