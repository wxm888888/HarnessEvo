from __future__ import annotations

from .accepted_lineage import AcceptedLineageMemory
from .none import NoMemory

MEMORY_REGISTRY = {
    NoMemory.name: NoMemory,
    AcceptedLineageMemory.name: AcceptedLineageMemory,
}
