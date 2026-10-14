from __future__ import annotations

from .agent_analyzed import AgentAnalyzedEvidence
from .instruction_only import InstructionOnlyEvidence
from .per_trace_then_aggregate import PerTraceThenAggregateEvidence
from .sanitized_instruction import SanitizedInstructionEvidence


EVIDENCE_REGISTRY = {
    AgentAnalyzedEvidence.name: AgentAnalyzedEvidence,
    PerTraceThenAggregateEvidence.name: PerTraceThenAggregateEvidence,
    SanitizedInstructionEvidence.name: SanitizedInstructionEvidence,
    InstructionOnlyEvidence.name: InstructionOnlyEvidence,
}
