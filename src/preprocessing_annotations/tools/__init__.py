"""AV-5+: standalone tools -- callable directly, and the future home for
any tool a recovery/agent stage would invoke (page classification,
region re-inspection, targeted re-OCR, ...). No tool in this package
depends on an agent/LLM being present; each must be independently
useful and independently testable.
"""

from .vocab_tool import VocabularyGapCandidate, propose_vocabulary_gaps
from .wallband_tool import WallBandTool

__all__ = [
    "VocabularyGapCandidate",
    "WallBandTool",
    "propose_vocabulary_gaps",
]
