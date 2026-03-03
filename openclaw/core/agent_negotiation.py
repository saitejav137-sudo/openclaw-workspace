"""
Agent Negotiation Protocol for OpenClaw

Multi-agent consensus mechanisms:
- Proposals: agents submit approaches with confidence scores
- Voting: majority vote, weighted vote, leader-decides
- Consensus: automatic decision based on configured protocol
- Event bus integration for transparency
"""

import time
import threading
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from .logger import get_logger

logger = get_logger("negotiation")


# ============== Enums ==============

class ConsensusMethod(str, Enum):
    MAJORITY = "majority"          # Simple majority wins
    WEIGHTED = "weighted"          # Weighted by agent confidence/success rate
    LEADER_DECIDES = "leader"      # Designated leader picks
    UNANIMOUS = "unanimous"        # All must agree
    BEST_SCORE = "best_score"      # Highest confidence wins


class NegotiationStatus(str, Enum):
    OPEN = "open"
    COLLECTING = "collecting"
    DECIDED = "decided"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ============== Data Classes ==============

@dataclass
class Proposal:
    """An agent's proposed approach."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    agent_id: str = ""
    agent_name: str = ""
    approach: str = ""                 # Text description of proposed approach
    confidence: float = 0.5           # 0-1, how confident the agent is
    reasoning: str = ""               # Why this approach
    estimated_cost: float = 0.0       # Estimated tokens/time
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    votes: int = 0                    # Votes received during voting phase


@dataclass
class NegotiationRound:
    """A single negotiation round."""
    id: str = field(default_factory=lambda: str(uuid4())[:10])
    question: str = ""                 # What are we deciding?
    context: Dict[str, Any] = field(default_factory=dict)
    proposals: List[Proposal] = field(default_factory=list)
    method: ConsensusMethod = ConsensusMethod.BEST_SCORE
    status: NegotiationStatus = NegotiationStatus.OPEN
    winner: Optional[Proposal] = None
    started_at: float = field(default_factory=time.time)
    decided_at: float = 0.0
    timeout: float = 60.0             # Seconds to wait for proposals

    @property
    def duration(self) -> float:
        if self.decided_at:
            return self.decided_at - self.started_at
        return time.time() - self.started_at


# ============== Scoring Functions ==============

def score_majority(proposals: List[Proposal]) -> Optional[Proposal]:
    """Simple: most votes wins. Falls back to highest confidence."""
    if not proposals:
        return None
    # If votes exist, use them
    max_votes = max(p.votes for p in proposals)
    if max_votes > 0:
        candidates = [p for p in proposals if p.votes == max_votes]
        return max(candidates, key=lambda p: p.confidence)
    # Fallback to confidence
    return max(proposals, key=lambda p: p.confidence)


def score_weighted(proposals: List[Proposal], agent_weights: Dict[str, float] = None) -> Optional[Proposal]:
    """Weighted by agent confidence × agent weight."""
    if not proposals:
        return None
    weights = agent_weights or {}

    def weighted_score(p: Proposal) -> float:
        agent_weight = weights.get(p.agent_id, 1.0)
        return p.confidence * agent_weight

    return max(proposals, key=weighted_score)


def score_best(proposals: List[Proposal]) -> Optional[Proposal]:
    """Highest confidence wins."""
    return max(proposals, key=lambda p: p.confidence) if proposals else None


def score_unanimous(proposals: List[Proposal]) -> Optional[Proposal]:
    """All proposals must have same approach (by agent_name grouping). Returns None if disagreement."""
    if not proposals:
        return None
    # Group by approach text (simplified: first 50 chars)
    approaches = set(p.approach[:50].strip().lower() for p in proposals)
    if len(approaches) == 1:
        return max(proposals, key=lambda p: p.confidence)
    return None  # No consensus


SCORING_FUNCTIONS = {
    ConsensusMethod.MAJORITY: score_majority,
    ConsensusMethod.WEIGHTED: score_weighted,
    ConsensusMethod.BEST_SCORE: score_best,
    ConsensusMethod.UNANIMOUS: score_unanimous,
}


# ============== Negotiation Engine ==============

class NegotiationEngine:
    """
    Runs agent negotiation sessions.

    Flow:
    1. Open a negotiation round with a question
    2. Agents submit proposals
    3. (Optional) Agents vote on proposals
    4. Engine decides based on consensus method
    5. Winner is announced via event bus

    Usage:
        engine = NegotiationEngine()

        round_id = engine.open_round(
            question="How should we handle this user's request?",
            method=ConsensusMethod.BEST_SCORE,
        )

        engine.submit_proposal(round_id, Proposal(
            agent_id="researcher",
            approach="Search the web for relevant info",
            confidence=0.8,
        ))
        engine.submit_proposal(round_id, Proposal(
            agent_id="coder",
            approach="Write code to solve it directly",
            confidence=0.6,
        ))

        winner = engine.decide(round_id)
        # winner.approach == "Search the web for relevant info"
    """

    def __init__(self, agent_weights: Dict[str, float] = None):
        self._rounds: Dict[str, NegotiationRound] = {}
        self._agent_weights = agent_weights or {}
        self._lock = threading.Lock()
        self._total_rounds = 0
        self._total_proposals = 0

    def open_round(
        self,
        question: str,
        method: ConsensusMethod = ConsensusMethod.BEST_SCORE,
        context: Dict[str, Any] = None,
        timeout: float = 60.0,
    ) -> str:
        """Open a new negotiation round. Returns round ID."""
        round_ = NegotiationRound(
            question=question,
            method=method,
            context=context or {},
            status=NegotiationStatus.COLLECTING,
            timeout=timeout,
        )
        with self._lock:
            self._rounds[round_.id] = round_
            self._total_rounds += 1

        self._emit("negotiation.started", f"Negotiation opened: {question[:60]}",
                    {"round_id": round_.id, "method": method.value})

        logger.info("Negotiation '%s' opened: %s (method=%s)", round_.id, question[:60], method.value)
        return round_.id

    def submit_proposal(self, round_id: str, proposal: Proposal) -> bool:
        """Submit a proposal to a round."""
        with self._lock:
            round_ = self._rounds.get(round_id)
            if not round_ or round_.status != NegotiationStatus.COLLECTING:
                return False
            round_.proposals.append(proposal)
            self._total_proposals += 1

        self._emit("negotiation.proposal", f"Proposal from {proposal.agent_name or proposal.agent_id}",
                    {"round_id": round_id, "agent_id": proposal.agent_id,
                     "confidence": proposal.confidence})

        logger.info("  Proposal from '%s': %.0f%% confidence",
                    proposal.agent_name or proposal.agent_id, proposal.confidence * 100)
        return True

    def vote(self, round_id: str, voter_agent_id: str, proposal_id: str) -> bool:
        """Cast a vote for a proposal."""
        with self._lock:
            round_ = self._rounds.get(round_id)
            if not round_ or round_.status != NegotiationStatus.COLLECTING:
                return False
            for p in round_.proposals:
                if p.id == proposal_id:
                    p.votes += 1
                    return True
        return False

    def decide(self, round_id: str) -> Optional[Proposal]:
        """Close the round and decide the winner."""
        with self._lock:
            round_ = self._rounds.get(round_id)
            if not round_:
                return None

            if not round_.proposals:
                round_.status = NegotiationStatus.FAILED
                return None

            # Apply scoring
            scoring_fn = SCORING_FUNCTIONS.get(round_.method, score_best)

            if round_.method == ConsensusMethod.WEIGHTED:
                winner = scoring_fn(round_.proposals, self._agent_weights)
            else:
                winner = scoring_fn(round_.proposals)

            if winner:
                round_.winner = winner
                round_.status = NegotiationStatus.DECIDED
                round_.decided_at = time.time()

                self._emit("negotiation.decided",
                           f"Consensus reached: {winner.agent_name or winner.agent_id}'s approach "
                           f"({winner.confidence:.0%} confidence)",
                           {"round_id": round_id, "winner_agent": winner.agent_id,
                            "approach": winner.approach[:100]})

                logger.info("Negotiation '%s' decided: winner=%s (%.0f%%)",
                           round_id, winner.agent_name or winner.agent_id,
                           winner.confidence * 100)
            else:
                round_.status = NegotiationStatus.FAILED
                logger.warning("Negotiation '%s' failed to reach consensus", round_id)

            return winner

    def get_round(self, round_id: str) -> Optional[NegotiationRound]:
        return self._rounds.get(round_id)

    def get_stats(self) -> Dict[str, Any]:
        active = sum(1 for r in self._rounds.values() if r.status == NegotiationStatus.COLLECTING)
        decided = sum(1 for r in self._rounds.values() if r.status == NegotiationStatus.DECIDED)
        return {
            "total_rounds": self._total_rounds,
            "total_proposals": self._total_proposals,
            "active_rounds": active,
            "decided_rounds": decided,
        }

    def _emit(self, event_type_str: str, message: str, data: dict = None):
        try:
            from .event_bus import get_event_bus, EventType
            bus = get_event_bus()
            bus.emit(EventType.SYSTEM_STARTUP, message, data=data, source="negotiation")
        except Exception:
            pass


# ============== Global Instance ==============

_engine: Optional[NegotiationEngine] = None


def get_negotiation_engine(**kwargs) -> NegotiationEngine:
    global _engine
    if _engine is None:
        _engine = NegotiationEngine(**kwargs)
    return _engine


__all__ = [
    "ConsensusMethod",
    "NegotiationStatus",
    "Proposal",
    "NegotiationRound",
    "NegotiationEngine",
    "get_negotiation_engine",
]
