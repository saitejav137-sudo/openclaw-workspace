"""
Adaptive Learning Engine for OpenClaw

Learns from agent actions to improve future performance:
- Pattern recognition from successful action sequences
- Strategy scoring and selection
- Feedback-driven improvement
- Experience replay for skill reinforcement
"""

import time
import json
import hashlib
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import os

from .logger import get_logger

logger = get_logger("adaptive_learning")


@dataclass
class Experience:
    """A recorded experience from agent execution."""
    id: str
    context: str
    action: str
    result: str
    reward: float  # -1.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class Strategy:
    """A learned strategy for handling specific situations."""
    id: str
    name: str
    pattern: str  # Context pattern this strategy applies to
    actions: List[str]  # Ordered list of action types
    score: float = 0.5  # Effectiveness score (0-1)
    uses: int = 0
    successes: int = 0
    failures: int = 0
    last_used: Optional[float] = None
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses > 0 else 0.0


class ExperienceReplay:
    """
    Experience replay buffer for learning from past actions.
    Prioritizes experiences with high reward variance (surprising outcomes).
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: List[Experience] = []
        self._lock = threading.Lock()

    def add(self, experience: Experience):
        """Add experience to buffer."""
        with self._lock:
            self._buffer.append(experience)
            if len(self._buffer) > self.max_size:
                # Remove lowest-reward experiences
                self._buffer.sort(key=lambda e: abs(e.reward), reverse=True)
                self._buffer = self._buffer[:self.max_size]

    def sample(self, n: int = 10) -> List[Experience]:
        """Sample experiences, prioritizing high-information ones."""
        import random
        with self._lock:
            if len(self._buffer) <= n:
                return list(self._buffer)

            # Weight by absolute reward (high reward or high penalty = informative)
            weights = [abs(e.reward) + 0.1 for e in self._buffer]
            total = sum(weights)
            probs = [w / total for w in weights]

            indices = random.choices(range(len(self._buffer)), weights=probs, k=n)
            return [self._buffer[i] for i in set(indices)]

    def get_by_context(self, context_pattern: str, limit: int = 10) -> List[Experience]:
        """Get experiences matching a context pattern."""
        with self._lock:
            matching = [
                e for e in self._buffer
                if context_pattern.lower() in e.context.lower()
            ]
            matching.sort(key=lambda e: e.reward, reverse=True)
            return matching[:limit]

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)


class StrategyManager:
    """
    Manages and selects strategies based on learned performance.
    Uses Thompson Sampling for exploration vs exploitation.
    """

    def __init__(self):
        self._strategies: Dict[str, Strategy] = {}
        self._lock = threading.Lock()

    def add_strategy(self, strategy: Strategy):
        """Register a new strategy."""
        with self._lock:
            self._strategies[strategy.id] = strategy
            logger.info(f"Added strategy: {strategy.name} (score={strategy.score:.2f})")

    def select_strategy(self, context: str) -> Optional[Strategy]:
        """
        Select the best strategy for a given context.
        Uses a simple scoring mechanism with exploration bonus.
        """
        import random
        with self._lock:
            candidates = []
            for strategy in self._strategies.values():
                # Check if pattern matches context
                if strategy.pattern.lower() in context.lower():
                    # Score = success_rate * exploitation + exploration_bonus
                    exploitation = strategy.success_rate
                    exploration = 1.0 / (strategy.uses + 1)  # Less-used = more exploration
                    score = 0.7 * exploitation + 0.3 * exploration
                    candidates.append((strategy, score))

            if not candidates:
                return None

            # Sort by score and add some randomness for exploration
            candidates.sort(key=lambda x: x[1], reverse=True)

            # 80% chance pick top, 20% chance pick random
            if random.random() < 0.8 or len(candidates) == 1:
                return candidates[0][0]
            else:
                return random.choice(candidates)[0]

    def update_strategy(self, strategy_id: str, success: bool, reward: float = 0.0):
        """Update strategy performance metrics."""
        with self._lock:
            strategy = self._strategies.get(strategy_id)
            if not strategy:
                return

            strategy.uses += 1
            strategy.last_used = time.time()

            if success:
                strategy.successes += 1
            else:
                strategy.failures += 1

            # Update score using exponential moving average
            alpha = 0.3  # Learning rate
            new_score = 1.0 if success else 0.0
            strategy.score = (1 - alpha) * strategy.score + alpha * new_score

            logger.debug(
                f"Strategy '{strategy.name}' updated: "
                f"score={strategy.score:.2f} success_rate={strategy.success_rate:.2f}"
            )

    def get_top_strategies(self, n: int = 5) -> List[Strategy]:
        """Get top performing strategies."""
        with self._lock:
            sorted_strategies = sorted(
                self._strategies.values(),
                key=lambda s: s.score,
                reverse=True
            )
            return sorted_strategies[:n]

    def remove_ineffective(self, min_uses: int = 10, max_score: float = 0.2):
        """Remove strategies that have proven ineffective."""
        with self._lock:
            to_remove = [
                sid for sid, s in self._strategies.items()
                if s.uses >= min_uses and s.score < max_score
            ]
            for sid in to_remove:
                name = self._strategies[sid].name
                del self._strategies[sid]
                logger.info(f"Removed ineffective strategy: {name}")


class AdaptiveLearner:
    """
    Main adaptive learning engine.

    Combines experience replay with strategy selection to
    continuously improve agent performance.

    Usage:
        learner = AdaptiveLearner()

        # Record actions
        learner.record_experience(
            context="User asked to search for files",
            action="file_search",
            result="Found 5 files",
            reward=0.8
        )

        # Get recommendations
        strategy = learner.recommend_strategy("search for files")
    """

    def __init__(self, storage_dir: str = "~/.openclaw/learning"):
        self.storage_dir = os.path.expanduser(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

        self.replay = ExperienceReplay(max_size=5000)
        self.strategies = StrategyManager()
        self._action_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()

        # Load existing knowledge
        self._load_knowledge()

    def record_experience(
        self,
        context: str,
        action: str,
        result: str,
        reward: float,
        tags: List[str] = None,
        metadata: Dict = None
    ) -> Experience:
        """Record an agent experience for learning."""
        exp_id = hashlib.sha256(
            f"{context}{action}{time.time()}".encode()
        ).hexdigest()[:12]

        experience = Experience(
            id=exp_id,
            context=context,
            action=action,
            result=result,
            reward=max(-1.0, min(1.0, reward)),  # Clamp to [-1, 1]
            tags=tags or [],
            metadata=metadata or {}
        )

        self.replay.add(experience)

        # Track action patterns
        with self._lock:
            self._action_patterns[action].append({
                "context": context,
                "reward": reward,
                "timestamp": time.time()
            })

        # Auto-discover strategies from patterns
        self._discover_patterns(context, action, reward)

        logger.debug(f"Recorded experience: {action} reward={reward:.2f}")
        return experience

    def recommend_strategy(self, context: str) -> Optional[Strategy]:
        """Get the best strategy recommendation for a context."""
        # Check existing strategies
        strategy = self.strategies.select_strategy(context)
        if strategy:
            return strategy

        # Fall back to experience-based recommendation
        similar = self.replay.get_by_context(context, limit=5)
        if similar:
            best = max(similar, key=lambda e: e.reward)
            return Strategy(
                id="dynamic",
                name=f"Best practice: {best.action}",
                pattern=context,
                actions=[best.action],
                score=best.reward
            )

        return None

    def report_outcome(self, strategy_id: str, success: bool, reward: float = 0.0):
        """Report the outcome of following a strategy."""
        self.strategies.update_strategy(strategy_id, success, reward)

    def _discover_patterns(self, context: str, action: str, reward: float):
        """Auto-discover successful patterns from experiences."""
        with self._lock:
            patterns = self._action_patterns[action]

            # Need at least 5 uses to evaluate
            if len(patterns) < 5:
                return

            # Calculate average reward for this action
            recent = patterns[-10:]
            avg_reward = sum(p["reward"] for p in recent) / len(recent)

            # If consistently good, create a strategy
            if avg_reward > 0.7:
                strategy_id = hashlib.sha256(
                    f"{action}{context[:50]}".encode()
                ).hexdigest()[:10]

                if strategy_id not in [s.id for s in self.strategies.get_top_strategies(100)]:
                    self.strategies.add_strategy(Strategy(
                        id=strategy_id,
                        name=f"Auto: {action} for similar contexts",
                        pattern=context[:50],
                        actions=[action],
                        score=avg_reward
                    ))

    def get_insights(self) -> Dict[str, Any]:
        """Get learning insights and statistics."""
        top_strategies = self.strategies.get_top_strategies(10)

        with self._lock:
            action_stats = {
                action: {
                    "total_uses": len(patterns),
                    "avg_reward": (
                        sum(p["reward"] for p in patterns) / len(patterns)
                        if patterns else 0
                    )
                }
                for action, patterns in self._action_patterns.items()
            }

        return {
            "total_experiences": self.replay.size(),
            "total_strategies": len(top_strategies),
            "top_strategies": [
                {"name": s.name, "score": round(s.score, 3), "uses": s.uses}
                for s in top_strategies
            ],
            "action_stats": action_stats
        }

    def _save_knowledge(self):
        """Persist learned knowledge to disk."""
        try:
            filepath = os.path.join(self.storage_dir, "knowledge.json")
            data = {
                "strategies": [
                    {
                        "id": s.id, "name": s.name, "pattern": s.pattern,
                        "actions": s.actions, "score": s.score, "uses": s.uses,
                        "successes": s.successes, "failures": s.failures
                    }
                    for s in self.strategies.get_top_strategies(100)
                ],
                "saved_at": time.time()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")

    def _load_knowledge(self):
        """Load previously learned knowledge."""
        try:
            filepath = os.path.join(self.storage_dir, "knowledge.json")
            if not os.path.exists(filepath):
                return

            with open(filepath, 'r') as f:
                data = json.load(f)

            for s_data in data.get("strategies", []):
                strategy = Strategy(
                    id=s_data["id"],
                    name=s_data["name"],
                    pattern=s_data["pattern"],
                    actions=s_data["actions"],
                    score=s_data.get("score", 0.5),
                    uses=s_data.get("uses", 0),
                    successes=s_data.get("successes", 0),
                    failures=s_data.get("failures", 0)
                )
                self.strategies.add_strategy(strategy)

            logger.info(f"Loaded {len(data.get('strategies', []))} strategies")

        except Exception as e:
            logger.error(f"Failed to load knowledge: {e}")


# ============== Global Instance ==============

_learner: Optional[AdaptiveLearner] = None


def get_learner() -> AdaptiveLearner:
    """Get global adaptive learner."""
    global _learner
    if _learner is None:
        _learner = AdaptiveLearner()
    return _learner


__all__ = [
    "Experience",
    "Strategy",
    "ExperienceReplay",
    "StrategyManager",
    "AdaptiveLearner",
    "get_learner",
]
