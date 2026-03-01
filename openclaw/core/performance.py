"""
OpenClaw Performance Module — Phase 4C

Async workflow execution, connection pooling, and optimised vector search.
"""

import asyncio
import time
import heapq
import threading
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from .logger import get_logger

logger = get_logger("performance")


# ============== Async Workflow Runner ==============

class AsyncWorkflowRunner:
    """
    Run workflow nodes with true async parallelism.

    Wraps the existing synchronous WorkflowEngine but executes
    parallel branches concurrently using asyncio.gather().
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._results: Dict[str, Any] = {}

    async def run_node(self, func: Callable, node_id: str, context: Dict) -> Any:
        """Run a single node, offloading blocking work to the thread pool."""
        loop = asyncio.get_event_loop()
        start = time.time()
        try:
            result = await loop.run_in_executor(self._executor, func, context)
            duration = time.time() - start
            logger.debug(f"Node {node_id} completed in {duration:.2f}s")
            self._results[node_id] = result
            return result
        except Exception as e:
            logger.error(f"Node {node_id} failed: {e}")
            self._results[node_id] = {"error": str(e)}
            raise

    async def run_parallel(self, nodes: List[Tuple[str, Callable, Dict]]) -> Dict[str, Any]:
        """Run multiple nodes concurrently and collect results."""
        tasks = [
            self.run_node(func, node_id, ctx)
            for node_id, func, ctx in nodes
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        return {nid: self._results.get(nid) for nid, _, _ in nodes}

    async def run_sequential(self, nodes: List[Tuple[str, Callable, Dict]]) -> Dict[str, Any]:
        """Run nodes one at a time, passing previous result to next."""
        results = {}
        prev_result = None
        for node_id, func, ctx in nodes:
            if prev_result is not None:
                ctx["previous_result"] = prev_result
            result = await self.run_node(func, node_id, ctx)
            results[node_id] = result
            prev_result = result
        return results

    async def run_pipeline(self, stages: List[List[Tuple[str, Callable, Dict]]]) -> Dict:
        """
        Run a multi-stage pipeline where each stage is a group of
        parallel nodes, and stages run sequentially.
        """
        all_results = {}
        for i, stage in enumerate(stages):
            logger.info(f"Pipeline stage {i+1}/{len(stages)} — {len(stage)} nodes")
            stage_results = await self.run_parallel(stage)
            all_results.update(stage_results)
        return all_results

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=True)


# ============== Connection Pool ==============

class ConnectionPool:
    """
    Thread-safe connection pool for HTTP sessions.

    Reuses `requests.Session` objects to benefit from TCP keep-alive
    and TLS session resumption.
    """

    def __init__(self, pool_size: int = 10):
        self._pool_size = pool_size
        self._lock = threading.Lock()
        self._sessions: List = []
        self._in_use: int = 0

    def acquire(self):
        """Acquire a session from the pool."""
        import requests

        with self._lock:
            if self._sessions:
                session = self._sessions.pop()
                self._in_use += 1
                return session

            if self._in_use < self._pool_size:
                session = requests.Session()
                # Configure retries and timeouts
                adapter = requests.adapters.HTTPAdapter(
                    max_retries=3,
                    pool_connections=self._pool_size,
                    pool_maxsize=self._pool_size
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                self._in_use += 1
                return session

        # Pool exhausted — wait and retry
        time.sleep(0.1)
        return self.acquire()

    def release(self, session):
        """Return a session to the pool."""
        with self._lock:
            self._in_use -= 1
            self._sessions.append(session)

    def get_stats(self) -> Dict:
        """Get pool statistics."""
        with self._lock:
            return {
                "pool_size": self._pool_size,
                "available": len(self._sessions),
                "in_use": self._in_use,
            }

    def close_all(self):
        """Close all sessions."""
        with self._lock:
            for s in self._sessions:
                try:
                    s.close()
                except Exception:
                    pass
            self._sessions.clear()
            self._in_use = 0

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        pass  # Sessions managed via release()


# ============== Optimised Vector Index ==============

@dataclass
class VectorEntry:
    """Entry in the vector index."""
    doc_id: str
    embedding: List[float]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizedVectorIndex:
    """
    Optimised in-memory vector index using sorted projections
    for faster approximate nearest-neighbor search.

    For small-to-medium datasets (< 100K docs), this provides
    O(n·log(n)) first-time build + O(k·log(n)) per-query performance
    vs the naive O(n) scan.
    """

    def __init__(self, num_projections: int = 8):
        self._entries: Dict[str, VectorEntry] = {}
        self._num_projections = num_projections
        self._projections: List[List[Tuple[float, str]]] = []
        self._dirty = True
        self._lock = threading.Lock()

    def add(self, doc_id: str, text: str, embedding: List[float],
            metadata: Dict = None):
        """Add a document to the index."""
        with self._lock:
            self._entries[doc_id] = VectorEntry(
                doc_id=doc_id, embedding=embedding,
                text=text, metadata=metadata or {}
            )
            self._dirty = True

    def delete(self, doc_id: str):
        """Remove a document."""
        with self._lock:
            self._entries.pop(doc_id, None)
            self._dirty = True

    def _rebuild_projections(self):
        """Build sorted projection lists for fast lookup."""
        if not self._entries:
            self._projections = []
            return

        dim = len(next(iter(self._entries.values())).embedding)
        self._projections = []

        # Project onto random directions (using dim indices as simple projections)
        for p in range(min(self._num_projections, dim)):
            proj = []
            for doc_id, entry in self._entries.items():
                val = entry.embedding[p] if p < len(entry.embedding) else 0.0
                proj.append((val, doc_id))
            proj.sort()
            self._projections.append(proj)

        self._dirty = False

    def search(self, query_embedding: List[float], top_k: int = 5,
               filters: Dict = None) -> List[Dict]:
        """
        Search for nearest neighbours.

        Uses projection-based candidate selection followed by
        exact cosine similarity on the candidates.
        """
        with self._lock:
            if self._dirty:
                self._rebuild_projections()

            if not self._entries:
                return []

            # For small datasets, just do exact search
            n = len(self._entries)
            if n <= 200:
                return self._exact_search(query_embedding, top_k, filters)

            # Candidate selection via projections
            candidate_ids = set()
            window = max(top_k * 4, int(math.sqrt(n)))

            for p, proj in enumerate(self._projections):
                if p >= len(query_embedding):
                    break
                target = query_embedding[p]

                # Binary search for closest projection value
                lo, hi = 0, len(proj) - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if proj[mid][0] < target:
                        lo = mid + 1
                    else:
                        hi = mid

                # Collect nearby candidates
                start = max(0, lo - window // 2)
                end = min(len(proj), lo + window // 2)
                for i in range(start, end):
                    candidate_ids.add(proj[i][1])

            # Exact search on candidates
            results = []
            for doc_id in candidate_ids:
                entry = self._entries.get(doc_id)
                if not entry:
                    continue
                if filters:
                    skip = False
                    for k, v in filters.items():
                        if entry.metadata.get(k) != v:
                            skip = True
                            break
                    if skip:
                        continue
                sim = self._cosine_similarity(query_embedding, entry.embedding)
                results.append((sim, entry))

            results.sort(key=lambda x: -x[0])

            return [
                {
                    "doc_id": e.doc_id,
                    "text": e.text,
                    "similarity": sim,
                    "metadata": e.metadata,
                }
                for sim, e in results[:top_k]
            ]

    def _exact_search(self, query: List[float], top_k: int,
                      filters: Dict = None) -> List[Dict]:
        """Exact brute-force search for small datasets."""
        results = []
        for entry in self._entries.values():
            if filters:
                skip = False
                for k, v in filters.items():
                    if entry.metadata.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            sim = self._cosine_similarity(query, entry.embedding)
            results.append((sim, entry))

        results.sort(key=lambda x: -x[0])
        return [
            {
                "doc_id": e.doc_id,
                "text": e.text,
                "similarity": sim,
                "metadata": e.metadata,
            }
            for sim, e in results[:top_k]
        ]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1e-10
        norm_b = math.sqrt(sum(x * x for x in b)) or 1e-10
        return dot / (norm_a * norm_b)

    def count(self) -> int:
        """Number of documents in the index."""
        return len(self._entries)

    def clear(self):
        """Clear the entire index."""
        with self._lock:
            self._entries.clear()
            self._projections.clear()
            self._dirty = True


# ============== Request Batcher ==============

class RequestBatcher:
    """
    Batch multiple rapid-fire requests into a single processing cycle.

    Useful for LLM calls, embedding generation, and API requests
    where batching dramatically reduces latency.
    """

    def __init__(self, max_batch_size: int = 16,
                 max_wait_ms: float = 50.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: List[Tuple[Any, threading.Event, Dict]] = []
        self._lock = threading.Lock()
        self._process_fn: Optional[Callable] = None

    def set_processor(self, fn: Callable):
        """Set the batch processing function. It receives List[input] → List[output]."""
        self._process_fn = fn

    def submit(self, item: Any, timeout: float = 5.0) -> Any:
        """Submit an item for batch processing. Blocks until result is ready."""
        event = threading.Event()
        result_holder: Dict[str, Any] = {}

        with self._lock:
            self._queue.append((item, event, result_holder))
            queue_len = len(self._queue)

        # If batch is full, trigger immediately
        if queue_len >= self.max_batch_size:
            self._flush()

        # Wait for result
        event.wait(timeout=timeout)
        return result_holder.get("result")

    def _flush(self):
        """Process all queued items."""
        with self._lock:
            if not self._queue:
                return
            batch = self._queue[:]
            self._queue.clear()

        if not self._process_fn:
            for _, event, holder in batch:
                holder["result"] = None
                event.set()
            return

        inputs = [item for item, _, _ in batch]
        try:
            results = self._process_fn(inputs)
        except Exception as e:
            results = [{"error": str(e)}] * len(inputs)

        for (_, event, holder), result in zip(batch, results):
            holder["result"] = result
            event.set()

    def get_stats(self) -> Dict:
        """Get batcher statistics."""
        with self._lock:
            return {
                "pending": len(self._queue),
                "max_batch": self.max_batch_size,
            }


# ============== Performance Monitor ==============

class PerformanceMonitor:
    """
    Lightweight real-time performance monitoring.

    Tracks latency percentiles, throughput, and memory usage
    for any instrumented function.
    """

    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        self._metrics: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, name: str, duration_ms: float):
        """Record a timing measurement."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []
                self._counts[name] = 0

            self._metrics[name].append(duration_ms)
            self._counts[name] += 1

            # Keep window bounded
            if len(self._metrics[name]) > self._window_size:
                self._metrics[name] = self._metrics[name][-self._window_size:]

    def get_stats(self, name: str) -> Dict:
        """Get statistics for a named metric."""
        with self._lock:
            values = self._metrics.get(name, [])
            if not values:
                return {"name": name, "count": 0}

            sorted_vals = sorted(values)
            n = len(sorted_vals)

            return {
                "name": name,
                "count": self._counts.get(name, 0),
                "p50": sorted_vals[n // 2],
                "p95": sorted_vals[int(n * 0.95)],
                "p99": sorted_vals[int(n * 0.99)],
                "avg": sum(sorted_vals) / n,
                "min": sorted_vals[0],
                "max": sorted_vals[-1],
            }

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all metrics."""
        with self._lock:
            names = list(self._metrics.keys())
        return {name: self.get_stats(name) for name in names}

    def instrument(self, name: str = None):
        """Decorator to auto-record function execution time."""
        def decorator(func):
            metric_name = name or func.__name__
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration_ms = (time.time() - start) * 1000
                    self.record(metric_name, duration_ms)
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper
        return decorator


# ---------- Global Instances ----------

_pool: Optional[ConnectionPool] = None
_monitor: Optional[PerformanceMonitor] = None


def get_connection_pool() -> ConnectionPool:
    """Get or create the global connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool()
    return _pool


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


__all__ = [
    "AsyncWorkflowRunner",
    "ConnectionPool",
    "OptimizedVectorIndex",
    "RequestBatcher",
    "PerformanceMonitor",
    "get_connection_pool",
    "get_performance_monitor",
]
