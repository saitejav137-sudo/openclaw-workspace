"""
Phase 4 Tests — Integration Hub + Performance Module

Tests for:
- IntegrationHub component lazy-init and tool registration
- AsyncWorkflowRunner parallel/sequential/pipeline modes
- ConnectionPool acquire/release lifecycle
- OptimizedVectorIndex add/search/delete
- RequestBatcher batching + flush
- PerformanceMonitor percentiles + instrumentation
"""

import sys
import os
import time
import math
import unittest
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================ Integration Hub Tests ================

class TestIntegrationHub(unittest.TestCase):
    """Tests for the central IntegrationHub."""

    def test_create_hub(self):
        from openclaw.core.integration_hub import IntegrationHub
        hub = IntegrationHub()
        self.assertFalse(hub._started)
        self.assertEqual(len(hub._components), 0)

    def test_get_status_empty(self):
        from openclaw.core.integration_hub import IntegrationHub
        hub = IntegrationHub()
        status = hub.get_status()
        self.assertFalse(status["started"])
        self.assertEqual(status["component_count"], 0)

    def test_lazy_scheduler(self):
        from openclaw.core.integration_hub import IntegrationHub
        hub = IntegrationHub()
        sched = hub.scheduler
        self.assertIsNotNone(sched)
        self.assertIn("scheduler", hub._components)

    def test_lazy_sandbox(self):
        from openclaw.core.integration_hub import IntegrationHub
        hub = IntegrationHub()
        sb = hub.sandbox
        self.assertIsNotNone(sb)
        self.assertIn("sandbox", hub._components)

    def test_lazy_knowledge_graph(self):
        from openclaw.core.integration_hub import IntegrationHub
        hub = IntegrationHub()
        kg = hub.knowledge_graph
        self.assertIsNotNone(kg)
        self.assertIn("knowledge_graph", hub._components)

    def test_lazy_react_agent(self):
        from openclaw.core.integration_hub import IntegrationHub
        hub = IntegrationHub()
        ra = hub.react_agent
        self.assertIsNotNone(ra)
        self.assertIn("react_agent", hub._components)

    def test_global_hub(self):
        from openclaw.core.integration_hub import get_integration_hub
        hub1 = get_integration_hub()
        hub2 = get_integration_hub()
        self.assertIs(hub1, hub2)

    def test_register_tools(self):
        from openclaw.core.integration_hub import IntegrationHub
        from openclaw.core.agent_tools import get_tool_registry
        hub = IntegrationHub()
        hub.register_tools()
        registry = get_tool_registry()
        tool_names = [t["name"] for t in registry.list_tools()]
        self.assertIn("run_code", tool_names)
        self.assertIn("research", tool_names)
        self.assertIn("automate", tool_names)
        self.assertIn("query_knowledge", tool_names)
        self.assertIn("schedule_task", tool_names)


# ================ Performance Module Tests ================

class TestOptimizedVectorIndex(unittest.TestCase):
    """Tests for the optimised vector index."""

    def setUp(self):
        from openclaw.core.performance import OptimizedVectorIndex
        self.index = OptimizedVectorIndex()

    def test_add_and_count(self):
        self.index.add("doc1", "hello world", [1.0, 0.0, 0.0])
        self.index.add("doc2", "goodbye world", [0.0, 1.0, 0.0])
        self.assertEqual(self.index.count(), 2)

    def test_search_exact_match(self):
        self.index.add("doc1", "hello", [1.0, 0.0])
        self.index.add("doc2", "goodbye", [0.0, 1.0])
        results = self.index.search([1.0, 0.0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "doc1")

    def test_search_with_filters(self):
        self.index.add("doc1", "a", [1.0, 0.0], {"type": "note"})
        self.index.add("doc2", "b", [0.9, 0.1], {"type": "code"})
        results = self.index.search([1.0, 0.0], top_k=5, filters={"type": "code"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "doc2")

    def test_delete(self):
        self.index.add("doc1", "hello", [1.0, 0.0])
        self.index.delete("doc1")
        self.assertEqual(self.index.count(), 0)

    def test_clear(self):
        for i in range(10):
            self.index.add(f"doc{i}", f"text{i}", [float(i), 0.0])
        self.index.clear()
        self.assertEqual(self.index.count(), 0)

    def test_cosine_similarity(self):
        from openclaw.core.performance import OptimizedVectorIndex
        sim = OptimizedVectorIndex._cosine_similarity([1, 0], [1, 0])
        self.assertAlmostEqual(sim, 1.0)

        sim = OptimizedVectorIndex._cosine_similarity([1, 0], [0, 1])
        self.assertAlmostEqual(sim, 0.0)

    def test_large_dataset_search(self):
        """Test search with > 200 entries triggers projection-based lookup."""
        import random
        random.seed(42)
        for i in range(300):
            emb = [random.random() for _ in range(10)]
            self.index.add(f"doc{i}", f"text{i}", emb)
        query = [0.5] * 10
        results = self.index.search(query, top_k=5)
        self.assertEqual(len(results), 5)
        # Results should be sorted by similarity descending
        sims = [r["similarity"] for r in results]
        self.assertEqual(sims, sorted(sims, reverse=True))


class TestConnectionPool(unittest.TestCase):
    """Tests for the connection pool."""

    def test_create_pool(self):
        from openclaw.core.performance import ConnectionPool
        pool = ConnectionPool(pool_size=3)
        stats = pool.get_stats()
        self.assertEqual(stats["pool_size"], 3)
        self.assertEqual(stats["in_use"], 0)

    def test_acquire_release(self):
        from openclaw.core.performance import ConnectionPool
        pool = ConnectionPool(pool_size=3)
        session = pool.acquire()
        self.assertIsNotNone(session)
        stats = pool.get_stats()
        self.assertEqual(stats["in_use"], 1)
        pool.release(session)
        stats = pool.get_stats()
        self.assertEqual(stats["in_use"], 0)
        self.assertEqual(stats["available"], 1)

    def test_close_all(self):
        from openclaw.core.performance import ConnectionPool
        pool = ConnectionPool(pool_size=3)
        s1 = pool.acquire()
        pool.release(s1)
        pool.close_all()
        stats = pool.get_stats()
        self.assertEqual(stats["available"], 0)


class TestPerformanceMonitor(unittest.TestCase):
    """Tests for the performance monitor."""

    def test_record_and_stats(self):
        from openclaw.core.performance import PerformanceMonitor
        mon = PerformanceMonitor()
        for i in range(100):
            mon.record("test_op", float(i))
        stats = mon.get_stats("test_op")
        self.assertEqual(stats["count"], 100)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 99.0)
        self.assertAlmostEqual(stats["avg"], 49.5)

    def test_percentiles(self):
        from openclaw.core.performance import PerformanceMonitor
        mon = PerformanceMonitor()
        for i in range(100):
            mon.record("latency", float(i))
        stats = mon.get_stats("latency")
        self.assertEqual(stats["p50"], 50.0)
        self.assertGreaterEqual(stats["p95"], 94.0)

    def test_empty_stats(self):
        from openclaw.core.performance import PerformanceMonitor
        mon = PerformanceMonitor()
        stats = mon.get_stats("nonexistent")
        self.assertEqual(stats["count"], 0)

    def test_instrument_decorator(self):
        from openclaw.core.performance import PerformanceMonitor
        mon = PerformanceMonitor()

        @mon.instrument("my_func")
        def slow_func():
            time.sleep(0.01)
            return 42

        result = slow_func()
        self.assertEqual(result, 42)
        stats = mon.get_stats("my_func")
        self.assertEqual(stats["count"], 1)
        self.assertGreater(stats["avg"], 5)  # > 5ms

    def test_get_all_stats(self):
        from openclaw.core.performance import PerformanceMonitor
        mon = PerformanceMonitor()
        mon.record("a", 1.0)
        mon.record("b", 2.0)
        all_stats = mon.get_all_stats()
        self.assertIn("a", all_stats)
        self.assertIn("b", all_stats)

    def test_window_size(self):
        from openclaw.core.performance import PerformanceMonitor
        mon = PerformanceMonitor(window_size=10)
        for i in range(50):
            mon.record("test", float(i))
        stats = mon.get_stats("test")
        self.assertEqual(stats["count"], 50)  # Total count recorded
        self.assertEqual(stats["min"], 40.0)  # Window only keeps last 10


class TestRequestBatcher(unittest.TestCase):
    """Tests for the request batcher."""

    def test_create_batcher(self):
        from openclaw.core.performance import RequestBatcher
        batcher = RequestBatcher(max_batch_size=4)
        stats = batcher.get_stats()
        self.assertEqual(stats["max_batch"], 4)
        self.assertEqual(stats["pending"], 0)

    def test_set_processor(self):
        from openclaw.core.performance import RequestBatcher
        batcher = RequestBatcher()

        def double_all(items):
            return [x * 2 for x in items]

        batcher.set_processor(double_all)
        self.assertIsNotNone(batcher._process_fn)

    def test_batch_flush(self):
        from openclaw.core.performance import RequestBatcher
        batcher = RequestBatcher(max_batch_size=2)
        results_collected = []

        def process_batch(items):
            return [x * 10 for x in items]

        batcher.set_processor(process_batch)

        # Submit in separate threads since submit blocks
        def submit_and_collect(val):
            result = batcher.submit(val, timeout=2.0)
            results_collected.append(result)

        t1 = threading.Thread(target=submit_and_collect, args=(3,))
        t2 = threading.Thread(target=submit_and_collect, args=(7,))
        t1.start()
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)
        self.assertEqual(sorted(results_collected), [30, 70])


class TestAsyncWorkflowRunner(unittest.TestCase):
    """Tests for the async workflow runner."""

    def test_create_runner(self):
        from openclaw.core.performance import AsyncWorkflowRunner
        runner = AsyncWorkflowRunner(max_workers=4)
        self.assertIsNotNone(runner)

    def test_run_parallel(self):
        from openclaw.core.performance import AsyncWorkflowRunner
        import asyncio

        runner = AsyncWorkflowRunner()

        def worker(ctx):
            return ctx.get("value", 0) * 2

        nodes = [
            ("node1", worker, {"value": 5}),
            ("node2", worker, {"value": 10}),
            ("node3", worker, {"value": 15}),
        ]

        results = asyncio.get_event_loop().run_until_complete(
            runner.run_parallel(nodes)
        )

        self.assertEqual(results["node1"], 10)
        self.assertEqual(results["node2"], 20)
        self.assertEqual(results["node3"], 30)
        runner.shutdown()

    def test_run_sequential(self):
        from openclaw.core.performance import AsyncWorkflowRunner
        import asyncio

        runner = AsyncWorkflowRunner()

        def step(ctx):
            return ctx.get("value", 0) + 1

        nodes = [
            ("s1", step, {"value": 0}),
            ("s2", step, {"value": 10}),
        ]

        results = asyncio.get_event_loop().run_until_complete(
            runner.run_sequential(nodes)
        )

        self.assertIn("s1", results)
        self.assertIn("s2", results)
        runner.shutdown()


# ================ Error Handler Thread-Safety Tests ================

class TestErrorHandlerThreadSafety(unittest.TestCase):
    """Verify ErrorHandler is thread-safe after Phase 4A fix."""

    def test_instance_level_errors(self):
        from openclaw.utils.errors import ErrorHandler
        h1 = ErrorHandler()
        h2 = ErrorHandler()
        h1.record_error(ValueError("err1"))
        # h2 should NOT see h1's errors (instance-level, not class-level)
        self.assertEqual(len(h2.get_recent_errors()), 0)

    def test_concurrent_record(self):
        from openclaw.utils.errors import ErrorHandler
        handler = ErrorHandler()
        errors_to_add = 100

        def record_errors():
            for i in range(errors_to_add):
                handler.record_error(ValueError(f"err-{i}"))

        threads = [threading.Thread(target=record_errors) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = len(handler.get_recent_errors(limit=500))
        self.assertEqual(total, 400)

    def test_has_lock(self):
        from openclaw.utils.errors import ErrorHandler
        handler = ErrorHandler()
        self.assertTrue(hasattr(handler, '_lock'))


if __name__ == "__main__":
    unittest.main()
