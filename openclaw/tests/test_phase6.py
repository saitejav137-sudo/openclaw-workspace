"""
Phase 6 Tests — Telemetry, Plugin System, Health Dashboard, and Integration Tests

Tests for:
- Tracer: span creation, nesting, traces
- MetricsCollector: counters, gauges, histograms
- PluginManager: registration, hooks, emit
- HealthDashboard: checks, aggregated status
- Integration tests: auth, HTTP endpoints, browser agent
"""

import sys
import os
import time
import unittest
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============== Tracer Tests ==============

class TestTracer(unittest.TestCase):

    def test_create_span(self):
        from openclaw.core.telemetry import Tracer
        tracer = Tracer()
        with tracer.span("test_op") as s:
            s.add_event("started")
        self.assertGreater(s.duration_ms, 0)
        self.assertEqual(s.status, "ok")

    def test_nested_spans(self):
        from openclaw.core.telemetry import Tracer
        tracer = Tracer()
        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                pass
        # Child should share parent's trace_id
        self.assertEqual(parent.trace_id, child.trace_id)
        self.assertEqual(child.parent_id, parent.span_id)

    def test_error_span(self):
        from openclaw.core.telemetry import Tracer
        tracer = Tracer()
        try:
            with tracer.span("failing") as s:
                raise ValueError("boom")
        except ValueError:
            pass
        self.assertEqual(s.status, "error")
        self.assertEqual(len(s.events), 1)
        self.assertEqual(s.events[0]["name"], "exception")

    def test_span_count(self):
        from openclaw.core.telemetry import Tracer
        tracer = Tracer()
        for i in range(5):
            with tracer.span(f"op_{i}"):
                pass
        self.assertEqual(tracer.get_span_count(), 5)

    def test_get_traces(self):
        from openclaw.core.telemetry import Tracer
        tracer = Tracer()
        with tracer.span("root"):
            with tracer.span("child1"):
                pass
            with tracer.span("child2"):
                pass
        traces = tracer.get_traces()
        self.assertGreater(len(traces), 0)
        # Should have one trace with 3 spans
        self.assertEqual(len(traces[0]["spans"]), 3)

    def test_clear(self):
        from openclaw.core.telemetry import Tracer
        tracer = Tracer()
        with tracer.span("x"):
            pass
        tracer.clear()
        self.assertEqual(tracer.get_span_count(), 0)

    def test_span_tags(self):
        from openclaw.core.telemetry import Tracer
        tracer = Tracer()
        with tracer.span("tagged", tags={"env": "test"}) as s:
            pass
        self.assertEqual(s.tags["env"], "test")

    def test_thread_safety(self):
        from openclaw.core.telemetry import Tracer
        tracer = Tracer()

        def create_spans():
            for i in range(20):
                with tracer.span(f"thread_op_{i}"):
                    pass

        threads = [threading.Thread(target=create_spans) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(tracer.get_span_count(), 100)


# ============== Metrics Collector Tests ==============

class TestMetricsCollector(unittest.TestCase):

    def test_increment_counter(self):
        from openclaw.core.telemetry import MetricsCollector
        m = MetricsCollector()
        m.increment("requests")
        m.increment("requests")
        m.increment("requests", 3)
        all_metrics = m.get_all()
        self.assertEqual(all_metrics["counters"]["requests"], 5.0)

    def test_counter_with_tags(self):
        from openclaw.core.telemetry import MetricsCollector
        m = MetricsCollector()
        m.increment("http_requests", tags={"method": "GET"})
        m.increment("http_requests", tags={"method": "POST"})
        all_metrics = m.get_all()
        self.assertIn("http_requests{method=GET}", all_metrics["counters"])
        self.assertIn("http_requests{method=POST}", all_metrics["counters"])

    def test_gauge(self):
        from openclaw.core.telemetry import MetricsCollector
        m = MetricsCollector()
        m.gauge("cpu_usage", 45.2)
        m.gauge("cpu_usage", 67.8)
        all_metrics = m.get_all()
        self.assertEqual(all_metrics["gauges"]["cpu_usage"], 67.8)

    def test_histogram(self):
        from openclaw.core.telemetry import MetricsCollector
        m = MetricsCollector()
        for i in range(100):
            m.histogram("latency", float(i))
        all_metrics = m.get_all()
        self.assertEqual(all_metrics["histograms"]["latency"]["count"], 100)
        self.assertAlmostEqual(all_metrics["histograms"]["latency"]["avg"], 49.5)

    def test_empty_metrics(self):
        from openclaw.core.telemetry import MetricsCollector
        m = MetricsCollector()
        all_metrics = m.get_all()
        self.assertEqual(all_metrics["counters"], {})
        self.assertEqual(all_metrics["gauges"], {})


# ============== Plugin Manager Tests ==============

class TestPluginManager(unittest.TestCase):

    def test_register_plugin(self):
        from openclaw.core.telemetry import PluginManager, PluginInfo
        pm = PluginManager()
        pm.register(
            PluginInfo(name="test_plugin", version="1.0", description="Test"),
            hooks={"on_startup": lambda: "started"}
        )
        plugins = pm.list_plugins()
        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0]["name"], "test_plugin")

    def test_emit_hook(self):
        from openclaw.core.telemetry import PluginManager, PluginInfo
        pm = PluginManager()
        results = []
        pm.register(
            PluginInfo(name="logger", version="1.0", description="Logs events"),
            hooks={"on_task_start": lambda **kw: results.append(kw)}
        )
        pm.emit("on_task_start", task_name="test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["task_name"], "test")

    def test_invalid_hook(self):
        from openclaw.core.telemetry import PluginManager, PluginInfo
        pm = PluginManager()
        with self.assertRaises(ValueError):
            pm.register(
                PluginInfo(name="bad", version="1", description=""),
                hooks={"invalid_hook": lambda: None}
            )

    def test_emit_no_handlers(self):
        from openclaw.core.telemetry import PluginManager
        pm = PluginManager()
        results = pm.emit("on_startup")
        self.assertEqual(results, [])

    def test_multiple_plugins(self):
        from openclaw.core.telemetry import PluginManager, PluginInfo
        pm = PluginManager()
        call_order = []
        pm.register(
            PluginInfo(name="p1", version="1", description=""),
            hooks={"on_startup": lambda: call_order.append("p1")}
        )
        pm.register(
            PluginInfo(name="p2", version="1", description=""),
            hooks={"on_startup": lambda: call_order.append("p2")}
        )
        pm.emit("on_startup")
        self.assertEqual(call_order, ["p1", "p2"])

    def test_unregister(self):
        from openclaw.core.telemetry import PluginManager, PluginInfo
        pm = PluginManager()
        pm.register(
            PluginInfo(name="temp", version="1", description=""),
            hooks={"on_startup": lambda: None}
        )
        pm.unregister("temp")
        self.assertEqual(len(pm.list_plugins()), 0)


# ============== Health Dashboard Tests ==============

class TestHealthDashboard(unittest.TestCase):

    def test_register_and_run(self):
        from openclaw.core.telemetry import HealthDashboard
        hd = HealthDashboard()
        hd.register_check("memory", lambda: {"healthy": True, "usage_mb": 256})
        result = hd.run_all()
        self.assertEqual(result["overall"], "healthy")
        self.assertEqual(result["total_checks"], 1)

    def test_degraded_health(self):
        from openclaw.core.telemetry import HealthDashboard
        hd = HealthDashboard()
        hd.register_check("ok_service", lambda: {"healthy": True})
        hd.register_check("bad_service", lambda: {"healthy": False, "reason": "down"})
        result = hd.run_all()
        self.assertEqual(result["overall"], "degraded")
        self.assertEqual(result["healthy_count"], 1)

    def test_check_exception(self):
        from openclaw.core.telemetry import HealthDashboard
        hd = HealthDashboard()
        hd.register_check("failing", lambda: 1 / 0)
        result = hd.run_all()
        self.assertEqual(result["overall"], "degraded")
        self.assertFalse(result["checks"]["failing"]["healthy"])

    def test_empty_dashboard(self):
        from openclaw.core.telemetry import HealthDashboard
        hd = HealthDashboard()
        result = hd.run_all()
        self.assertEqual(result["overall"], "healthy")
        self.assertEqual(result["total_checks"], 0)

    def test_check_timing(self):
        from openclaw.core.telemetry import HealthDashboard
        hd = HealthDashboard()
        hd.register_check("slow", lambda: (time.sleep(0.01), {"healthy": True})[1])
        result = hd.run_all()
        self.assertGreater(result["checks"]["slow"]["duration_ms"], 5)


# ============== Auth Integration Tests ==============

class TestAuthIntegration(unittest.TestCase):

    def test_auth_manager_create(self):
        from openclaw.integrations.auth import AuthManager
        am = AuthManager()
        self.assertIsNotNone(am)

    def test_default_users_created(self):
        from openclaw.integrations.auth import AuthManager
        am = AuthManager()
        users = am.list_users()
        self.assertGreater(len(users), 0)

    def test_create_and_authenticate_user(self):
        from openclaw.integrations.auth import AuthManager, UserRole
        am = AuthManager()
        user = am.create_user("test_user_phase6", "securepass123", UserRole.VIEWER)
        self.assertIsNotNone(user)
        # Authenticate
        auth_user = am.authenticate("test_user_phase6", "securepass123")
        self.assertIsNotNone(auth_user)
        self.assertEqual(auth_user.username, "test_user_phase6")

    def test_wrong_password_fails(self):
        from openclaw.integrations.auth import AuthManager, UserRole
        am = AuthManager()
        am.create_user("wrongpw_user", "correct", UserRole.VIEWER)
        result = am.authenticate("wrongpw_user", "incorrect")
        self.assertIsNone(result)

    def test_api_key_auth(self):
        from openclaw.integrations.auth import AuthManager, UserRole
        am = AuthManager()
        user = am.create_user("apikey_user", "pass123", UserRole.OPERATOR)
        self.assertIsNotNone(user.api_key)
        found = am.authenticate_api_key(user.api_key)
        self.assertIsNotNone(found)
        self.assertEqual(found.username, "apikey_user")

    def test_session_management(self):
        from openclaw.integrations.auth import AuthManager, UserRole
        am = AuthManager()
        user = am.create_user("session_user", "pass", UserRole.VIEWER)
        session = am.create_session(user.id)
        self.assertIsNotNone(session)
        # Retrieve session
        got = am.get_session(session.id)
        self.assertIsNotNone(got)
        self.assertEqual(got.user_id, user.id)

    def test_rbac_permissions(self):
        from openclaw.integrations.auth import AuthManager, UserRole, Permission
        am = AuthManager()
        viewer = am.create_user("viewer_user", "pass", UserRole.VIEWER)
        admin = am.create_user("admin_user", "pass", UserRole.ADMIN)
        # Viewer has read access
        self.assertTrue(am.has_permission(viewer, Permission.TRIGGER_READ))
        # Viewer lacks create
        self.assertFalse(am.has_permission(viewer, Permission.TRIGGER_CREATE))
        # Admin has all
        self.assertTrue(am.has_permission(admin, Permission.TRIGGER_CREATE))
        self.assertTrue(am.has_permission(admin, Permission.ADMIN_ALL))


# ============== HTTP Handler Tests ==============

class TestHTTPRateLimiter(unittest.TestCase):

    def test_rate_limiter_allow(self):
        from openclaw.integrations.http import RateLimiter
        rl = RateLimiter(rate=10, per=1.0)
        self.assertTrue(rl.is_allowed())

    def test_rate_limiter_exhaust(self):
        from openclaw.integrations.http import RateLimiter
        rl = RateLimiter(rate=3, per=60.0)
        for _ in range(3):
            rl.is_allowed()
        self.assertFalse(rl.is_allowed())

    def test_rate_limiter_reset(self):
        from openclaw.integrations.http import RateLimiter
        rl = RateLimiter(rate=1, per=60.0)
        rl.is_allowed()
        rl.reset()
        self.assertTrue(rl.is_allowed())


class TestHTTPValidation(unittest.TestCase):

    def test_validate_url_valid(self):
        from openclaw.integrations.http import validate_url
        self.assertTrue(validate_url("https://example.com"))
        self.assertTrue(validate_url("http://google.com/search?q=test"))

    def test_validate_url_invalid(self):
        from openclaw.integrations.http import validate_url
        self.assertFalse(validate_url(""))
        self.assertFalse(validate_url("ftp://example.com"))
        self.assertFalse(validate_url("file:///etc/passwd"))

    def test_validate_action_valid(self):
        from openclaw.integrations.http import validate_action
        self.assertTrue(validate_action("start"))
        self.assertTrue(validate_action("click"))
        self.assertTrue(validate_action("ctrl+c"))
        self.assertTrue(validate_action("Return"))

    def test_validate_action_invalid(self):
        from openclaw.integrations.http import validate_action
        self.assertFalse(validate_action(""))
        self.assertFalse(validate_action(None))
        self.assertFalse(validate_action("rm -rf /"))

    def test_sanitize_string(self):
        from openclaw.integrations.http import sanitize_string
        self.assertEqual(sanitize_string("hello"), "hello")
        self.assertEqual(sanitize_string("he\x00llo"), "hello")

    def test_validate_selector(self):
        from openclaw.integrations.http import validate_selector
        self.assertTrue(validate_selector("#myButton"))
        self.assertTrue(validate_selector(".class-name"))
        self.assertFalse(validate_selector(""))
        self.assertFalse(validate_selector("<script>alert(1)</script>"))


# ============== Global Singleton Tests ==============

class TestTelemetrySingletons(unittest.TestCase):

    def test_get_tracer(self):
        from openclaw.core.telemetry import get_tracer
        t1 = get_tracer()
        t2 = get_tracer()
        self.assertIs(t1, t2)

    def test_get_metrics(self):
        from openclaw.core.telemetry import get_metrics
        m1 = get_metrics()
        m2 = get_metrics()
        self.assertIs(m1, m2)

    def test_get_plugin_manager(self):
        from openclaw.core.telemetry import get_plugin_manager
        p1 = get_plugin_manager()
        p2 = get_plugin_manager()
        self.assertIs(p1, p2)

    def test_get_health_dashboard(self):
        from openclaw.core.telemetry import get_health_dashboard
        h1 = get_health_dashboard()
        h2 = get_health_dashboard()
        self.assertIs(h1, h2)


if __name__ == "__main__":
    unittest.main()
