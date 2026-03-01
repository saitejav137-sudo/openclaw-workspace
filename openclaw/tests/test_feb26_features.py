"""
Tests for OpenClaw v2026.2.26 Feature Port

Tests security hardening, secrets management enhancements,
delivery queue resilience, and Telegram improvements.
"""

import os
import time
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# ============== Security Enhanced Tests ==============

class TestSSRFGuard(unittest.TestCase):
    """Test SSRF protection with IPv6 multicast blocking."""

    def setUp(self):
        from openclaw.core.security_enhanced import SSRFGuard
        self.guard = SSRFGuard()

    def test_blocks_localhost(self):
        self.assertFalse(self.guard._is_safe_ip("127.0.0.1"))
        self.assertFalse(self.guard._is_safe_ip("::1"))

    def test_blocks_private_ranges(self):
        self.assertFalse(self.guard._is_safe_ip("10.0.0.1"))
        self.assertFalse(self.guard._is_safe_ip("172.16.0.1"))
        self.assertFalse(self.guard._is_safe_ip("192.168.1.1"))

    def test_blocks_ipv6_multicast(self):
        """IPv6 multicast (ff00::/8) — v2026.2.26."""
        self.assertFalse(self.guard._is_safe_ip("ff02::1"))
        self.assertFalse(self.guard._is_safe_ip("ff05::2"))
        self.assertFalse(self.guard._is_safe_ip("ff0e::1"))

    def test_blocks_ipv4_multicast(self):
        self.assertFalse(self.guard._is_safe_ip("224.0.0.1"))
        self.assertFalse(self.guard._is_safe_ip("239.255.255.250"))

    def test_allows_public_ips(self):
        self.assertTrue(self.guard._is_safe_ip("8.8.8.8"))
        self.assertTrue(self.guard._is_safe_ip("1.1.1.1"))

    def test_blocks_link_local(self):
        self.assertFalse(self.guard._is_safe_ip("169.254.1.1"))
        self.assertFalse(self.guard._is_safe_ip("fe80::1"))

    def test_allowlist(self):
        from openclaw.core.security_enhanced import SSRFGuard
        g = SSRFGuard(allowed_hosts=["localhost"])
        with patch("socket.getaddrinfo", return_value=[(2, 1, 6, '', ("127.0.0.1", 0))]):
            self.assertTrue(g.is_safe_url("http://localhost/api"))

    def test_invalid_ip(self):
        self.assertFalse(self.guard._is_safe_ip("not_an_ip"))


class TestPathGuard(unittest.TestCase):
    """Test symlink/hardlink path traversal protection."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from openclaw.core.security_enhanced import PathGuard
        self.guard = PathGuard(workspace_root=self.tmpdir)

    def test_safe_path(self):
        self.assertTrue(self.guard.is_safe_path(os.path.join(self.tmpdir, "sub", "file.txt")))

    def test_blocks_outside(self):
        self.assertFalse(self.guard.is_safe_path("/etc/passwd"))

    def test_blocks_symlink_escape(self):
        link = os.path.join(self.tmpdir, "escape")
        os.symlink("/etc", link)
        self.assertFalse(self.guard.is_safe_path(os.path.join(link, "passwd")))

    def test_allows_internal_symlink(self):
        sub = os.path.join(self.tmpdir, "real")
        os.makedirs(sub, exist_ok=True)
        link = os.path.join(self.tmpdir, "link")
        os.symlink(sub, link)
        self.assertTrue(self.guard.is_safe_path(os.path.join(link, "file.txt")))

    def test_validate_write(self):
        self.assertTrue(self.guard.validate_write(os.path.join(self.tmpdir, "new.txt")))
        self.assertFalse(self.guard.validate_write("/etc/shadow"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestExecApprovalGuard(unittest.TestCase):
    """Test exec approval hardening."""

    def setUp(self):
        from openclaw.core.security_enhanced import ExecApprovalGuard
        self.guard = ExecApprovalGuard()
        self.tmpdir = tempfile.mkdtemp()

    def test_approve_and_check(self):
        self.guard.approve("p1", ["ls", "-la"], self.tmpdir)
        self.assertTrue(self.guard.check_approval("p1", ["ls", "-la"], self.tmpdir))

    def test_rejects_argv_mismatch(self):
        self.guard.approve("p2", ["ls"], self.tmpdir)
        self.assertFalse(self.guard.check_approval("p2", ["ls ", ""], self.tmpdir))

    def test_rejects_trailing_space(self):
        self.guard.approve("p3", ["python3"], self.tmpdir)
        self.assertFalse(self.guard.check_approval("p3", ["python3 "], self.tmpdir))

    def test_rejects_unknown(self):
        self.assertFalse(self.guard.check_approval("unknown", ["ls"], self.tmpdir))

    def test_revoke(self):
        self.guard.approve("p4", ["echo"], self.tmpdir)
        self.guard.revoke("p4")
        self.assertFalse(self.guard.check_approval("p4", ["echo"], self.tmpdir))

    def test_rejects_symlink_cwd(self):
        link = os.path.join(self.tmpdir, "sym")
        os.symlink("/tmp", link)
        self.assertFalse(self.guard.approve("p5", ["ls"], link))

    def test_stats(self):
        self.guard.approve("x1", ["echo"], self.tmpdir)
        stats = self.guard.get_stats()
        self.assertEqual(stats["active_approvals"], 1)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ============== Secrets Tests ==============

class TestSecretsV2026226(unittest.TestCase):
    """Test secrets v2026.2.26 features."""

    def setUp(self):
        from openclaw.core.secrets import SecretsManager
        SecretsManager.reset()
        self.tmpdir = tempfile.mkdtemp()
        self.sm = SecretsManager(
            env_file=os.path.join(self.tmpdir, ".env"),
            secrets_dir=os.path.join(self.tmpdir, "secrets")
        )

    def test_snapshot(self):
        self.sm.set("K1", "test_value_12345", persist=False)
        result = self.sm.snapshot("snap1")
        self.assertEqual(result["label"], "snap1")
        self.assertGreater(result["key_count"], 0)

    def test_configure(self):
        os.environ["CFG_KEY"] = "val"
        try:
            result = self.sm.configure({"CFG_KEY": "desc", "MISSING": "desc"})
            self.assertEqual(result["total"], 2)
            self.assertTrue(result["keys"]["CFG_KEY"]["configured"])
            self.assertFalse(result["keys"]["MISSING"]["configured"])
        finally:
            del os.environ["CFG_KEY"]

    def test_apply_blocks_bad_path(self):
        result = self.sm.apply(target_path="/etc/evil.env")
        self.assertFalse(result["success"])

    def test_apply_blocks_symlink(self):
        link = os.path.join(self.tmpdir, "link.env")
        os.symlink("/etc/passwd", link)
        result = self.sm.apply(target_path=link)
        self.assertFalse(result["success"])

    def test_audit(self):
        self.sm.set("AK", "audit_val_1234", persist=False)
        result = self.sm.audit()
        self.assertIn("AK", result)
        self.assertTrue(result["AK"]["present"])

    def test_reload(self):
        self.sm.set("RK", "reload_val", persist=False)
        self.sm.reload()
        self.assertNotIn("RK", self.sm._cache)

    def tearDown(self):
        from openclaw.core.secrets import SecretsManager
        SecretsManager.reset()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ============== Task Queue Tests ==============

class TestTaskQueueV2026226(unittest.TestCase):
    """Test delivery queue v2026.2.26 features."""

    def setUp(self):
        from openclaw.core.task_queue import TaskQueue
        self.queue = TaskQueue(max_concurrent=5)

    def test_drain_rejects(self):
        from openclaw.core.task_queue import Priority
        self.queue.start_drain()
        with self.assertRaises(RuntimeError):
            self.queue.enqueue("test", lambda: None, priority=Priority.NORMAL)
        self.queue.stop_drain()

    def test_drain_wait_resets(self):
        result = self.queue.drain_and_wait(timeout=1.0)
        self.assertTrue(result)
        self.assertFalse(self.queue._draining)

    def test_drain_timeout_resets(self):
        from openclaw.core.task_queue import QueuedTask, QueuedTaskStatus
        fake = QueuedTask(name="stuck", func=lambda: time.sleep(100))
        fake.status = QueuedTaskStatus.RUNNING
        fake.started_at = time.time()
        self.queue._running["fake"] = fake
        result = self.queue.drain_and_wait(timeout=1.0)
        self.assertFalse(result)
        self.assertFalse(self.queue._draining)
        self.queue._running.clear()

    def test_backoff_persistence(self):
        from openclaw.core.task_queue import Priority
        def fail():
            raise ValueError("fail")
        tid = self.queue.enqueue("f", fail, priority=Priority.NORMAL, max_retries=1)
        self.queue.process_next()
        task = self.queue.get_task(tid)
        self.assertIsNotNone(task.last_attempt_at)

    def test_stats_draining(self):
        stats = self.queue.get_stats()
        self.assertIn("draining", stats)

    def test_recover_deferred(self):
        from openclaw.core.task_queue import QueuedTask, Priority
        import heapq
        t = QueuedTask(name="def", func=lambda: "ok",
                       priority=Priority.NORMAL, retry_count=1,
                       last_attempt_at=time.time() - 1000)
        with self.queue._lock:
            heapq.heappush(self.queue._heap, t)
            self.queue._tasks[t.id] = t
        recovered = self.queue.recover_deferred()
        self.assertIn(t.id, recovered)


# ============== Telegram Tests ==============

class TestTelegramV2026226(unittest.TestCase):
    """Test Telegram v2026.2.26 improvements."""

    def test_typing_backoff(self):
        from openclaw.integrations.telegram import TelegramBot
        bot = TelegramBot(token="test", chat_id="123")
        bot.enabled = False
        bot._typing_failures = 3
        bot._typing_suppressed = True
        bot._typing_backoff_until = time.time() + 60
        bot.send_typing_action()
        self.assertTrue(bot._typing_suppressed)

    def test_callback_auth(self):
        from openclaw.integrations.telegram import TelegramBot
        bot = TelegramBot(token="test", chat_id="123")
        bot.enabled = False
        q = {"id": "q1", "from": {"id": 999}, "data": "/help",
             "message": {"chat": {"id": 456}}}
        self.assertIsNone(bot._handle_callback_query(q))

    def test_reply_media_photo(self):
        from openclaw.integrations.telegram import TelegramBot
        bot = TelegramBot(token="test", chat_id="123")
        bot.enabled = False
        msg = {"reply_to_message": {"photo": [{"file_id": "abc"}], "text": "orig"}}
        ctx = bot._extract_reply_media_context(msg)
        self.assertIn("Replied to image", ctx)

    def test_reply_media_none(self):
        from openclaw.integrations.telegram import TelegramBot
        bot = TelegramBot(token="test", chat_id="123")
        bot.enabled = False
        self.assertIsNone(bot._extract_reply_media_context({"text": "hi"}))

    def test_reply_media_doc(self):
        from openclaw.integrations.telegram import TelegramBot
        bot = TelegramBot(token="test", chat_id="123")
        bot.enabled = False
        msg = {"reply_to_message": {"document": {"file_name": "report.pdf"}}}
        ctx = bot._extract_reply_media_context(msg)
        self.assertIn("report.pdf", ctx)

    def test_command_degradation_attr(self):
        from openclaw.integrations.telegram import TelegramBot
        bot = TelegramBot(token="test", chat_id="123")
        bot.enabled = False
        self.assertFalse(bot._command_registration_degraded)
        self.assertEqual(bot._max_bot_commands, 100)


if __name__ == "__main__":
    unittest.main()
