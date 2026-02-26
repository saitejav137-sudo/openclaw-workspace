"""Comprehensive tests for advanced AI features"""

import unittest
import tempfile
import os
import sys
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openclaw.core.ai_advanced import (
    YOLOVersion,
    YOLODetector,
    DetectionResult,
    SAMSegmenter,
    AgenticAI,
    AgentState,
    AgentAction,
    EventDrivenOrchestrator,
    EventType,
    AutomationEvent,
    MultiAgentCoordinator,
    AgentInfo,
    MCPClient,
)
from openclaw.core.streaming_analysis import (
    StreamSource,
    StreamConfig,
    StreamProcessor,
    OpticalFlowDetector,
)


class TestYOLODetector(unittest.TestCase):
    """Test YOLO detector"""

    def test_yolo_version_enum(self):
        """Test YOLO version enum"""
        self.assertEqual(YOLOVersion.YOLO11.value, "yolo11")
        self.assertEqual(YOLOVersion.YOLO26.value, "yolo26")

    def test_detector_init(self):
        """Test detector initialization"""
        detector = YOLODetector(
            model_version=YOLOVersion.YOLO11,
            conf_threshold=0.3
        )
        self.assertEqual(detector.conf_threshold, 0.3)
        self.assertFalse(detector._initialized)

    def test_detector_initialize_no_model(self):
        """Test detector without loading model"""
        detector = YOLODetector(model_path="nonexistent.pt")
        # Should not crash, just return False
        result = detector.initialize()
        self.assertFalse(result)


class TestAgenticAI(unittest.TestCase):
    """Test agentic AI"""

    def test_agent_state_enum(self):
        """Test agent state enum"""
        self.assertEqual(AgentState.IDLE.value, "idle")
        self.assertEqual(AgentState.THINKING.value, "thinking")

    def test_agent_action_creation(self):
        """Test agent action creation"""
        action = AgentAction(
            action_type="trigger",
            params={"key": "value"},
            confidence=0.9,
            reasoning="Test reasoning"
        )
        self.assertEqual(action.action_type, "trigger")
        self.assertEqual(action.confidence, 0.9)

    def test_agentic_ai_init(self):
        """Test agentic AI initialization"""
        agent = AgenticAI(name="test_agent", vision_model="yolo11n.pt")
        self.assertEqual(agent.name, "test_agent")
        self.assertEqual(agent.state, AgentState.IDLE)


class TestEventOrchestrator(unittest.TestCase):
    """Test event-driven orchestrator"""

    def test_event_type_enum(self):
        """Test event type enum"""
        self.assertEqual(EventType.SCREEN_CHANGE.value, "screen_change")
        self.assertEqual(EventType.OBJECT_DETECTED.value, "object_detected")

    def test_orchestrator_init(self):
        """Test orchestrator initialization"""
        orchestrator = EventDrivenOrchestrator()
        self.assertFalse(orchestrator._running)

    def test_subscribe_unsubscribe(self):
        """Test subscribe/unsubscribe"""
        orchestrator = EventDrivenOrchestrator()
        called = []

        def handler(event):
            called.append(event)

        # Subscribe
        orchestrator.subscribe(EventType.CUSTOM, handler)
        self.assertIn(EventType.CUSTOM, orchestrator._listeners)

        # Unsubscribe - just verify handler was removed
        orchestrator.unsubscribe(EventType.CUSTOM, handler)
        # Handler list may be empty but key exists
        self.assertEqual(len(orchestrator._listeners.get(EventType.CUSTOM, [])), 0)


class TestMultiAgentCoordinator(unittest.TestCase):
    """Test multi-agent coordinator"""

    def test_coordinator_init(self):
        """Test coordinator initialization"""
        coordinator = MultiAgentCoordinator()
        self.assertEqual(len(coordinator._agents), 0)

    def test_register_agent(self):
        """Test agent registration"""
        coordinator = MultiAgentCoordinator()
        agent = coordinator.register_agent(
            agent_id="agent_1",
            name="Test Agent",
            role="vision",
            capabilities=["detection", "tracking"]
        )

        self.assertIsNotNone(agent)
        info = coordinator._agent_info["agent_1"]
        self.assertEqual(info.name, "Test Agent")
        self.assertEqual(info.role, "vision")

    def test_unregister_agent(self):
        """Test agent unregistration"""
        coordinator = MultiAgentCoordinator()
        coordinator.register_agent("agent_1", "Test", "vision", [])

        result = coordinator.unregister_agent("agent_1")
        self.assertTrue(result)
        self.assertNotIn("agent_1", coordinator._agents)

    def test_shared_context(self):
        """Test shared context"""
        coordinator = MultiAgentCoordinator()
        coordinator.update_shared_context("screen_size", (1920, 1080))

        context = coordinator.get_shared_context()
        self.assertEqual(context["screen_size"], (1920, 1080))


class TestMCPClient(unittest.TestCase):
    """Test MCP client"""

    def test_client_init(self):
        """Test client initialization"""
        client = MCPClient(server_url="http://localhost:8000")
        self.assertEqual(client.server_url, "http://localhost:8000")
        self.assertFalse(client._connected)

    def test_register_tool(self):
        """Test tool registration"""
        client = MCPClient()
        client.register_tool(
            name="vision_detect",
            description="Detect objects in image",
            parameters={"image": "string"}
        )

        tools = client.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "vision_detect")


class TestStreamingAnalysis(unittest.TestCase):
    """Test streaming analysis"""

    def test_stream_source_enum(self):
        """Test stream source enum"""
        self.assertEqual(StreamSource.SCREEN.value, "screen")
        self.assertEqual(StreamSource.WEBCAM.value, "webcam")

    def test_stream_config(self):
        """Test stream config"""
        config = StreamConfig(
            source=StreamSource.SCREEN,
            fps=30,
            resolution=(1920, 1080)
        )
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.resolution, (1920, 1080))

    def test_stream_processor_init(self):
        """Test processor initialization"""
        processor = StreamProcessor()
        self.assertFalse(processor.is_running)
        self.assertEqual(processor.frame_count, 0)

    def test_optical_flow_detector(self):
        """Test optical flow detector"""
        import numpy as np

        detector = OpticalFlowDetector(threshold=2.0)
        self.assertEqual(detector.threshold, 2.0)

        # Create test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_motion(frame)
        # First frame returns None
        self.assertIsNone(result)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_full_agent_flow(self):
        """Test complete agent flow"""
        # Create agent
        agent = AgenticAI(name="integration_test")
        self.assertEqual(agent.name, "integration_test")

        # Create coordinator
        coordinator = MultiAgentCoordinator()
        coordinator.register_agent(
            agent_id="test",
            name="Test",
            role="test",
            capabilities=[]
        )

        # Update shared context
        coordinator.update_shared_context("test_key", "test_value")
        context = coordinator.get_shared_context()

        self.assertEqual(context["test_key"], "test_value")

    def test_event_to_agent_flow(self):
        """Test event triggering agent"""
        orchestrator = EventDrivenOrchestrator()
        events_received = []

        def test_handler(event):
            events_received.append(event)

        orchestrator.subscribe(EventType.CUSTOM, test_handler)
        self.assertIn(EventType.CUSTOM, orchestrator._listeners)


if __name__ == "__main__":
    unittest.main()
