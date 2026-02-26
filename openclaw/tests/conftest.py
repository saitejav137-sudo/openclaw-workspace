"""
Pytest configuration and shared fixtures for OpenClaw tests.
"""

import os
import sys
import pytest
from typing import Generator
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# Add the openclaw package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============== Fixtures ==============

@pytest.fixture
def mock_config():
    """Create a mock VisionConfig for testing."""
    from openclaw.core.config import VisionConfig, VisionMode

    return VisionConfig(
        mode=VisionMode.OCR,
        target_text="test",
        polling=False,
        poll_interval=0.5,
        action="alt+o",
        action_delay=0.1,
    )


@pytest.fixture
def mock_vision_engine(mock_config):
    """Create a mock VisionEngine for testing."""
    from openclaw.core.vision import VisionEngine

    engine = VisionEngine(mock_config)
    engine.process = Mock(return_value=True)
    return engine


@pytest.fixture
def mock_ocr_engine():
    """Create a mock OCREngine for testing."""
    from openclaw.core.vision import OCREngine

    engine = OCREngine(languages=["en"])
    engine.read = Mock(return_value="test text")
    return engine


@pytest.fixture
def sample_image():
    """Create a sample numpy array image for testing."""
    import numpy as np

    # Create a small test image (100x100 RGB)
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_screen_capture():
    """Mock ScreenCapture for testing."""
    with patch("openclaw.core.vision.ScreenCapture") as mock:
        mock.capture_region = Mock(return_value=None)
        mock.capture_full = Mock(return_value=None)
        mock.clear_cache = Mock()
        yield mock


@pytest.fixture
def mock_trigger_action():
    """Mock TriggerAction for testing."""
    with patch("openclaw.core.actions.TriggerAction") as mock:
        mock.execute = Mock(return_value=True)
        yield mock


@pytest.fixture
def mock_keyboard_action():
    """Mock KeyboardAction for testing."""
    with patch("openclaw.core.actions.KeyboardAction") as mock:
        mock.press = Mock(return_value=True)
        mock.type_text = Mock(return_value=True)
        mock.hotkey = Mock(return_value=True)
        yield mock


@pytest.fixture
def mock_mouse_action():
    """Mock MouseAction for testing."""
    with patch("openclaw.core.actions.MouseAction") as mock:
        mock.move = Mock(return_value=True)
        mock.click = Mock(return_value=True)
        mock.double_click = Mock(return_value=True)
        mock.drag = Mock(return_value=True)
        mock.scroll = Mock(return_value=True)
        yield mock


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    import yaml

    config_data = {
        "mode": "ocr",
        "target_text": "test",
        "action": "alt+o",
        "action_delay": 0.5,
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return str(config_file)


@pytest.fixture
def api_client():
    """Create a test client for FastAPI testing."""
    from fastapi.testclient import TestClient
    from openclaw.integrations.fastapi_server import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authentication headers for API testing."""
    return {"Authorization": "Bearer test-api-key"}


# ============== Async Fixtures ==============

@pytest.fixture
def mock_async_browser():
    """Create a mock async browser for testing."""
    with patch("playwright.async_api") as mock:
        mock_browser = AsyncMock()
        mock_browser.close = AsyncMock()
        mock.new_page = AsyncMock()
        yield mock_browser


# ============== Pytest Hooks ==============

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "asyncio: marks tests as async")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Add integration marker to tests in integration directory
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


# ============== Test Utilities ==============

def create_mock_response(status_code: int = 200, json_data: dict = None):
    """Create a mock HTTP response."""
    response = Mock()
    response.status_code = status_code
    response.json = Mock(return_value=json_data or {})
    response.text = ""
    return response


def assert_valid_response(response: dict, required_fields: list):
    """Assert that a response has all required fields."""
    for field in required_fields:
        assert field in response, f"Missing required field: {field}"


__all__ = [
    "mock_config",
    "mock_vision_engine",
    "mock_ocr_engine",
    "sample_image",
    "mock_screen_capture",
    "mock_trigger_action",
    "mock_keyboard_action",
    "mock_mouse_action",
    "temp_config_file",
    "api_client",
    "auth_headers",
    "mock_async_browser",
    "create_mock_response",
    "assert_valid_response",
]
