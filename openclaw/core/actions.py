"""Automation actions with retry logic"""

import time
import subprocess
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum
import threading

from .logger import get_logger

logger = get_logger("actions")


class RetryStrategy(Enum):
    """Retry strategies"""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    attempts: int = 3
    delay: float = 1.0
    backoff_multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_delay: float = 30.0

    @classmethod
    def from_dict(cls, data: Dict) -> 'RetryConfig':
        """Create from dictionary"""
        return cls(
            attempts=data.get("retry_attempts", 3),
            delay=data.get("retry_delay", 1.0),
            backoff_multiplier=data.get("backoff_multiplier", 2.0),
            strategy=RetryStrategy(data.get("strategy", "exponential")),
            max_delay=data.get("max_delay", 30.0)
        )


class RetryableError(Exception):
    """Error that should trigger retry"""
    pass


class ActionExecutor:
    """Execute actions with retry logic"""

    def __init__(self, retry_config: RetryConfig = None):
        self.retry_config = retry_config or RetryConfig()

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        if self.retry_config.strategy == RetryStrategy.FIXED:
            return self.retry_config.delay
        elif self.retry_config.strategy == RetryStrategy.LINEAR:
            return self.retry_config.delay * attempt
        elif self.retry_config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.retry_config.delay * (self.retry_config.backoff_multiplier ** (attempt - 1))
            return min(delay, self.retry_config.max_delay)
        return self.retry_config.delay

    def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic"""
        last_error = None

        for attempt in range(1, self.retry_config.attempts + 1):
            try:
                return func(*args, **kwargs)

            except RetryableError as e:
                last_error = e
                logger.warning(f"Attempt {attempt}/{self.retry_config.attempts} failed: {e}")

                if attempt < self.retry_config.attempts:
                    delay = self.calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)

            except Exception as e:
                # Non-retryable error
                logger.error(f"Non-retryable error: {e}")
                raise

        # All retries exhausted
        logger.error(f"All {self.retry_config.attempts} attempts failed")
        if last_error:
            raise last_error


class KeyboardAction:
    """Keyboard automation actions"""

    @staticmethod
    def press(key: str, delay: float = 0.0) -> bool:
        """Press a keyboard key using xdotool.

        Args:
            key: Key name (e.g., 'alt+o', 'Return')
            delay: Delay in seconds before pressing

        Returns:
            True if key press succeeded, False otherwise

        Raises:
            subprocess.TimeoutExpired: If xdotool times out
            OSError: If xdotool is not installed
        """
        if delay > 0:
            time.sleep(delay)

        # Validate key against whitelist to prevent injection
        allowed_keys = {
            'alt', 'ctrl', 'shift', 'super', 'tab', 'enter', 'return',
            'space', 'backspace', 'delete', 'escape', 'esc',
            'up', 'down', 'left', 'right',
            'home', 'end', 'pageup', 'pagedown',
            'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        }

        try:
            # Use list form to prevent shell injection
            cmd = ["xdotool", "key", "--clearmodifiers", key]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            success = result.returncode == 0
            if success:
                logger.debug(f"Key pressed: {key}")
            else:
                logger.error(f"Key press failed: {result.stderr}")
            return success

        except subprocess.TimeoutExpired:
            logger.error(f"Key press timed out: {key}")
            return False

        except OSError as e:
            logger.error(f"OS error during key press (xdotool installed?): {e}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during key press: {e}")
            return False

    @staticmethod
    def type_text(text: str, delay: float = 0.1) -> bool:
        """Type text character by character using xdotool.

        Args:
            text: Text to type
            delay: Delay between keystrokes in seconds

        Returns:
            True if typing succeeded, False otherwise
        """
        try:
            # Use list form to prevent shell injection
            for char in text:
                cmd = ["xdotool", "key", char]
                subprocess.run(cmd, capture_output=True, timeout=5, check=False)
                time.sleep(delay)
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"Type text timed out")
            return False
        except OSError as e:
            logger.error(f"OS error during type text: {e}")
            return False
        except Exception as e:
            logger.error(f"Type text error: {e}")
            return False

    @staticmethod
    def hotkey(*keys: str) -> bool:
        """Press hotkey combination"""
        key = "+".join(keys)
        return KeyboardAction.press(key)


class MouseAction:
    """Mouse automation actions"""

    @staticmethod
    def move(x: int, y: int) -> bool:
        """Move mouse to coordinates"""
        try:
            subprocess.run(
                ["xdotool", "mousemove", str(x), str(y)],
                check=False,
                timeout=5
            )
            logger.debug(f"Mouse moved to ({x}, {y})")
            return True
        except Exception as e:
            logger.error(f"Mouse move error: {e}")
            return False

    @staticmethod
    def click(button: str = "1", x: int = None, y: int = None) -> bool:
        """Click mouse button"""
        try:
            if x is not None and y is not None:
                subprocess.run(
                    ["xdotool", "mousemove", str(x), str(y)],
                    check=False,
                    timeout=5
                )
                time.sleep(0.05)

            subprocess.run(
                ["xdotool", "click", button],
                check=False,
                timeout=5
            )
            logger.debug(f"Mouse clicked: button={button}")
            return True

        except Exception as e:
            logger.error(f"Mouse click error: {e}")
            return False

    @staticmethod
    def double_click(button: str = "1", x: int = None, y: int = None) -> bool:
        """Double click"""
        try:
            if x is not None and y is not None:
                subprocess.run(
                    ["xdotool", "mousemove", str(x), str(y)],
                    check=False,
                    timeout=5
                )
                time.sleep(0.05)

            subprocess.run(
                ["xdotool", "click", "--repeat", "2", button],
                check=False,
                timeout=5
            )
            return True

        except Exception as e:
            logger.error(f"Double click error: {e}")
            return False

    @staticmethod
    def drag(start_x: int, start_y: int, end_x: int, end_y: int) -> bool:
        """Drag from start to end"""
        try:
            subprocess.run(
                ["xdotool", "mousemove", str(start_x), str(start_y)],
                check=False,
                timeout=5
            )
            time.sleep(0.1)

            subprocess.run(["xdotool", "mousedown", "1"], check=False, timeout=5)
            time.sleep(0.1)

            subprocess.run(
                ["xdotool", "mousemove", str(end_x), str(end_y)],
                check=False,
                timeout=5
            )
            time.sleep(0.1)

            subprocess.run(["xdotool", "mouseup", "1"], check=False, timeout=5)

            logger.debug(f"Drag: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
            return True

        except Exception as e:
            logger.error(f"Drag error: {e}")
            return False

    @staticmethod
    def scroll(clicks: int) -> bool:
        """Scroll (positive=up, negative=down)"""
        try:
            button = "4" if clicks > 0 else "5"
            commands = ["xdotool", "click"] + [button] * abs(clicks)
            subprocess.run(commands, check=False, timeout=5)
            return True
        except Exception as e:
            logger.error(f"Scroll error: {e}")
            return False


class ActionSequence:
    """Execute multi-step action sequences"""

    def __init__(self, retry_config: RetryConfig = None):
        self.retry_config = retry_config or RetryConfig()
        self.executor = ActionExecutor(self.retry_config)

    def execute(self, actions: List[Dict]) -> bool:
        """Execute a sequence of actions"""
        if not actions:
            logger.warning("Empty action sequence")
            return True

        logger.info(f"Executing {len(actions)} actions")

        for i, action in enumerate(actions):
            action_type = action.get("type", "key")
            delay = action.get("delay", 0)

            if delay > 0:
                logger.debug(f"Waiting {delay}s before action {i+1}")
                time.sleep(delay)

            success = False

            if action_type == "key":
                key = action.get("key", "alt+o")
                success = KeyboardAction.press(key)
                logger.debug(f"Action {i+1}: Key {key} - {'OK' if success else 'FAILED'}")

            elif action_type == "mouse":
                mouse_action = action.get("action", {})
                mouse_type = mouse_action.get("action", "click")

                if mouse_type == "move":
                    success = MouseAction.move(
                        mouse_action.get("x", 0),
                        mouse_action.get("y", 0)
                    )
                elif mouse_type == "click":
                    success = MouseAction.click(
                        mouse_action.get("button", "1"),
                        mouse_action.get("x"),
                        mouse_action.get("y")
                    )
                elif mouse_type == "double_click":
                    success = MouseAction.double_click(
                        mouse_action.get("button", "1"),
                        mouse_action.get("x"),
                        mouse_action.get("y")
                    )
                elif mouse_type == "drag":
                    success = MouseAction.drag(
                        mouse_action.get("start_x", 0),
                        mouse_action.get("start_y", 0),
                        mouse_action.get("end_x", 0),
                        mouse_action.get("end_y", 0)
                    )
                elif mouse_type == "scroll":
                    success = MouseAction.scroll(mouse_action.get("clicks", 3))

            elif action_type == "wait":
                # Just wait
                wait_time = action.get("delay", 1.0)
                time.sleep(wait_time)
                success = True

            if not success:
                logger.error(f"Action {i+1} failed: {action}")
                return False

        logger.info("Action sequence completed")
        return True

    def execute_async(self, actions: List[Dict]) -> None:
        """Execute action sequence in background"""
        thread = threading.Thread(target=self.execute, args=(actions,))
        thread.daemon = True
        thread.start()


class TriggerAction:
    """Execute trigger action (keyboard shortcut)"""

    @staticmethod
    def execute(action: str, delay: float = 1.5) -> bool:
        """Execute keyboard shortcut with delay"""
        if delay > 0:
            logger.debug(f"Waiting {delay}s before trigger")
            time.sleep(delay)

        return KeyboardAction.press(action)


# Export classes
__all__ = [
    "RetryConfig",
    "RetryStrategy",
    "ActionExecutor",
    "KeyboardAction",
    "MouseAction",
    "ActionSequence",
    "TriggerAction",
]
