"""
Window Monitor for OpenClaw

Watch window titles for specific signals and trigger actions.
"""

import time
import threading
import logging
from typing import Optional, Callable, List, Dict, Any

logger = logging.getLogger("openclaw.window")


class WindowMonitor:
    """Monitor window titles for specific signals"""

    def __init__(
        self,
        trigger_signal: str = "TRIGGER_CLAW",
        poll_interval: float = 0.3,
        callback: Optional[Callable] = None
    ):
        self.trigger_signal = trigger_signal
        self.poll_interval = poll_interval
        self.callback = callback
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.last_title: str = ""
        self.last_trigger_time: float = 0
        self.debounce_seconds: float = 3.0

    def start(self):
        """Start monitoring windows"""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Window monitor started - watching for '{self.trigger_signal}'")

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Window monitor stopped")

    def set_callback(self, callback: Callable):
        """Set the callback function"""
        self.callback = callback

    def set_debounce(self, seconds: float):
        """Set debounce time"""
        self.debounce_seconds = seconds

    def _monitor_loop(self):
        """Main monitoring loop"""
        import subprocess

        while self.running:
            try:
                # Get active window info
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if result.returncode == 0:
                    current_title = result.stdout.strip()

                    # Check for trigger signal
                    if self.trigger_signal in current_title:
                        # Check if this is a new occurrence (debounce)
                        if current_title != self.last_title:
                            current_time = time.time()

                            if current_time - self.last_trigger_time > self.debounce_seconds:
                                logger.info(f">>> SIGNAL DETECTED: {current_title}")
                                self.last_trigger_time = current_time

                                # Execute callback
                                if self.callback:
                                    try:
                                        self.callback(current_title)
                                    except (TypeError, ValueError) as e:
                                        logger.error(f"Callback error (invalid arguments): {e}")
                                    except Exception as e:
                                        logger.error(f"Callback error: {e}")

                    self.last_title = current_title

            except subprocess.TimeoutExpired:
                logger.warning("Window query timeout")
            except FileNotFoundError:
                logger.error("xdotool not found - install xdotool")
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(self.poll_interval)

    def get_active_window(self) -> Optional[str]:
        """Get current active window title"""
        import subprocess

        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                return result.stdout.strip()

        except subprocess.TimeoutExpired:
            logger.warning("Window query timeout")
        except FileNotFoundError:
            logger.error("xdotool not found")
        except (OSError, ValueError) as e:
            logger.error(f"Get window error: {e}")

        return None

    def get_all_windows(self) -> List[Dict[str, str]]:
        """Get all visible windows"""
        import subprocess

        windows = []

        try:
            # Get all window IDs
            result = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--name", ".*"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                window_ids = result.stdout.strip().split('\n')

                for win_id in window_ids:
                    if not win_id.strip():
                        continue

                    # Get window name
                    name_result = subprocess.run(
                        ["xdotool", "getwindowname", win_id],
                        capture_output=True,
                        text=True,
                        timeout=1
                    )

                    if name_result.returncode == 0:
                        name = name_result.stdout.strip()
                        if name:
                            windows.append({
                                "id": win_id.strip(),
                                "title": name
                            })

        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Get windows error: {e}")
        except Exception as e:
            logger.error(f"Get windows error (unexpected): {e}")

        return windows


class WindowAction:
    """Execute actions based on window events"""

    @staticmethod
    def activate_window(window_id: str = None, window_name: str = None) -> bool:
        """Activate a window by ID or name"""
        import subprocess

        try:
            if window_id:
                subprocess.run(["xdotool", "windowactivate", "--sync", window_id], check=True)
            elif window_name:
                subprocess.run(["xdotool", "search", "--name", window_name, "windowactivate"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Activate window error (command failed): {e}")
            return False
        except (OSError, ValueError) as e:
            logger.error(f"Activate window error: {e}")
            return False
        except Exception as e:
            logger.error(f"Activate window error (unexpected): {e}")
            return False

    @staticmethod
    def close_window(window_id: str = None) -> bool:
        """Close a window"""
        import subprocess

        try:
            if window_id:
                subprocess.run(["xdotool", "windowclose", window_id], check=True)
                return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Close window error (command failed): {e}")
            return False
        except (OSError, ValueError) as e:
            logger.error(f"Close window error: {e}")
            return False
        except Exception as e:
            logger.error(f"Close window error (unexpected): {e}")
            return False

    @staticmethod
    def focus_window(window_name: str) -> bool:
        """Focus a window by name"""
        import subprocess

        try:
            subprocess.run(
                ["xdotool", "search", "--name", "--onlyvisible", window_name, "focus"],
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Focus window error (command failed): {e}")
            return False
        except (OSError, ValueError) as e:
            logger.error(f"Focus window error: {e}")
            return False
        except Exception as e:
            logger.error(f"Focus window error (unexpected): {e}")
            return False


# Global window monitor instance
_window_monitor: Optional[WindowMonitor] = None


def start_window_monitor(
    trigger_signal: str = "TRIGGER_CLAW",
    callback: Callable = None,
    poll_interval: float = 0.3
) -> WindowMonitor:
    """Start the global window monitor"""
    global _window_monitor

    _window_monitor = WindowMonitor(
        trigger_signal=trigger_signal,
        callback=callback,
        poll_interval=poll_interval
    )
    _window_monitor.start()

    return _window_monitor


def stop_window_monitor():
    """Stop the global window monitor"""
    global _window_monitor

    if _window_monitor:
        _window_monitor.stop()
        _window_monitor = None


def get_window_monitor() -> Optional[WindowMonitor]:
    """Get the global window monitor instance"""
    return _window_monitor


# Export
__all__ = [
    "WindowMonitor",
    "WindowAction",
    "start_window_monitor",
    "stop_window_monitor",
    "get_window_monitor",
]
