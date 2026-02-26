"""
Cross-Platform Automation Backend

Abstraction layer for keyboard/mouse automation
supporting Linux (xdotool), Windows (pywin32), and Mac (pyobjc).
"""

import os
import time
import subprocess
import platform
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from ..core.logger import get_logger

logger = get_logger("automation")


class Platform(Enum):
    """Supported platforms"""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    UNKNOWN = "unknown"


@dataclass
class Point:
    """Screen point"""
    x: int
    y: int


@dataclass
class Key:
    """Keyboard key"""
    key: str
    modifiers: List[str] = None

    def __post_init__(self):
        if self.modifiers is None:
            self.modifiers = []


class AutomationBackend(ABC):
    """Abstract base class for automation backends"""

    @abstractmethod
    def press_key(self, key: str, modifiers: List[str] = None) -> bool:
        """Press a key"""
        pass

    @abstractmethod
    def type_text(self, text: str) -> bool:
        """Type text"""
        pass

    @abstractmethod
    def click(self, x: int, y: int, button: str = "left") -> bool:
        """Click at position"""
        pass

    @abstractmethod
    def double_click(self, x: int, y: int, button: str = "left") -> bool:
        """Double click at position"""
        pass

    @abstractmethod
    def right_click(self, x: int, y: int) -> bool:
        """Right click at position"""
        pass

    @abstractmethod
    def move_to(self, x: int, y: int) -> bool:
        """Move mouse to position"""
        pass

    @abstractmethod
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> bool:
        """Drag from start to end"""
        pass

    @abstractmethod
    def scroll(self, x: int, y: int, delta: int) -> bool:
        """Scroll at position"""
        pass

    @abstractmethod
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen resolution"""
        pass

    @abstractmethod
    def get_cursor_position(self) -> Point:
        """Get current cursor position"""
        pass


class LinuxAutomationBackend(AutomationBackend):
    """Linux automation using xdotool"""

    def __init__(self):
        self._verify_xdotool()

    def _verify_xdotool(self):
        """Verify xdotool is available"""
        try:
            subprocess.run(["xdotool", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("xdotool not found - automation may not work")

    def press_key(self, key: str, modifiers: List[str] = None) -> bool:
        """Press a key using xdotool"""
        try:
            cmd = ["xdotool"]

            # Add modifiers
            if modifiers:
                for mod in modifiers:
                    cmd.extend(["keydown", mod])

            cmd.extend(["key", key])

            if modifiers:
                for mod in reversed(modifiers):
                    cmd.extend(["keyup", mod])

            subprocess.run(cmd, check=True, capture_output=True)
            return True

        except Exception as e:
            logger.error(f"Failed to press key: {e}")
            return False

    def type_text(self, text: str) -> bool:
        """Type text using xdotool"""
        try:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", text],
                check=True,
                capture_output=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return False

    def click(self, x: int, y: int, button: str = "left") -> bool:
        """Click at position"""
        try:
            button_map = {"left": 1, "middle": 2, "right": 3}
            btn = button_map.get(button.lower(), 1)

            subprocess.run(
                ["xdotool", "mousemove", str(x), str(y), "click", str(btn)],
                check=True,
                capture_output=True
            )
            return True

        except Exception as e:
            logger.error(f"Failed to click: {e}")
            return False

    def double_click(self, x: int, y: int, button: str = "left") -> bool:
        """Double click at position"""
        try:
            button_map = {"left": 1, "middle": 2, "right": 3}
            btn = button_map.get(button.lower(), 1)

            subprocess.run(
                ["xdotool", "mousemove", str(x), str(y), "click", "--repeat", "2", str(btn)],
                check=True,
                capture_output=True
            )
            return True

        except Exception as e:
            logger.error(f"Failed to double click: {e}")
            return False

    def right_click(self, x: int, y: int) -> bool:
        """Right click at position"""
        return self.click(x, y, "right")

    def move_to(self, x: int, y: int) -> bool:
        """Move mouse to position"""
        try:
            subprocess.run(
                ["xdotool", "mousemove", str(x), str(y)],
                check=True,
                capture_output=True
            )
            return True

        except Exception as e:
            logger.error(f"Failed to move mouse: {e}")
            return False

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> bool:
        """Drag from start to end"""
        try:
            # Mouse down
            subprocess.run(
                ["xdotool", "mousemove", str(start_x), str(start_y), "mousedown", "1"],
                check=True,
                capture_output=True
            )

            # Move to end
            time.sleep(0.1)
            subprocess.run(
                ["xdotool", "mousemove", str(end_x), str(end_y)],
                check=True,
                capture_output=True
            )

            # Mouse up
            time.sleep(0.1)
            subprocess.run(
                ["xdotool", "mouseup", "1"],
                check=True,
                capture_output=True
            )

            return True

        except Exception as e:
            logger.error(f"Failed to drag: {e}")
            return False

    def scroll(self, x: int, y: int, delta: int) -> bool:
        """Scroll at position"""
        try:
            # 4 and 5 are scroll up/down
            button = 4 if delta > 0 else 5
            clicks = abs(delta)

            for _ in range(clicks):
                subprocess.run(
                    ["xdotool", "mousemove", str(x), str(y), "click", str(button)],
                    check=True,
                    capture_output=True
                )

            return True

        except Exception as e:
            logger.error(f"Failed to scroll: {e}")
            return False

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen resolution"""
        try:
            # Get primary monitor size
            result = subprocess.run(
                ["xdotool", "getdisplaygeometry"],
                capture_output=True,
                text=True,
                check=True
            )
            width, height = map(int, result.stdout.strip().split())
            return width, height

        except Exception:
            return 1920, 1080  # Default

    def get_cursor_position(self) -> Point:
        """Get current cursor position"""
        try:
            result = subprocess.run(
                ["xdotool", "getmouselocation"],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output: x:1234 y:5678 screen:0
            output = result.stdout.strip()
            parts = output.split()
            x = int(parts[0].split(":")[1])
            y = int(parts[1].split(":")[1])

            return Point(x, y)

        except Exception:
            return Point(0, 0)


class WindowsAutomationBackend(AutomationBackend):
    """Windows automation using pywin32"""

    def __init__(self):
        try:
            import win32api
            import win32con
            import win32gui
            import win32com.client
            self._win32api = win32api
            self._win32con = win32con
            self._win32gui = win32gui
            self._shell = win32com.client.Dispatch("WScript.Shell")
            logger.info("Windows automation backend initialized")
        except ImportError:
            logger.error("pywin32 not installed - Windows automation not available")
            self._win32api = None

    def press_key(self, key: str, modifiers: List[str] = None) -> bool:
        """Press a key"""
        if not self._win32api:
            return False

        try:
            # Map modifier keys
            mod_map = {
                "ctrl": win32con.VK_CONTROL,
                "alt": win32con.VK_MENU,
                "shift": win32con.VK_SHIFT,
                "cmd": win32con.VK_LWIN,
            }

            # Press modifiers
            if modifiers:
                for mod in modifiers:
                    if mod.lower() in mod_map:
                        self._win32api.keybd_event(mod_map[mod.lower()], 0, 0, 0)

            # Press key
            self._win32api.keybd_event(self._vk_from_key(key), 0, 0, 0)
            time.sleep(0.05)
            self._win32api.keybd_event(self._vk_from_key(key), 0, win32con.KEYEVENTF_KEYUP, 0)

            # Release modifiers
            if modifiers:
                for mod in reversed(modifiers):
                    if mod.lower() in mod_map:
                        self._win32api.keybd_event(mod_map[mod.lower()], 0, win32con.KEYEVENTF_KEYUP, 0)

            return True

        except Exception as e:
            logger.error(f"Failed to press key: {e}")
            return False

    def type_text(self, text: str) -> bool:
        """Type text"""
        if not self._shell:
            return False

        try:
            self._shell.SendKeys(text)
            return True

        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return False

    def click(self, x: int, y: int, button: str = "left") -> bool:
        """Click at position"""
        if not self._win32api:
            return False

        try:
            self._win32api.SetCursorPos((x, y))
            time.sleep(0.05)

            btn_down = win32con.MOUSEEVENTF_LEFTDOWN
            btn_up = win32con.MOUSEEVENTF_LEFTUP

            if button.lower() == "right":
                btn_down = win32con.MOUSEEVENTF_RIGHTDOWN
                btn_up = win32con.MOUSEEVENTF_RIGHTUP

            self._win32api.mouse_event(btn_down, 0, 0, 0, 0)
            time.sleep(0.05)
            self._win32api.mouse_event(btn_up, 0, 0, 0, 0)

            return True

        except Exception as e:
            logger.error(f"Failed to click: {e}")
            return False

    def double_click(self, x: int, y: int, button: str = "left") -> bool:
        """Double click"""
        self.click(x, y, button)
        time.sleep(0.1)
        return self.click(x, y, button)

    def right_click(self, x: int, y: int) -> bool:
        """Right click"""
        return self.click(x, y, "right")

    def move_to(self, x: int, y: int) -> bool:
        """Move mouse"""
        if not self._win32api:
            return False

        try:
            self._win32api.SetCursorPos((x, y))
            return True

        except Exception as e:
            logger.error(f"Failed to move mouse: {e}")
            return False

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> bool:
        """Drag from start to end"""
        if not self._win32api:
            return False

        try:
            self._win32api.SetCursorPos((start_x, start_y))
            time.sleep(0.1)

            # Mouse down
            self._win32api.mouse_event(0x0001, 0, 0, 0, 0)  # LEFTDOWN
            time.sleep(0.1)

            # Move
            self._win32api.SetCursorPos((end_x, end_y))
            time.sleep(0.1)

            # Mouse up
            self._win32api.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP

            return True

        except Exception as e:
            logger.error(f"Failed to drag: {e}")
            return False

    def scroll(self, x: int, y: int, delta: int) -> bool:
        """Scroll at position"""
        if not self._win32api:
            return False

        try:
            self._win32api.SetCursorPos((x, y))
            time.sleep(0.05)

            # Scroll event
            self._win32api.mouse_event(0x0800, 0, 0, delta * 120, 0)

            return True

        except Exception as e:
            logger.error(f"Failed to scroll: {e}")
            return False

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen resolution"""
        if not self._win32gui:
            return 1920, 1080

        try:
            width = self._win32api.GetSystemMetrics(0)  # SM_CXSCREEN
            height = self._win32api.GetSystemMetrics(1)  # SM_CYSCREEN
            return width, height

        except Exception:
            return 1920, 1080

    def get_cursor_position(self) -> Point:
        """Get cursor position"""
        if not self._win32api:
            return Point(0, 0)

        try:
            x, y = self._win32api.GetCursorPos()
            return Point(x, y)

        except Exception:
            return Point(0, 0)

    def _vk_from_key(self, key: str) -> int:
        """Convert key to virtual key code"""
        vk_map = {
            "a": 0x41, "b": 0x42, "c": 0x43, "d": 0x44, "e": 0x45,
            "f": 0x46, "g": 0x47, "h": 0x48, "i": 0x49, "j": 0x4a,
            "k": 0x4b, "l": 0x4c, "m": 0x4d, "n": 0x4e, "o": 0x4f,
            "p": 0x50, "q": 0x51, "r": 0x52, "s": 0x53, "t": 0x54,
            "u": 0x55, "v": 0x56, "w": 0x57, "x": 0x58, "y": 0x59, "z": 0x5a,
            "enter": 0x0d, "return": 0x0d,
            "tab": 0x09,
            "escape": 0x1b, "esc": 0x1b,
            "space": 0x20,
            "backspace": 0x08,
            "delete": 0x2e,
            "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
            "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
            "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
            "f9": 0x78, "f10": 0x79, "f11": 0x7a, "f12": 0x7b,
        }

        key = key.lower()
        if key in vk_map:
            return vk_map[key]

        # Try ord() for single characters
        if len(key) == 1:
            return ord(key.upper())

        return 0


class MacOSAutomationBackend(AutomationBackend):
    """macOS automation using pyobjc and osascript"""

    def __init__(self):
        logger.info("macOS automation backend initialized")

    def press_key(self, key: str, modifiers: List[str] = None) -> bool:
        """Press a key using osascript"""
        try:
            mod_map = {
                "ctrl": "control",
                "alt": "option",
                "shift": "shift",
                "cmd": "command"
            }

            mods = ""
            if modifiers:
                mods = " with " + " ".join(mod_map.get(m.lower(), m) for m in modifiers)

            # Use AppleScript
            script = f'tell application "System Events" to keystroke "{key}"{mods}'
            subprocess.run(["osascript", "-e", script], check=True)

            return True

        except Exception as e:
            logger.error(f"Failed to press key: {e}")
            return False

    def type_text(self, text: str) -> bool:
        """Type text using osascript"""
        try:
            # Escape quotes
            text = text.replace('"', '\\"')
            script = f'tell application "System Events" to keystroke "{text}"'
            subprocess.run(["osascript", "-e", script], check=True)
            return True

        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return False

    def click(self, x: int, y: int, button: str = "left") -> bool:
        """Click at position"""
        try:
            btn = "button 1"
            if button.lower() == "right":
                btn = "button 2"

            script = f'''
            tell application "System Events"
                set mousePos to current application's (do shell script "echo #{x},#{y}")
                click at mousePos
            end tell
            '''

            subprocess.run(["osascript", "-e", script], check=True)
            return True

        except Exception as e:
            logger.error(f"Failed to click: {e}")
            return False

    def double_click(self, x: int, y: int, button: str = "left") -> bool:
        """Double click"""
        try:
            script = f'''
            tell application "System Events"
                click at {{ {x}, {y} }}
                delay 0.05
                click at {{ {x}, {y} }}
            end tell
            '''
            subprocess.run(["osascript", "-e", script], check=True)
            return True

        except Exception as e:
            logger.error(f"Failed to double click: {e}")
            return False

    def right_click(self, x: int, y: int) -> bool:
        """Right click"""
        try:
            script = f'''
            tell application "System Events"
                click at {{ {x}, {y} }} with command
            end tell
            '''
            subprocess.run(["osascript", "-e", script], check=True)
            return True

        except Exception as e:
            logger.error(f"Failed to right click: {e}")
            return False

    def move_to(self, x: int, y: int) -> bool:
        """Move mouse"""
        try:
            script = f'''
            tell application "System Events"
                set mouse position {{ {x}, {y} }}
            end tell
            '''
            subprocess.run(["osascript", "-e", script], check=True)
            return True

        except Exception as e:
            logger.error(f"Failed to move mouse: {e}")
            return False

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> bool:
        """Drag from start to end"""
        try:
            script = f'''
            tell application "System Events"
                set mouse position {{ {start_x}, {start_y} }}
                delay 0.1
                click drag to {{ {end_x}, {end_y} }}
            end tell
            '''
            subprocess.run(["osascript", "-e", script], check=True)
            return True

        except Exception as e:
            logger.error(f"Failed to drag: {e}")
            return False

    def scroll(self, x: int, y: int, delta: int) -> bool:
        """Scroll at position"""
        try:
            script = f'''
            tell application "System Events"
                set mouse position {{ {x}, {y} }}
                scroll {{ 0, {delta * -120} }}
            end tell
            '''
            subprocess.run(["osascript", "-e", script], check=True)
            return True

        except Exception as e:
            logger.error(f"Failed to scroll: {e}")
            return False

    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen resolution"""
        try:
            script = '''
            tell application "System Events"
                set screenSize to size of window 1 of process "Finder"
                return screenSize
            end tell
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=True
            )
            width, height = map(int, result.stdout.strip().split(","))
            return width, height

        except Exception:
            return 1920, 1080

    def get_cursor_position(self) -> Point:
        """Get cursor position"""
        try:
            script = '''
            tell application "System Events"
                set mousePos to do shell script "echo $(GetCLocation | cut -d',' -f1),$(GetCLocation | cut -d',' -f2)"
                return mousePos
            end tell
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                x, y = map(int, result.stdout.strip().split(","))
                return Point(x, y)

        except Exception:
            pass

        return Point(0, 0)


def get_platform() -> Platform:
    """Detect current platform"""
    system = platform.system().lower()

    if system == "linux":
        return Platform.LINUX
    elif system == "windows":
        return Platform.WINDOWS
    elif system == "darwin":
        return Platform.MACOS
    else:
        return Platform.UNKNOWN


def create_automation_backend(platform: Platform = None) -> AutomationBackend:
    """Create automation backend for platform"""
    if platform is None:
        platform = get_platform()

    if platform == Platform.LINUX:
        return LinuxAutomationBackend()
    elif platform == Platform.WINDOWS:
        return WindowsAutomationBackend()
    elif platform == Platform.MACOS:
        return MacOSAutomationBackend()
    else:
        raise ValueError(f"Unsupported platform: {platform}")


# Global backend instance
_backend: Optional[AutomationBackend] = None


def get_automation_backend() -> AutomationBackend:
    """Get global automation backend"""
    global _backend

    if _backend is None:
        _backend = create_automation_backend()

    return _backend


# Convenience functions
def press(key: str, modifiers: List[str] = None) -> bool:
    """Press a key"""
    return get_automation_backend().press_key(key, modifiers)


def type_text(text: str) -> bool:
    """Type text"""
    return get_automation_backend().type_text(text)


def click(x: int, y: int, button: str = "left") -> bool:
    """Click at position"""
    return get_automation_backend().click(x, y, button)


def move_to(x: int, y: int) -> bool:
    """Move mouse"""
    return get_automation_backend().move_to(x, y)


def get_screen_size() -> Tuple[int, int]:
    """Get screen size"""
    return get_automation_backend().get_screen_size()


def get_cursor_position() -> Point:
    """Get cursor position"""
    return get_automation_backend().get_cursor_position()


__all__ = [
    "Platform",
    "AutomationBackend",
    "LinuxAutomationBackend",
    "WindowsAutomationBackend",
    "MacOSAutomationBackend",
    "get_platform",
    "create_automation_backend",
    "get_automation_backend",
    "press",
    "type_text",
    "click",
    "move_to",
    "get_screen_size",
    "get_cursor_position",
    "Point",
    "Key",
]
