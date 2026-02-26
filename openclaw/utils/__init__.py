"""Utility modules"""

import os
import time
import subprocess
from typing import List, Dict, Optional, Tuple

import mss


class MultiMonitor:
    """Multi-monitor support utilities"""

    @staticmethod
    def get_monitors() -> List[Dict]:
        """Get all connected monitors with positions and sizes"""
        monitors = []

        try:
            # Try xrandr first
            result = subprocess.run(
                ["xrandr"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.split("\n")
                for line in lines:
                    if " connected" in line:
                        parts = line.split()
                        name = parts[0]

                        # Parse resolution and position
                        if "(" in line:
                            res_part = line.split("(")[0].strip().split()[-1]
                            if "x" in res_part:
                                w, h = map(int, res_part.split("x"))
                                if "+" in line:
                                    pos = line.split("+")[1].split()
                                    x = int(pos[0])
                                    y = int(pos[1]) if len(pos) > 1 else 0
                                else:
                                    x, y = 0, 0

                                monitors.append({
                                    "name": name,
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h
                                })

        except Exception:
            pass

        # Fallback to mss
        if not monitors:
            try:
                with mss.mss() as sct:
                    for i, mon in enumerate(sct.monitors):
                        if i > 0:
                            monitors.append({
                                "name": f"monitor-{i}",
                                "x": mon["left"],
                                "y": mon["top"],
                                "width": mon["width"],
                                "height": mon["height"]
                            })
            except:
                pass

        # Default primary
        if not monitors:
            monitors.append({
                "name": "primary",
                "x": 0,
                "y": 0,
                "width": 1920,
                "height": 1080
            })

        return monitors

    @classmethod
    def get_primary(cls) -> Dict:
        """Get primary monitor"""
        monitors = cls.get_monitors()
        return monitors[0] if monitors else {"name": "primary", "x": 0, "y": 0, "width": 1920, "height": 1080}


class RegionSelector:
    """Interactive screen region selector"""

    @staticmethod
    def select_region() -> Optional[Tuple[int, int, int, int]]:
        """Open interactive overlay to select screen region"""
        try:
            import tkinter as tk

            selection = {"start": None, "end": None, "done": False}

            root = tk.Tk()
            root.attributes("-fullscreen", True)
            root.attributes("-alpha", 0.3)
            root.configure(bg="black")
            root.cursor = "crosshair"

            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()

            canvas = tk.Canvas(
                root,
                width=screen_width,
                height=screen_height,
                bg="black",
                highlightthickness=0
            )
            canvas.pack()

            rect_id = canvas.create_rectangle(
                0, 0, 0, 0,
                outline="red",
                width=3,
                fill=""
            )

            label = canvas.create_text(
                20, 20,
                text="Click and drag to select region. Press ESC to cancel.",
                fill="white",
                font=("Arial", 16),
                anchor="nw"
            )

            def on_mouse_down(event):
                selection["start"] = (event.x, event.y)
                canvas.coords(rect_id, event.x, event.y, event.x, event.y)

            def on_mouse_move(event):
                if selection["start"]:
                    canvas.coords(
                        rect_id,
                        selection["start"][0],
                        selection["start"][1],
                        event.x,
                        event.y
                    )

            def on_mouse_up(event):
                selection["end"] = (event.x, event.y)
                selection["done"] = True
                root.quit()

            def on_key(event):
                if event.keysym == "Escape":
                    selection["done"] = True
                    root.quit()

            canvas.bind("<Button-1>", on_mouse_down)
            canvas.bind("<B1-Motion>", on_mouse_move)
            canvas.bind("<ButtonRelease-1>", on_mouse_up)
            root.bind("<Key>", on_key)

            root.mainloop()
            root.destroy()

            if (selection["start"] and selection["end"] and
                    selection["start"] != selection["end"]):
                x1 = min(selection["start"][0], selection["end"][0])
                y1 = min(selection["start"][1], selection["end"][1])
                x2 = max(selection["start"][0], selection["end"][0])
                y2 = max(selection["start"][1], selection["end"][1])
                width = x2 - x1
                height = y2 - y1

                print(f"[RegionSelector] Selected: x={x1}, y={y1}, w={width}, h={height}")
                return (x1, y1, width, height)

            print("[RegionSelector] Selection cancelled")
            return None

        except Exception as e:
            print(f"[RegionSelector] Error: {e}")
            return None


class ScreenRecorder:
    """Screen recording utilities"""

    @staticmethod
    def capture_and_save(
        region: Optional[Tuple[int, int, int, int]] = None,
        output_dir: str = "/tmp/openclaw_records"
    ) -> str:
        """Capture screenshot and save to file"""
        # Import here to avoid circular imports
        import cv2
        from openclaw.core.vision import ScreenCapture

        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        img = ScreenCapture.capture_region(region)
        cv2.imwrite(filepath, img)

        return filepath


# Export utilities
__all__ = [
    "MultiMonitor",
    "RegionSelector",
    "ScreenRecorder",
    "ConfigEncryption",
    "EncryptedConfig",
    "encrypt_sensitive_fields",
    "decrypt_sensitive_fields",
]

# Security module
from openclaw.utils.security import (
    ConfigEncryption,
    EncryptedConfig,
    encrypt_sensitive_fields,
    decrypt_sensitive_fields
)
