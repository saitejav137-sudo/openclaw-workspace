"""
Screen Recording for OpenClaw

Record screen captures on trigger events.
Supports GIF, MP4, and image sequences.
"""

import os
import time
import threading
import subprocess
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from .logger import get_logger
from .vision import ScreenCapture

logger = get_logger("recorder")


class RecordingFormat(Enum):
    """Recording format"""
    GIF = "gif"
    MP4 = "mp4"
    WEBM = "webm"
    IMAGES = "images"


class RecordingQuality(Enum):
    """Recording quality"""
    LOW = 10
    MEDIUM = 20
    HIGH = 30
    ULTRA = 40


@dataclass
class RecordingConfig:
    """Recording configuration"""
    format: RecordingFormat = RecordingFormat.MP4
    quality: RecordingQuality = RecordingQuality.MEDIUM
    fps: int = 10
    duration: float = 10.0  # seconds
    max_duration: float = 60.0  # max seconds
    output_dir: str = "~/.openclaw/recordings"
    prefix: str = "trigger"
    codec: str = "mp4v"


@dataclass
class RecordingSession:
    """Recording session"""
    id: str
    output_path: str
    format: RecordingFormat
    fps: int
    frames: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "recording"


class ScreenRecorder:
    """
    Screen recorder for capturing trigger events.

    Records screen before and after trigger events.
    """

    def __init__(self, config: RecordingConfig = None):
        self.config = config or RecordingConfig()
        self._output_dir = os.path.expanduser(self.config.output_dir)
        self._current_session: Optional[RecordingSession] = None
        self._frames: List[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()
        self._pre_buffer: List[np.ndarray] = []
        self._pre_buffer_size = 30  # Keep last 30 frames

        # Create output directory
        os.makedirs(self._output_dir, exist_ok=True)

    def start_recording(self, session_id: str = None) -> RecordingSession:
        """Start a new recording session"""
        if self._recording:
            logger.warning("Already recording")
            return self._current_session

        if session_id is None:
            session_id = f"{self.config.prefix}_{int(time.time())}"

        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_{timestamp}.{self.config.format.value}"
        output_path = os.path.join(self._output_dir, filename)

        self._current_session = RecordingSession(
            id=session_id,
            output_path=output_path,
            format=self.config.format,
            fps=self.config.fps
        )

        self._frames = []
        self._pre_buffer = []
        self._recording = True

        logger.info(f"Started recording: {output_path}")
        return self._current_session

    def stop_recording(self) -> Optional[RecordingSession]:
        """Stop recording and save"""
        if not self._recording:
            return None

        self._recording = False
        session = self._current_session

        if session:
            session.end_time = time.time()
            session.frames = len(self._frames)

            # Save recording
            self._save_recording(session)

            logger.info(f"Stopped recording: {session.output_path} ({session.frames} frames)")

        self._current_session = None
        return session

    def capture_frame(self, region: tuple = None):
        """Capture a frame"""
        if not self._recording:
            return

        try:
            # Capture screen
            if region:
                frame = ScreenCapture.capture_region(region)
            else:
                frame = ScreenCapture.capture_full()

            with self._lock:
                # Add to buffer
                self._frames.append(frame)

                # Maintain pre-buffer
                if len(self._pre_buffer) >= self._pre_buffer_size:
                    self._pre_buffer.pop(0)
                self._pre_buffer.append(frame)

        except Exception as e:
            logger.error(f"Capture error: {e}")

    def record_trigger(self, duration: float = None) -> Optional[RecordingSession]:
        """Record for a specific duration"""
        if duration is None:
            duration = self.config.duration

        session = self.start_recording()

        # Record for duration
        interval = 1.0 / self.config.fps
        start = time.time()

        while time.time() - start < duration:
            self.capture_frame()
            time.sleep(interval)

        return self.stop_recording()

    def record_with_buffer(self, pre_duration: float = 3.0, post_duration: float = 5.0) -> Optional[RecordingSession]:
        """Record including pre-trigger buffer"""
        session = self.start_recording()

        # Add pre-buffer frames
        with self._lock:
            self._frames.extend(self._pre_buffer)

        # Continue recording
        interval = 1.0 / self.config.fps
        start = time.time()
        post_frames = int(post_duration * self.config.fps)

        for _ in range(post_frames):
            self.capture_frame()
            time.sleep(interval)

        return self.stop_recording()

    def _save_recording(self, session: RecordingSession):
        """Save recording to file"""
        if not self._frames:
            logger.warning("No frames to save")
            return

        if session.format == RecordingFormat.IMAGES:
            self._save_images(session)
        else:
            self._save_video(session)

    def _save_video(self, session: RecordingSession):
        """Save as video"""
        if not self._frames:
            return

        # Get frame size
        frame = self._frames[0]
        height, width = frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        writer = cv2.VideoWriter(
            session.output_path,
            fourcc,
            session.fps,
            (width, height)
        )

        if not writer.isOpened():
            logger.error("Failed to open video writer")
            return

        # Write frames
        for frame in self._frames:
            # Convert BGR to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.write(rgb_frame)

        writer.release()

        logger.info(f"Saved video: {session.output_path}")

    def _save_images(self, session: RecordingSession):
        """Save as image sequence"""
        output_dir = session.output_path.replace(".images", "")
        os.makedirs(output_dir, exist_ok=True)

        for i, frame in enumerate(self._frames):
            filename = os.path.join(output_dir, f"frame_{i:05d}.png")
            cv2.imwrite(filename, frame)

        logger.info(f"Saved {len(self._frames)} images to: {output_dir}")

    def get_recordings(self) -> List[Dict[str, Any]]:
        """Get list of recordings"""
        recordings = []

        if not os.path.exists(self._output_dir):
            return recordings

        for filename in os.listdir(self._output_dir):
            filepath = os.path.join(self._output_dir, filename)

            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                recordings.append({
                    "filename": filename,
                    "path": filepath,
                    "size": stat.st_size,
                    "created": stat.st_ctime
                })

        return sorted(recordings, key=lambda x: x["created"], reverse=True)

    def delete_recording(self, filename: str) -> bool:
        """Delete a recording"""
        filepath = os.path.join(self._output_dir, filename)

        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted: {filepath}")
            return True

        return False

    @property
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self._recording


# Global recorder
_recorder: Optional[ScreenRecorder] = None


def get_recorder(config: RecordingConfig = None) -> ScreenRecorder:
    """Get global screen recorder"""
    global _recorder

    if _recorder is None:
        _recorder = ScreenRecorder(config)

    return _recorder


def start_recording(session_id: str = None) -> RecordingSession:
    """Quick start recording"""
    return get_recorder().start_recording(session_id)


def stop_recording() -> Optional[RecordingSession]:
    """Quick stop recording"""
    return get_recorder().stop_recording()


def record_on_trigger(duration: float = 10.0) -> Optional[RecordingSession]:
    """Quick record on trigger"""
    return get_recorder().record_trigger(duration)


__all__ = [
    "RecordingFormat",
    "RecordingQuality",
    "RecordingConfig",
    "RecordingSession",
    "ScreenRecorder",
    "get_recorder",
    "start_recording",
    "stop_recording",
    "record_on_trigger",
]
