"""
Real-time Streaming Video Analysis for OpenClaw

Process video streams in real-time with AI-powered analysis.
"""

import time
import asyncio
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import cv2

from .logger import get_logger
from .vision import ScreenCapture

logger = get_logger("streaming")


class StreamSource(Enum):
    """Video stream sources"""
    SCREEN = "screen"
    WEBCAM = "webcam"
    RTSP = "rtsp"
    HTTP = "http"
    FILE = "file"


@dataclass
class StreamConfig:
    """Streaming configuration"""
    source: StreamSource = StreamSource.SCREEN
    fps: int = 30
    resolution: tuple = (1920, 1080)
    buffer_size: int = 10
    analysis_interval: int = 1  # Analyze every N frames


@dataclass
class AnalysisResult:
    """Stream analysis result"""
    timestamp: float
    frame_id: int
    detections: List[Dict]
    changes: Dict[str, Any]
    anomalies: List[str]


class StreamProcessor:
    """
    Process video streams in real-time with AI analysis.
    """

    def __init__(self, config: StreamConfig = None):
        self.config = config or StreamConfig()
        self._running = False
        self._frames: asyncio.Queue = None
        self._results: List[AnalysisResult] = []
        self._frame_count = 0
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()

    async def start(self):
        """Start processing the stream"""
        self._running = True
        self._frames = asyncio.Queue(maxsize=self.config.buffer_size)

        # Start capture loop
        asyncio.create_task(self._capture_loop())
        asyncio.create_task(self._process_loop())

        logger.info(f"Stream processor started: {self.config.source.value}")

    async def stop(self):
        """Stop processing"""
        self._running = False
        logger.info("Stream processor stopped")

    async def _capture_loop(self):
        """Capture frames from source"""
        while self._running:
            try:
                if self.config.source == StreamSource.SCREEN:
                    frame = ScreenCapture.capture_full()
                    await asyncio.sleep(1.0 / self.config.fps)

                elif self.config.source == StreamSource.WEBCAM:
                    # Would need cv2.VideoCapture for webcam
                    await asyncio.sleep(1.0 / self.config.fps)
                    continue

                else:
                    await asyncio.sleep(1.0 / self.config.fps)
                    continue

                # Add to queue
                try:
                    self._frames.put_nowait(frame)
                except asyncio.QueueFull:
                    pass  # Skip frame if buffer full

            except Exception as e:
                logger.error(f"Capture error: {e}")
                await asyncio.sleep(1)

    async def _process_loop(self):
        """Process frames"""
        while self._running:
            try:
                frame = await asyncio.wait_for(
                    self._frames.get(),
                    timeout=1.0
                )

                self._frame_count += 1

                # Analyze at interval
                if self._frame_count % self.config.analysis_interval == 0:
                    result = await self._analyze_frame(frame)
                    self._results.append(result)

                    # Notify callbacks
                    for callback in self._callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(result)
                            else:
                                callback(result)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")

    async def _analyze_frame(self, frame: np.ndarray) -> AnalysisResult:
        """Analyze a single frame"""
        # Basic analysis - can extend with AI
        result = AnalysisResult(
            timestamp=time.time(),
            frame_id=self._frame_count,
            detections=[],
            changes={},
            anomalies=[]
        )

        # Calculate frame statistics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result.changes = {
            "mean_brightness": float(np.mean(gray)),
            "std_brightness": float(np.std(gray))
        }

        return result

    def add_callback(self, callback: Callable):
        """Add analysis callback"""
        self._callbacks.append(callback)

    def get_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Get recent analysis results"""
        with self._lock:
            return self._results[-limit:]

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count


class OpticalFlowDetector:
    """
    Detect moving objects using optical flow.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self._prev_frame = None

    def detect_motion(self, frame: np.ndarray) -> Optional[tuple]:
        """Detect motion between frames"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame is None:
            self._prev_frame = gray
            return None

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_frame,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        # Calculate magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Get average motion
        avg_motion = np.mean(magnitude)

        self._prev_frame = gray

        if avg_motion > self.threshold:
            # Find region of motion
            rows, cols = np.where(magnitude > self.threshold)
            if len(rows) > 0:
                x1, y1 = cols.min(), rows.min()
                x2, y2 = cols.max(), rows.max()
                return ((x1, y1), (x2, y2), avg_motion)

        return None

    def reset(self):
        """Reset detector state"""
        self._prev_frame = None


# Global instances
_stream_processor: Optional[StreamProcessor] = None
_optical_flow: Optional[OpticalFlowDetector] = None


def get_stream_processor(config: StreamConfig = None) -> StreamProcessor:
    """Get global stream processor"""
    global _stream_processor
    if _stream_processor is None:
        _stream_processor = StreamProcessor(config)
    return _stream_processor


def get_optical_flow_detector(threshold: float = 1.0) -> OpticalFlowDetector:
    """Get global optical flow detector"""
    global _optical_flow
    if _optical_flow is None:
        _optical_flow = OpticalFlowDetector(threshold)
    return _optical_flow


__all__ = [
    "StreamSource",
    "StreamConfig",
    "AnalysisResult",
    "StreamProcessor",
    "OpticalFlowDetector",
    "get_stream_processor",
    "get_optical_flow_detector",
]
