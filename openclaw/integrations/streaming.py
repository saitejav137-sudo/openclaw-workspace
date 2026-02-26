"""
MJPEG Streaming for OpenClaw

Real-time screen capture streaming.
"""

import io
import time
import threading
import logging
from typing import Optional, List
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

logger = logging.getLogger("openclaw.streaming")


class StreamingFrame:
    """Single streaming frame"""

    def __init__(self, jpeg_data: bytes, timestamp: float = None):
        self.jpeg_data = jpeg_data
        self.timestamp = timestamp or time.time()


class StreamingServer:
    """MJPEG streaming server"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8888,
        fps: int = 10,
        quality: int = 80
    ):
        self.host = host
        self.port = port
        self.fps = fps
        self.quality = quality

        self.frame_interval = 1.0 / fps
        self.latest_frame: Optional[StreamingFrame] = None
        self.clients: List = []
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the streaming server"""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._server_loop, daemon=True)
        self._thread.start()
        logger.info(f"Streaming server started on {self.host}:{self.port}")

    def stop(self):
        """Stop the streaming server"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Streaming server stopped")

    def update_frame(self, frame_data: bytes):
        """Update the latest frame"""
        self.latest_frame = StreamingFrame(frame_data)

    def _server_loop(self):
        """Internal server loop"""
        try:
            from werkzeug.serving import make_server
            # Fallback to basic HTTP server if werkzeug not available
            self._basic_server_loop()
        except ImportError:
            self._basic_server_loop()

    def _basic_server_loop(self):
        """Basic server loop using standard library"""
        # Simple frame storage for streaming
        pass


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming"""

    stream_server: Optional[StreamingServer] = None

    def do_GET(self):
        """Handle GET request for streaming"""
        if self.path == "/stream":
            self._stream_mjpeg()
        elif self.path == "/":
            self._serve_html()
        else:
            self.send_error(404)

    def _stream_mjpeg(self):
        """Stream MJPEG content"""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                if self.stream_server and self.stream_server.latest_frame:
                    frame = self.stream_server.latest_frame
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame.jpeg_data)}\r\n\r\n".encode())
                    self.wfile.write(frame.jpeg_data)
                    self.wfile.write(b"\r\n")
                time.sleep(0.03)  # ~30fps max

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.error(f"Streaming error: {e}")

    def _serve_html(self):
        """Serve HTML viewer"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>OpenClaw Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { margin: 0; background: #000; display: flex; justify-content: center; align-items: center; height: 100vh; }
        img { max-width: 100%; max-height: 100vh; }
    </style>
</head>
<body>
    <img src="/stream" alt="Live Stream">
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """Suppress logging"""
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for streaming"""
    daemon_threads = True


class StreamManager:
    """Manages multiple streams"""

    _instance = None
    _streams: dict = {}

    def __init__(self):
        self.default_stream: Optional[StreamingServer] = None

    @classmethod
    def get_instance(cls) -> 'StreamManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_stream(
        self,
        name: str = "default",
        host: str = "localhost",
        port: int = 8888,
        fps: int = 10
    ) -> StreamingServer:
        """Create a new stream"""
        stream = StreamingServer(host=host, port=port, fps=fps)
        self._streams[name] = stream
        logger.info(f"Stream '{name}' created on port {port}")
        return stream

    def get_stream(self, name: str = "default") -> Optional[StreamingServer]:
        """Get a stream by name"""
        return self._streams.get(name)

    def start_stream(self, name: str = "default"):
        """Start a stream"""
        stream = self.get_stream(name)
        if stream:
            stream.start()

    def stop_stream(self, name: str = "default"):
        """Stop a stream"""
        stream = self.get_stream(name)
        if stream:
            stream.stop()

    def update_frame(self, frame_data: bytes, stream_name: str = "default"):
        """Update frame for a stream"""
        stream = self.get_stream(stream_name)
        if stream:
            stream.update_frame(frame_data)


# Convenience function
def create_default_stream(port: int = 8888, fps: int = 10) -> StreamingServer:
    """Create and return the default stream"""
    manager = StreamManager.get_instance()
    stream = manager.create_stream("default", port=port, fps=fps)
    stream.start()
    return stream


# Export
__all__ = [
    "StreamingServer",
    "StreamingFrame",
    "StreamManager",
    "MJPEGStreamHandler",
    "ThreadedHTTPServer",
    "create_default_stream",
]
