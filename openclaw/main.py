"""OpenClaw Main Entry Point"""

import sys
import os
import argparse
import signal
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openclaw.core import (
    VisionConfig,
    VisionMode,
    ConfigManager,
    VisionEngine,
    ScreenCapture,
    TriggerAction,
    ActionSequence,
    setup_logging,
    get_logger
)
from openclaw.integrations import (
    VisionHTTPServer,
    RateLimiter,
    APIKeyAuth,
    TelegramBot,
    WebSocketManagerSync
)
from openclaw.storage import DatabaseManager

# Initialize logger
logger = get_logger("main")


class OpenClaw:
    """Main OpenClaw application"""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.running = False
        self.vision_engine = None
        self.http_server = None
        self.telegram_bot = None
        self.ws_manager = None
        self.db_manager = None

    def setup(self):
        """Setup all components"""
        # Setup logging
        if self.config.log_enabled:
            log_file = self.config.log_file or os.path.expanduser("~/.openclaw/logs/openclaw.log")
            log_dir = os.path.dirname(log_file)
            setup_logging(log_dir=log_dir)
        else:
            setup_logging()

        logger.info("OpenClaw v2.0.0 starting...")

        # Initialize database
        if self.config.db_enabled:
            db_path = self.config.db_path or os.path.expanduser("~/.openclaw/triggers.db")
            self.db_manager = DatabaseManager.get_instance(db_path)
            logger.info(f"Database enabled: {db_path}")

        # Initialize vision engine
        self.vision_engine = VisionEngine(self.config)
        logger.info(f"Vision engine initialized: {self.config.mode.value}")

        # Initialize HTTP server
        if self.config.http_enabled:
            self.http_server = VisionHTTPServer(
                8765, self.config,
                tls_enabled=self.config.tls_enabled,
                cert_path=self.config.tls_cert_path,
                key_path=self.config.tls_key_path
            )
            logger.info("HTTP server initialized on port 8765")

        # Initialize Telegram
        if self.config.telegram_enabled:
            self.telegram_bot = TelegramBot(
                token=self.config.telegram_token,
                chat_id=self.config.telegram_chat_id
            )
            if self.telegram_bot.enabled:
                logger.info("Telegram bot enabled")
                self.telegram_bot.start_command_listener()
            else:
                logger.warning("Telegram bot not configured")

        # Initialize WebSocket
        if self.config.websocket_enabled:
            self.ws_manager = WebSocketManagerSync(
                host=self.config.websocket_host,
                port=self.config.websocket_port
            )
            self.ws_manager.start()
            logger.info(f"WebSocket server enabled on port {self.config.websocket_port}")

        logger.info("Setup complete")

    def run(self):
        """Run the application"""
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("OpenClaw running. Press Ctrl+C to stop.")

        # Start HTTP server (blocking)
        try:
            self.http_server.start()
        except Exception as e:
            logger.error(f"HTTP server error: {e}")

    def stop(self):
        """Stop the application"""
        logger.info("Stopping OpenClaw...")
        self.running = False

        if self.http_server:
            self.http_server.stop()

        if self.ws_manager:
            self.ws_manager.stop()

        if self.db_manager:
            self.db_manager.close()

        logger.info("OpenClaw stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        self.stop()
        sys.exit(0)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="OpenClaw - Vision-enabled automation framework"
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="ocr",
        choices=["ocr", "monitor", "template", "color", "yolo", "fuzzy", "regression", "window"],
        help="Detection mode"
    )

    # Window monitoring options
    parser.add_argument("--window-signal", type=str, default="TRIGGER_CLAW", help="Window title signal to watch")
    parser.add_argument("--window-poll", type=float, default=0.3, help="Window poll interval")
    parser.add_argument("--window-debounce", type=float, default=3.0, help="Debounce seconds")

    # Detection options
    parser.add_argument("--text", type=str, help="Text to detect (OCR)")
    parser.add_argument("--region", type=str, help="Region: x,y,width,height")
    parser.add_argument("--template", type=str, help="Template image path")
    parser.add_argument("--color", type=str, help="Color: B,G,R")
    parser.add_argument("--threshold", type=float, default=0.05, help="Detection threshold")

    # Polling
    parser.add_argument("--poll", action="store_true", help="Enable polling")
    parser.add_argument("--interval", type=float, default=0.5, help="Poll interval")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive polling")

    # YOLO
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--yolo-classes", type=str, help="YOLO classes (comma-separated)")

    # Actions
    parser.add_argument("--action", type=str, default="alt+o", help="Action to trigger")
    parser.add_argument("--delay", type=float, default=1.5, help="Action delay")

    # Configuration
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--profile", type=str, help="Profile name")
    parser.add_argument("--save-profile", type=str, help="Save current config as profile")
    parser.add_argument("--list-profiles", action="store_true", help="List profiles")

    # Logging
    parser.add_argument("--log", type=str, help="Log file path")
    parser.add_argument("--log-enable", action="store_true", help="Enable logging")

    # Database
    parser.add_argument("--db-enable", action="store_true", help="Enable database")
    parser.add_argument("--db", type=str, help="Database path")
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    # Recording
    parser.add_argument("--record", action="store_true", help="Record on trigger")
    parser.add_argument("--record-dir", type=str, default="/tmp/openclaw_records", help="Recordings directory")

    # Webhooks
    parser.add_argument("--webhook", type=str, help="Webhook URL")
    parser.add_argument("--webhook-enable", action="store_true", help="Enable webhook")

    # Notifications
    parser.add_argument("--notify", action="store_true", help="Enable notifications")
    parser.add_argument("--notify-title", type=str, default="OpenClaw", help="Notification title")

    # Integrations
    parser.add_argument("--http-enable", dest="http_enabled", action="store_true", default=None, help="Enable HTTP server")
    parser.add_argument("--http-disable", dest="http_enabled", action="store_false", help="Disable HTTP server")
    parser.add_argument("--gateway-enable", action="store_true", help="Enable gateway")
    parser.add_argument("--gateway-port", type=int, default=18789, help="Gateway port")

    parser.add_argument("--telegram-enable", action="store_true", help="Enable Telegram")
    parser.add_argument("--telegram-token", type=str, help="Telegram bot token")
    parser.add_argument("--telegram-chat-id", type=str, help="Telegram chat ID")

    parser.add_argument("--websocket-enable", action="store_true", help="Enable WebSocket")
    parser.add_argument("--websocket-port", type=int, default=8766, help="WebSocket port")

    # Security
    parser.add_argument("--api-key", type=str, help="API key for HTTP endpoints")
    parser.add_argument("--rate-limit", type=int, default=60, help="Rate limit (req/min)")

    # Utilities
    parser.add_argument("--select-region", action="store_true", help="Select screen region")
    parser.add_argument("--list-monitors", action="store_true", help="List monitors")

    # Fuzzy/Advanced
    parser.add_argument("--fuzzy", action="store_true", help="Use fuzzy matching")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.8, help="Fuzzy threshold")
    parser.add_argument("--ocr-languages", type=str, default="en", help="OCR languages (comma-separated)")

    return parser.parse_args()


def build_config(args) -> VisionConfig:
    """Build config from args"""
    config_manager = ConfigManager()

    # Load from YAML if specified
    if args.config:
        return config_manager.load_config(args.config)

    # Load from profile if specified
    if args.profile:
        profile_path = os.path.expanduser(f"~/.openclaw/{args.profile}.yaml")
        if os.path.exists(profile_path):
            return config_manager.load_config(profile_path)

    # Parse region
    region = None
    if args.region:
        region = tuple(map(int, args.region.split(",")))

    # Parse color
    target_color = None
    if args.color:
        target_color = tuple(map(int, args.color.split(",")))

    # Parse YOLO classes
    yolo_classes = []
    if args.yolo_classes:
        yolo_classes = [c.strip() for c in args.yolo_classes.split(",")]

    # Parse OCR languages
    ocr_languages = [lang.strip() for lang in args.ocr_languages.split(",")]

    # Determine mode
    mode = VisionMode.FUZZY if args.fuzzy else VisionMode(args.mode)

    return VisionConfig(
        mode=mode,
        polling=args.poll,
        poll_interval=args.interval,
        adaptive_polling=args.adaptive,
        target_text=args.text,
        region=region,
        change_threshold=args.threshold,
        template_path=args.template,
        yolo_model=args.yolo_model,
        yolo_classes=yolo_classes,
        yolo_confidence=0.5,
        target_color=target_color,
        action=args.action,
        action_delay=args.delay,
        log_file=args.log,
        log_enabled=args.log_enable,
        record_on_trigger=args.record,
        record_dir=args.record_dir,
        db_enabled=args.db_enable,
        db_path=args.db,
        webhook_url=args.webhook,
        webhook_enabled=args.webhook_enable,
        notify_enabled=args.notify,
        notify_title=args.notify_title,
        gateway_enabled=args.gateway_enable,
        gateway_port=args.gateway_port,
        telegram_enabled=args.telegram_enable,
        telegram_token=args.telegram_token,
        telegram_chat_id=args.telegram_chat_id,
        websocket_enabled=args.websocket_enable,
        websocket_port=args.websocket_port,
        http_enabled=args.http_enabled if args.http_enabled is not None else True,
        api_key=args.api_key,
        rate_limit=args.rate_limit,
        fuzzy_threshold=args.fuzzy_threshold,
        ocr_languages=ocr_languages
    )


def main():
    """Main entry point"""
    args = parse_args()

    # List profiles
    if args.list_profiles:
        profile_dir = os.path.expanduser("~/.openclaw")
        if os.path.exists(profile_dir):
            profiles = [f.replace('.yaml', '') for f in os.listdir(profile_dir) if f.endswith('.yaml')]
            print(f"Available profiles: {profiles if profiles else 'None'}")
        return

    # Show stats
    if args.stats:
        db_path = args.db or os.path.expanduser("~/.openclaw/triggers.db")
        if os.path.exists(db_path):
            db = DatabaseManager.get_instance(db_path)
            stats = db.get_stats()
            print("\n=== Trigger Statistics ===")
            print(f"Total: {stats['total']}")
            print(f"Triggered: {stats['triggered']}")
            print(f"Failed: {stats['failed']}")
            print(f"Success Rate: {stats['success_rate']:.1f}%")
            print(f"By Mode: {stats['by_mode']}")
        else:
            print("No database found")
        return

    # Select region
    if args.select_region:
        from openclaw.core.vision import ScreenCapture
        import cv2
        try:
            region = None
            from openclaw.core.config import RegionSelector
            region = RegionSelector.select_region()
            if region:
                print(f"\nSelected region: {region}")
                print(f"Use with: --region '{region[0]},{region[1]},{region[2]},{region[3]}'")
        except Exception as e:
            print(f"Interactive selection failed: {e}")
            img = ScreenCapture.capture_full()
            cv2.imwrite("/tmp/region_capture.png", img)
            print(f"Screenshot saved: /tmp/region_capture.png")
        return

    # List monitors
    if args.list_monitors:
        from openclaw.core.vision import MultiMonitor
        monitors = MultiMonitor.get_monitors()
        print("\nConnected Monitors:")
        for i, m in enumerate(monitors):
            print(f"  {i+1}. {m['name']}: x={m['x']}, y={m['y']}, {m['width']}x{m['height']}")
        return

    # Build config
    config = build_config(args)

    # Save profile
    if args.save_profile:
        profile_dir = os.path.expanduser("~/.openclaw")
        os.makedirs(profile_dir, exist_ok=True)
        profile_path = os.path.join(profile_dir, f"{args.save_profile}.yaml")
        config_manager = ConfigManager()
        config_manager.save_config(profile_path, config.to_dict())
        print(f"Profile saved: {profile_path}")
        return

    # Run application
    app = OpenClaw(config)
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
