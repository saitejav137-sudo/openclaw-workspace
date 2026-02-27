"""
Voice Commands Module for OpenClaw

Speech recognition for voice-controlled automation.
Supports multiple backends: Google Speech Recognition, Whisper, Vosk.
"""

import time
import threading
import logging
import json
import queue
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("openclaw.voice")


class VoiceBackend(Enum):
    """Available voice recognition backends"""
    GOOGLE = "google"
    WHISPER = "whisper"
    VOSK = "vosk"
    SPHINX = "sphinx"


@dataclass
class VoiceCommand:
    """A voice command definition"""
    phrases: List[str]
    action: str
    description: str = ""
    requires_callback: bool = False


@dataclass
class VoiceConfig:
    """Configuration for voice recognition"""
    backend: VoiceBackend = VoiceBackend.GOOGLE
    language: str = "en-US"
    energy_threshold: int = 300
    phrase_time_limit: int = 5
    mic_index: Optional[int] = None
    confidence_threshold: float = 0.6
    continuous: bool = False
    verbose: bool = True


class VoiceEngine:
    """
    Main voice recognition engine with multiple backend support.
    """

    def __init__(self, config: VoiceConfig):
        self.config = config
        self._recognizer = None
        self._microphone = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, Callable] = {}
        self._commands: Dict[str, VoiceCommand] = {}
        self._event_queue: queue.Queue = queue.Queue()
        self._last_command_time: float = 0
        self._command_cooldown: float = 2.0  # seconds

    def initialize(self) -> bool:
        """Initialize the voice recognition engine"""
        try:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = self.config.energy_threshold

            # Get microphone
            self._microphone = sr.Microphone(
                device_index=self.config.mic_index
            )

            # Calibrate microphone
            logger.info("Calibrating microphone...")
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)

            logger.info(f"Voice engine initialized with {self.config.backend.value} backend")
            return True

        except ImportError as e:
            logger.error(f"Speech recognition library not available: {e}")
            return False
        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to initialize voice engine: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize voice engine (unexpected): {e}")
            return False

    def register_command(self, command: VoiceCommand):
        """Register a voice command"""
        for phrase in command.phrases:
            self._commands[phrase.lower()] = command
        logger.info(f"Registered command: {command.action}")

    def register_callback(self, action: str, callback: Callable):
        """Register a callback for an action"""
        self._callbacks[action] = callback
        logger.info(f"Registered callback for action: {action}")

    def start_listening(self):
        """Start continuous listening in background"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Voice listening started")

    def stop_listening(self):
        """Stop continuous listening"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Voice listening stopped")

    def listen_once(self, timeout: int = 5) -> Optional[str]:
        """Listen for a single command"""
        if not self._recognizer or not self._microphone:
            if not self.initialize():
                return None

        import speech_recognition as sr

        try:
            with self._microphone as source:
                logger.info("Listening...")
                audio = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=self.config.phrase_time_limit
                )

            # Recognize using configured backend
            if self.config.backend == VoiceBackend.GOOGLE:
                result = self._recognizer.recognize_google(
                    audio,
                    language=self.config.language
                )
            elif self.config.backend == VoiceBackend.SPHINX:
                result = self._recognizer.recognize_sphinx(audio)
            else:
                logger.warning(f"Backend {self.config.backend.value} not implemented")
                return None

            if result:
                logger.info(f"Recognized: {result}")
                return result.lower()

        except sr.WaitTimeoutError:
            logger.debug("No speech detected within timeout")
        except sr.UnknownValueError:
            logger.debug("Speech not understood")
        except Exception as e:
            logger.error(f"Recognition error: {e}")

        return None

    def _listen_loop(self):
        """Background listening loop"""
        import speech_recognition as sr

        while self._running:
            try:
                # Check for stop
                if not self._running:
                    break

                with self._microphone as source:
                    audio = self._recognizer.listen(
                        source,
                        phrase_time_limit=self.config.phrase_time_limit
                    )

                # Process in separate thread to not block listening
                threading.Thread(
                    target=self._process_audio,
                    args=(audio,),
                    daemon=True
                ).start()

            except (sr.WaitTimeoutError, sr.UnknownValueError) as e:
                logger.debug(f"Listen loop: {e}")
            except (OSError, RuntimeError) as e:
                logger.error(f"Listen loop error: {e}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Listen loop error (unexpected): {e}")
                time.sleep(1)

    def _process_audio(self, audio):
        """Process audio in background"""
        try:
            import speech_recognition as sr

            # Recognize
            if self.config.backend == VoiceBackend.GOOGLE:
                result = self._recognizer.recognize_google(
                    audio,
                    language=self.config.language
                )
            else:
                return

            if not result:
                return

            text = result.lower()
            logger.info(f"Recognized: {text}")

            # Check cooldown
            current_time = time.time()
            if current_time - self._last_command_time < self._command_cooldown:
                logger.debug("Command in cooldown")
                return

            # Match against registered commands
            matched_command = self._match_command(text)
            if matched_command:
                self._last_command_time = current_time
                self._execute_command(matched_command, text)

        except sr.UnknownValueError:
            logger.debug("Speech not understood")
        except Exception as e:
            logger.error(f"Audio processing error: {e}")

    def _match_command(self, text: str) -> Optional[VoiceCommand]:
        """Match recognized text against registered commands"""
        text = text.lower()

        # Exact match
        if text in self._commands:
            return self._commands[text]

        # Partial match (command phrase contained in text)
        for phrase, command in self._commands.items():
            if phrase in text:
                return command

        # Fuzzy match
        from openclaw.core.vision import FuzzyMatcher
        best_match = None
        best_score = 0

        for phrase, command in self._commands.items():
            score = FuzzyMatcher.similarity(text, phrase)
            if score > best_score and score >= self.config.confidence_threshold:
                best_score = score
                best_match = command

        return best_match

    def _execute_command(self, command: VoiceCommand, text: str):
        """Execute a matched command"""
        logger.info(f"Executing command: {command.action}")

        # Add to event queue
        self._event_queue.put({
            "command": command.action,
            "text": text,
            "timestamp": time.time()
        })

        # Call callback if registered
        if command.action in self._callbacks:
            try:
                self._callbacks[command.action](text)
            except (TypeError, ValueError) as e:
                logger.error(f"Callback error (invalid arguments): {e}")
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_event(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get a voice event from the queue"""
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_listening(self) -> bool:
        """Check if engine is listening"""
        return self._running

    @classmethod
    def is_available(cls) -> bool:
        """Check if voice recognition is available"""
        try:
            import speech_recognition as sr
            return True
        except ImportError:
            return False


class VoiceTrigger:
    """
    Voice-triggered automation for OpenClaw.
    Integrates voice commands with the automation system.
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self.engine = VoiceEngine(self.config)
        self._action_executor: Optional[Any] = None

    def initialize(self) -> bool:
        """Initialize the voice trigger"""
        return self.engine.initialize()

    def add_voice_command(
        self,
        phrases: List[str],
        action: str,
        description: str = ""
    ):
        """Add a voice command that triggers an action"""
        command = VoiceCommand(
            phrases=phrases,
            action=action,
            description=description
        )
        self.engine.register_command(command)

    def set_action_executor(self, executor):
        """Set the action executor for triggered commands"""
        self._action_executor = executor

    def start(self):
        """Start voice trigger"""
        if not self.engine.initialize():
            logger.error("Failed to initialize voice engine")
            return False

        # Register command handler
        self.engine.register_callback("execute", self._handle_command)

        # Start listening
        self.engine.start_listening()
        return True

    def stop(self):
        """Stop voice trigger"""
        self.engine.stop_listening()

    def _handle_command(self, text: str):
        """Handle executed voice command"""
        if self._action_executor:
            # Execute the action
            logger.info(f"Executing voice-triggered action: {text}")

    def wait_for_command(self, timeout: int = 10) -> Optional[str]:
        """Wait for a single voice command"""
        return self.engine.listen_once(timeout=timeout)


# Default voice commands for OpenClaw
DEFAULT_COMMANDS = [
    VoiceCommand(
        phrases=["trigger", "activate", "start"],
        action="trigger",
        description="Trigger the automation"
    ),
    VoiceCommand(
        phrases=["stop", "halt", "pause"],
        action="stop",
        description="Stop the automation"
    ),
    VoiceCommand(
        phrases=["status", "report"],
        action="status",
        description="Get automation status"
    ),
    VoiceCommand(
        phrases=["capture", "screenshot", "snapshot"],
        action="capture",
        description="Take a screenshot"
    ),
]


def create_voice_trigger(
    language: str = "en-US",
    backend: VoiceBackend = VoiceBackend.GOOGLE,
    commands: bool = True
) -> VoiceTrigger:
    """Create a voice trigger with default or custom commands"""
    config = VoiceConfig(
        backend=backend,
        language=language
    )

    trigger = VoiceTrigger(config)

    if commands:
        for cmd in DEFAULT_COMMANDS:
            trigger.add_voice_command(
                phrases=cmd.phrases,
                action=cmd.action,
                description=cmd.description
            )

    return trigger


__all__ = [
    "VoiceEngine",
    "VoiceTrigger",
    "VoiceConfig",
    "VoiceCommand",
    "VoiceBackend",
    "DEFAULT_COMMANDS",
    "create_voice_trigger",
]
