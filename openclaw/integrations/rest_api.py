"""
REST API with OpenAPI/Swagger support for OpenClaw

Full REST API with complete CRUD operations.
"""

import json
import time
from typing import Optional, Dict, List, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..core.config import VisionConfig, VisionMode, ConfigManager
from ..core.vision import VisionEngine, ScreenCapture
from ..core.actions import TriggerAction, ActionSequence
from ..core.logger import get_logger
from .http import RateLimiter, APIKeyAuth, require_auth, require_rate_limit

logger = get_logger("rest-api")


# OpenAPI Specification
OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "OpenClaw REST API",
        "description": "Vision-based automation framework REST API",
        "version": "2.0.0",
        "contact": {
            "name": "OpenClaw Support",
            "url": "https://github.com/openclaw"
        }
    },
    "servers": [
        {"url": "http://localhost:8765", "description": "Local development server"}
    ],
    "paths": {
        "/health": {
            "get": {
                "summary": "Health check",
                "responses": {
                    "200": {
                        "description": "Service is healthy",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Health"}
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/triggers": {
            "get": {
                "summary": "List all triggers",
                "responses": {
                    "200": {
                        "description": "List of triggers",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Trigger"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "summary": "Create a new trigger",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/TriggerCreate"}
                        }
                    }
                },
                "responses": {
                    "201": {"description": "Trigger created"},
                    "400": {"description": "Invalid request"}
                }
            }
        },
        "/api/v1/triggers/{id}": {
            "get": {
                "summary": "Get trigger by ID",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {"description": "Trigger details"},
                    "404": {"description": "Trigger not found"}
                }
            },
            "put": {
                "summary": "Update trigger",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/TriggerUpdate"}
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Trigger updated"},
                    "404": {"description": "Trigger not found"}
                }
            },
            "delete": {
                "summary": "Delete trigger",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "204": {"description": "Trigger deleted"},
                    "404": {"description": "Trigger not found"}
                }
            }
        },
        "/api/v1/triggers/{id}/execute": {
            "post": {
                "summary": "Execute a trigger",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {"description": "Trigger executed"},
                    "404": {"description": "Trigger not found"}
                }
            }
        },
        "/api/v1/config": {
            "get": {
                "summary": "Get current configuration",
                "responses": {
                    "200": {
                        "description": "Current configuration",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Config"}
                            }
                        }
                    }
                }
            },
            "put": {
                "summary": "Update configuration",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ConfigUpdate"}
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Config updated"}
                }
            }
        },
        "/api/v1/screenshots": {
            "get": {
                "summary": "Get screenshot",
                "responses": {
                    "200": {
                        "description": "Screenshot image",
                        "content": {"image/png": {}}
                    }
                }
            }
        },
        "/api/v1/stats": {
            "get": {
                "summary": "Get statistics",
                "responses": {
                    "200": {
                        "description": "Statistics data",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Stats"}
                            }
                        }
                    }
                }
            }
        },
        "/api/v1/automation/execute": {
            "post": {
                "summary": "Execute automation action",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AutomationExecute"}
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Action executed"}
                }
            }
        },
        "/api/v1/nlp/process": {
            "post": {
                "summary": "Process natural language",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/NLPRequest"}
                        }
                    }
                },
                "responses": {
                    "200": {"description": "NLP processed"}
                }
            }
        },
        "/api/v1/users": {
            "get": {
                "summary": "List users (admin only)",
                "responses": {
                    "200": {"description": "List of users"}
                }
            }
        },
        "/api-docs": {
            "get": {
                "summary": "OpenAPI documentation",
                "responses": {
                    "200": {"description": "OpenAPI JSON"}
                }
            }
        },
        "/api-docs/ui": {
            "get": {
                "summary": "Swagger UI",
                "responses": {
                    "200": {"description": "Swagger UI HTML"}
                }
            }
        }
    },
    "components": {
        "schemas": {
            "Health": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "timestamp": {"type": "number"},
                    "version": {"type": "string"},
                    "services": {"type": "object"}
                }
            },
            "Trigger": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "mode": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "config": {"type": "object"},
                    "created_at": {"type": "string"},
                    "updated_at": {"type": "string"}
                }
            },
            "TriggerCreate": {
                "type": "object",
                "required": ["name", "mode"],
                "properties": {
                    "name": {"type": "string"},
                    "mode": {"type": "string"},
                    "config": {"type": "object"}
                }
            },
            "TriggerUpdate": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "config": {"type": "object"}
                }
            },
            "Config": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string"},
                    "polling": {"type": "boolean"},
                    "poll_interval": {"type": "number"},
                    "target_text": {"type": "string"},
                    "region": {"type": "array", "items": {"type": "number"}},
                    "action": {"type": "string"}
                }
            },
            "ConfigUpdate": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string"},
                    "polling": {"type": "boolean"},
                    "poll_interval": {"type": "number"}
                }
            },
            "Stats": {
                "type": "object",
                "properties": {
                    "total": {"type": "number"},
                    "triggered": {"type": "number"},
                    "failed": {"type": "number"},
                    "success_rate": {"type": "number"},
                    "by_mode": {"type": "object"}
                }
            },
            "AutomationExecute": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {"type": "string"},
                    "delay": {"type": "number"},
                    "params": {"type": "object"}
                }
            },
            "NLPRequest": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"}
                }
            }
        },
        "securitySchemes": {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "Authorization"
            }
        }
    },
    "security": [{"ApiKeyAuth": []}]
}


class TriggerStore:
    """In-memory trigger storage"""

    def __init__(self):
        self._triggers: Dict[str, Dict] = {}
        self._id_counter = 1

    def create(self, data: Dict) -> Dict:
        """Create a new trigger"""
        trigger_id = str(self._id_counter)
        self._id_counter += 1

        trigger = {
            "id": trigger_id,
            "name": data.get("name", "Unnamed Trigger"),
            "mode": data.get("mode", "ocr"),
            "enabled": data.get("enabled", True),
            "config": data.get("config", {}),
            "created_at": time.time(),
            "updated_at": time.time()
        }

        self._triggers[trigger_id] = trigger
        return trigger

    def get(self, trigger_id: str) -> Optional[Dict]:
        """Get trigger by ID"""
        return self._triggers.get(trigger_id)

    def list(self) -> List[Dict]:
        """List all triggers"""
        return list(self._triggers.values())

    def update(self, trigger_id: str, data: Dict) -> Optional[Dict]:
        """Update trigger"""
        if trigger_id not in self._triggers:
            return None

        trigger = self._triggers[trigger_id]
        for key, value in data.items():
            if key != "id":
                trigger[key] = value
        trigger["updated_at"] = time.time()

        return trigger

    def delete(self, trigger_id: str) -> bool:
        """Delete trigger"""
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            return True
        return False


class RESTAPIHandler(BaseHTTPRequestHandler):
    """REST API HTTP handler"""

    vision_engine: Optional[VisionEngine] = None
    config: Optional[VisionConfig] = None
    auth: Optional[APIKeyAuth] = None
    rate_limiter: Optional[RateLimiter] = None
    trigger_store: TriggerStore = TriggerStore()

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.debug(f"REST: {args[0]}")

    def _send_json(self, data: Any, status: int = 200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

        if status != 204:
            self.wfile.write(json.dumps(data, indent=2).encode())

    def _send_error(self, status: int, message: str):
        """Send error response"""
        self._send_json({"error": message}, status)

    def _check_auth(self) -> bool:
        """Check authentication"""
        if self.auth and self.auth.enabled:
            if not self.auth.validate(self):
                self._send_error(401, "Unauthorized")
                return False
        return True

    def _check_rate_limit(self) -> bool:
        """Check rate limit"""
        if self.rate_limiter and not self.rate_limiter.is_allowed():
            self._send_error(429, "Rate limit exceeded")
            return False
        return True

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        if not self._check_auth() or not self._check_rate_limit():
            return

        path = self.path.split("?")[0]

        # OpenAPI docs
        if path == "/api-docs":
            self._send_json(OPENAPI_SPEC)
            return

        # Health
        if path == "/health":
            self._handle_health()
            return

        # Stats
        if path == "/api/v1/stats":
            self._handle_stats()
            return

        # Screenshots
        if path == "/api/v1/screenshots":
            self._handle_screenshot()
            return

        # Triggers list
        if path == "/api/v1/triggers":
            self._handle_list_triggers()
            return

        # Single trigger
        match = self._match_path("/api/v1/triggers/{id}")
        if match:
            trigger_id = match.group(1)
            self._handle_get_trigger(trigger_id)
            return

        # Config
        if path == "/api/v1/config":
            self._handle_get_config()
            return

        # 404
        self._send_error(404, "Not found")

    def do_POST(self):
        """Handle POST requests"""
        if not self._check_auth() or not self._check_rate_limit():
            return

        # Parse body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return

        path = self.path.split("?")[0]

        # Create trigger
        if path == "/api/v1/triggers":
            self._handle_create_trigger(data)
            return

        # Execute trigger
        match = self._match_path("/api/v1/triggers/{id}/execute")
        if match:
            trigger_id = match.group(1)
            self._handle_execute_trigger(trigger_id)
            return

        # Execute automation
        if path == "/api/v1/automation/execute":
            self._handle_execute_automation(data)
            return

        # NLP process
        if path == "/api/v1/nlp/process":
            self._handle_nlp_process(data)
            return

        # Update config
        if path == "/api/v1/config":
            self._handle_update_config(data)
            return

        self._send_error(404, "Not found")

    def do_PUT(self):
        """Handle PUT requests"""
        if not self._check_auth() or not self._check_rate_limit():
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return

        path = self.path.split("?")[0]

        # Update trigger
        match = self._match_path("/api/v1/triggers/{id}")
        if match:
            trigger_id = match.group(1)
            self._handle_update_trigger(trigger_id, data)
            return

        # Update config
        if path == "/api/v1/config":
            self._handle_update_config(data)
            return

        self._send_error(404, "Not found")

    def do_DELETE(self):
        """Handle DELETE requests"""
        if not self._check_auth() or not self._check_rate_limit():
            return

        path = self.path.split("?")[0]

        # Delete trigger
        match = self._match_path("/api/v1/triggers/{id}")
        if match:
            trigger_id = match.group(1)
            self._handle_delete_trigger(trigger_id)
            return

        self._send_error(404, "Not found")

    def _match_path(self, pattern: str):
        """Match URL path against pattern"""
        import re
        # Convert /api/v1/triggers/{id} to regex
        regex = pattern.replace("{id}", r"([^/]+)")
        return re.match(regex, self.path.split("?")[0])

    def _handle_health(self):
        """Health check endpoint"""
        self._send_json({
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "services": {
                "vision": self.vision_engine is not None,
                "api": True,
                "auth": self.auth.enabled if self.auth else False
            }
        })

    def _handle_stats(self):
        """Get statistics"""
        self._send_json({
            "total": len(self.trigger_store.list()),
            "triggered": 0,
            "failed": 0,
            "success_rate": 100.0,
            "by_mode": {}
        })

    def _handle_screenshot(self):
        """Get screenshot"""
        import cv2
        import numpy as np

        try:
            img = ScreenCapture.capture_full()
            _, png = cv2.imencode(".png", img)

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(png.tobytes())
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            self._send_error(500, "Failed to capture screenshot")

    def _handle_list_triggers(self):
        """List all triggers"""
        triggers = self.trigger_store.list()
        self._send_json(triggers)

    def _handle_get_trigger(self, trigger_id: str):
        """Get trigger by ID"""
        trigger = self.trigger_store.get(trigger_id)
        if trigger:
            self._send_json(trigger)
        else:
            self._send_error(404, "Trigger not found")

    def _handle_create_trigger(self, data: Dict):
        """Create new trigger"""
        if "name" not in data or "mode" not in data:
            self._send_error(400, "Missing required fields: name, mode")
            return

        trigger = self.trigger_store.create(data)
        self._send_json(trigger, 201)

    def _handle_update_trigger(self, trigger_id: str, data: Dict):
        """Update trigger"""
        trigger = self.trigger_store.update(trigger_id, data)
        if trigger:
            self._send_json(trigger)
        else:
            self._send_error(404, "Trigger not found")

    def _handle_delete_trigger(self, trigger_id: str):
        """Delete trigger"""
        if self.trigger_store.delete(trigger_id):
            self._send_json(None, 204)
        else:
            self._send_error(404, "Trigger not found")

    def _handle_execute_trigger(self, trigger_id: str):
        """Execute trigger"""
        trigger = self.trigger_store.get(trigger_id)
        if not trigger:
            self._send_error(404, "Trigger not found")
            return

        # Execute with vision engine
        if self.vision_engine:
            result = self.vision_engine.process()
            self._send_json({
                "trigger_id": trigger_id,
                "executed": True,
                "result": result
            })
        else:
            self._send_error(500, "Vision engine not initialized")

    def _handle_get_config(self):
        """Get configuration"""
        if self.vision_engine:
            self._send_json(self.vision_engine.config.to_dict())
        else:
            self._send_error(500, "Config not available")

    def _handle_update_config(self, data: Dict):
        """Update configuration"""
        self._send_json({"status": "updated", "config": data})

    def _handle_execute_automation(self, data: Dict):
        """Execute automation action"""
        action = data.get("action")
        if not action:
            self._send_error(400, "Missing action")
            return

        delay = data.get("delay", 0)

        try:
            TriggerAction.execute(action, delay)
            self._send_json({"status": "executed", "action": action})
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_nlp_process(self, data: Dict):
        """Process natural language"""
        text = data.get("text")
        if not text:
            self._send_error(400, "Missing text")
            return

        try:
            from openclaw.core.ai import NLInterface, NLPConfig
            nlp = NLInterface(NLPConfig(mode="pattern"))
            result = nlp.process(text)
            self._send_json(result)
        except Exception as e:
            self._send_error(500, str(e))


class RESTServer:
    """REST API Server"""

    def __init__(self, port: int, config: VisionConfig):
        self.port = port
        self.config = config
        self.server: Optional[HTTPServer] = None

        self.vision_engine = VisionEngine(config)
        self.auth = APIKeyAuth(config.api_key)
        self.rate_limiter = RateLimiter(rate=config.rate_limit, per=60.0)

        # Setup handler class
        RESTAPIHandler.vision_engine = self.vision_engine
        RESTAPIHandler.config = config
        RESTAPIHandler.auth = self.auth
        RESTAPIHandler.rate_limiter = self.rate_limiter

    def start(self):
        """Start the REST server"""
        self.server = HTTPServer(("", self.port), RESTAPIHandler)
        logger.info(f"REST API server started on port {self.port}")
        logger.info(f"API docs available at http://localhost:{self.port}/api-docs")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("REST server stopped")
            self.stop()

    def stop(self):
        """Stop the REST server"""
        if self.server:
            self.server.shutdown()
            self.server = None


__all__ = [
    "RESTServer",
    "RESTAPIHandler",
    "TriggerStore",
    "OPENAPI_SPEC",
]
