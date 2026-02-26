"""
Web-based Configuration Editor for OpenClaw

Visual editor for creating and editing automation configurations.
"""

import json
from typing import Dict, List, Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler

from ..core.config import VisionConfig, VisionMode, ConfigManager
from ..core.logger import get_logger

logger = get_logger("config-editor")


# Config Editor HTML
CONFIG_EDITOR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenClaw Config Editor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: #e2e8f0; }
        .card { background: #1e293b; border-radius: 12px; border: 1px solid #334155; }
        .input { background: #0f172a; border: 1px solid #334155; color: #e2e8f0; border-radius: 8px; padding: 10px 14px; }
        .input:focus { outline: none; border-color: #3b82f6; ring: 2px; }
        .btn { padding: 10px 20px; border-radius: 8px; font-weight: 500; transition: all 0.2s; }
        .btn-primary { background: #3b82f6; color: white; }
        .btn-primary:hover { background: #2563eb; }
        .btn-success { background: #10b981; color: white; }
        .btn-danger { background: #ef4444; color: white; }
        .btn-secondary { background: #64748b; color: white; }
        .mode-btn { padding: 8px 16px; border-radius: 6px; background: #334155; transition: all 0.2s; }
        .mode-btn.active { background: #3b82f6; }
        .mode-btn:hover:not(.active) { background: #475569; }
        .section { display: none; }
        .section.active { display: block; }
        .json-editor { font-family: 'Fira Code', monospace; font-size: 13px; }
        pre { background: #0f172a; padding: 16px; border-radius: 8px; overflow: auto; }
    </style>
</head>
<body class="min-h-screen">
    <nav class="bg-slate-800 border-b border-slate-700 px-6 py-4">
        <div class="flex items-center justify-between">
            <div class="flex items-center gap-4">
                <i class="fas fa-claw text-2xl text-blue-500"></i>
                <h1 class="text-xl font-bold">OpenClaw Config Editor</h1>
            </div>
            <div class="flex gap-3">
                <button onclick="loadConfig()" class="btn btn-secondary">
                    <i class="fas fa-download mr-2"></i>Load
                </button>
                <button onclick="saveConfig()" class="btn btn-primary">
                    <i class="fas fa-save mr-2"></i>Save
                </button>
                <button onclick="exportJSON()" class="btn btn-secondary">
                    <i class="fas fa-file-code mr-2"></i>Export
                </button>
            </div>
        </div>
    </nav>

    <div class="container mx-auto p-6">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Left Panel: Configuration -->
            <div class="lg:col-span-2 space-y-6">
                <!-- Mode Selection -->
                <div class="card p-6">
                    <h2 class="text-lg font-semibold mb-4">Detection Mode</h2>
                    <div class="flex flex-wrap gap-2" id="modeButtons">
                        <button class="mode-btn active" data-mode="ocr">OCR</button>
                        <button class="mode-btn" data-mode="fuzzy">Fuzzy</button>
                        <button class="mode-btn" data-mode="template">Template</button>
                        <button class="mode-btn" data-mode="color">Color</button>
                        <button class="mode-btn" data-mode="monitor">Monitor</button>
                        <button class="mode-btn" data-mode="yolo">YOLO</button>
                        <button class="mode-btn" data-mode="window">Window</button>
                        <button class="mode-btn" data-mode="multi">Multi</button>
                    </div>
                </div>

                <!-- Mode-specific Settings -->
                <div class="card p-6">
                    <h2 class="text-lg font-semibold mb-4">Settings</h2>

                    <!-- OCR/Fuzzy Settings -->
                    <div class="section active" data-section="ocr">
                        <div class="space-y-4">
                            <div>
                                <label class="block text-sm font-medium mb-2">Target Text</label>
                                <input type="text" id="targetText" class="input w-full" placeholder="Text to detect">
                            </div>
                            <div class="flex items-center gap-2">
                                <input type="checkbox" id="caseSensitive" class="w-4 h-4">
                                <label for="caseSensitive">Case Sensitive</label>
                            </div>
                        </div>
                    </div>

                    <!-- Template Settings -->
                    <div class="section" data-section="template">
                        <div class="space-y-4">
                            <div>
                                <label class="block text-sm font-medium mb-2">Template Path</label>
                                <input type="text" id="templatePath" class="input w-full" placeholder="/path/to/template.png">
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-2">Threshold</label>
                                <input type="range" id="templateThreshold" min="0" max="100" value="80" class="w-full">
                                <span id="thresholdValue">80%</span>
                            </div>
                        </div>
                    </div>

                    <!-- Color Settings -->
                    <div class="section" data-section="color">
                        <div class="space-y-4">
                            <div>
                                <label class="block text-sm font-medium mb-2">Target Color (BGR)</label>
                                <div class="flex gap-2">
                                    <input type="number" id="colorB" class="input w-20" placeholder="B" min="0" max="255">
                                    <input type="number" id="colorG" class="input w-20" placeholder="G" min="0" max="255">
                                    <input type="number" id="colorR" class="input w-20" placeholder="R" min="0" max="255">
                                    <div id="colorPreview" class="w-12 h-10 rounded border border-slate-600"></div>
                                </div>
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-2">Tolerance</label>
                                <input type="number" id="colorTolerance" class="input w-32" value="30" min="0" max="100">
                            </div>
                        </div>
                    </div>

                    <!-- Monitor Settings -->
                    <div class="section" data-section="monitor">
                        <div class="space-y-4">
                            <div>
                                <label class="block text-sm font-medium mb-2">Region (x,y,w,h)</label>
                                <div class="flex gap-2">
                                    <input type="number" id="regionX" class="input w-24" placeholder="X">
                                    <input type="number" id="regionY" class="input w-24" placeholder="Y">
                                    <input type="number" id="regionW" class="input w-24" placeholder="W">
                                    <input type="number" id="regionH" class="input w-24" placeholder="H">
                                </div>
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-2">Change Threshold</label>
                                <input type="range" id="changeThreshold" min="0" max="100" value="5" class="w-full">
                                <span id="changeValue">5%</span>
                            </div>
                        </div>
                    </div>

                    <!-- YOLO Settings -->
                    <div class="section" data-section="yolo">
                        <div class="space-y-4">
                            <div>
                                <label class="block text-sm font-medium mb-2">Model</label>
                                <select id="yoloModel" class="input w-full">
                                    <option value="yolov8n.pt">YOLOv8 Nano</option>
                                    <option value="yolov8s.pt">YOLOv8 Small</option>
                                    <option value="yolov8m.pt">YOLOv8 Medium</option>
                                </select>
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-2">Classes (comma-separated)</label>
                                <input type="text" id="yoloClasses" class="input w-full" placeholder="person, car, dog">
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-2">Confidence</label>
                                <input type="range" id="yoloConfidence" min="0" max="100" value="50" class="w-full">
                                <span id="confidenceValue">50%</span>
                            </div>
                        </div>
                    </div>

                    <!-- Window Settings -->
                    <div class="section" data-section="window">
                        <div class="space-y-4">
                            <div>
                                <label class="block text-sm font-medium mb-2">Window Signal</label>
                                <input type="text" id="windowSignal" class="input w-full" placeholder="TRIGGER_CLAW">
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-2">Poll Interval (s)</label>
                                <input type="number" id="windowPoll" class="input w-32" value="0.3" step="0.1">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Action Settings -->
                <div class="card p-6">
                    <h2 class="text-lg font-semibold mb-4">Action</h2>
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium mb-2">Keyboard Action</label>
                            <input type="text" id="action" class="input w-full" placeholder="alt+o">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-2">Action Delay (s)</label>
                            <input type="number" id="actionDelay" class="input w-32" value="1.5" step="0.1">
                        </div>
                    </div>
                </div>

                <!-- Polling Settings -->
                <div class="card p-6">
                    <h2 class="text-lg font-semibold mb-4">Polling</h2>
                    <div class="space-y-4">
                        <div class="flex items-center gap-2">
                            <input type="checkbox" id="pollingEnabled" class="w-4 h-4">
                            <label for="pollingEnabled">Enable Polling</label>
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-2">Poll Interval (s)</label>
                            <input type="number" id="pollInterval" class="input w-32" value="0.5" step="0.1">
                        </div>
                        <div class="flex items-center gap-2">
                            <input type="checkbox" id="adaptivePolling" class="w-4 h-4">
                            <label for="adaptivePolling">Adaptive Polling</label>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Panel: Preview -->
            <div class="space-y-6">
                <!-- JSON Preview -->
                <div class="card p-6">
                    <h2 class="text-lg font-semibold mb-4">Configuration JSON</h2>
                    <pre id="jsonPreview" class="json-editor text-sm"></pre>
                </div>

                <!-- Templates -->
                <div class="card p-6">
                    <h2 class="text-lg font-semibold mb-4">Templates</h2>
                    <div class="space-y-2">
                        <button onclick="loadTemplate('ocr')" class="btn btn-secondary w-full text-left">
                            <i class="fas fa-font mr-2"></i>Text Detection
                        </button>
                        <button onclick="loadTemplate('template')" class="btn btn-secondary w-full text-left">
                            <i class="fas fa-image mr-2"></i>Template Match
                        </button>
                        <button onclick="loadTemplate('color')" class="btn btn-secondary w-full text-left">
                            <i class="fas fa-palette mr-2"></i>Color Detection
                        </button>
                        <button onclick="loadTemplate('monitor')" class="btn btn-secondary w-full text-left">
                            <i class="fas fa-eye mr-2"></i>Region Monitor
                        </button>
                    </div>
                </div>

                <!-- Help -->
                <div class="card p-6">
                    <h2 class="text-lg font-semibold mb-4">Help</h2>
                    <div class="text-sm text-slate-400 space-y-2">
                        <p><strong>OCR:</strong> Detect text on screen</p>
                        <p><strong>Template:</strong> Match image templates</p>
                        <p><strong>Color:</strong> Detect color regions</p>
                        <p><strong>Monitor:</strong> Watch for changes</p>
                        <p><strong>YOLO:</strong> Object detection</p>
                        <p><strong>Window:</strong> Monitor window titles</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_KEY = new URLSearchParams(window.location.search).get('api_key') || '';

        function addApiKey(url) {
            if (!API_KEY) return url;
            const separator = url.includes('?') ? '&' : '?';
            return url + separator + 'api_key=' + API_KEY;
        }

        // Mode selection
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const mode = btn.dataset.mode;
                document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
                document.querySelector(`.section[data-section="${mode}"]`)?.classList.add('active');

                updateJSON();
            });
        });

        // Update JSON on input change
        document.querySelectorAll('input, select').forEach(input => {
            input.addEventListener('input', updateJSON);
        });

        function getConfig() {
            const mode = document.querySelector('.mode-btn.active').dataset.mode;
            const config = { mode };

            // Common settings
            if (document.getElementById('pollingEnabled').checked) {
                config.polling = true;
                config.poll_interval = parseFloat(document.getElementById('pollInterval').value);
            }
            if (document.getElementById('adaptivePolling').checked) {
                config.adaptive_polling = true;
            }

            // Action
            config.action = document.getElementById('action').value || 'alt+o';
            config.action_delay = parseFloat(document.getElementById('actionDelay').value);

            // Mode-specific
            if (mode === 'ocr' || mode === 'fuzzy') {
                config.target_text = document.getElementById('targetText').value;
                config.text_case_sensitive = document.getElementById('caseSensitive').checked;
                if (mode === 'fuzzy') config.fuzzy_threshold = 0.8;
            }

            if (mode === 'template') {
                config.template_path = document.getElementById('templatePath').value;
                config.template_threshold = parseInt(document.getElementById('templateThreshold').value) / 100;
            }

            if (mode === 'color') {
                const b = document.getElementById('colorB').value;
                const g = document.getElementById('colorG').value;
                const r = document.getElementById('colorR').value;
                if (b && g && r) {
                    config.target_color = [parseInt(b), parseInt(g), parseInt(r)];
                }
                config.color_tolerance = parseInt(document.getElementById('colorTolerance').value);
            }

            if (mode === 'monitor') {
                const x = document.getElementById('regionX').value;
                const y = document.getElementById('regionY').value;
                const w = document.getElementById('regionW').value;
                const h = document.getElementById('regionH').value;
                if (x && y && w && h) {
                    config.region = [parseInt(x), parseInt(y), parseInt(w), parseInt(h)];
                }
                config.change_threshold = parseInt(document.getElementById('changeThreshold').value) / 100;
            }

            if (mode === 'yolo') {
                config.yolo_model = document.getElementById('yoloModel').value;
                config.yolo_confidence = parseInt(document.getElementById('yoloConfidence').value) / 100;
                const classes = document.getElementById('yoloClasses').value;
                if (classes) config.yolo_classes = classes.split(',').map(c => c.trim());
            }

            if (mode === 'window') {
                config.window_signal = document.getElementById('windowSignal').value || 'TRIGGER_CLAW';
                config.window_poll_interval = parseFloat(document.getElementById('windowPoll').value);
            }

            return config;
        }

        function updateJSON() {
            const config = getConfig();
            document.getElementById('jsonPreview').textContent = JSON.stringify(config, null, 2);
        }

        function loadTemplate(type) {
            const templates = {
                ocr: { mode: 'fuzzy', target_text: 'Submit', polling: true, poll_interval: 0.5 },
                template: { mode: 'template', template_path: '/path/to/template.png', template_threshold: 0.8 },
                color: { mode: 'color', target_color: [0, 255, 0], color_tolerance: 30 },
                monitor: { mode: 'monitor', region: [100, 100, 200, 200], change_threshold: 0.05 }
            };

            const template = templates[type];
            if (template) {
                // Update UI based on template
                document.querySelectorAll('.mode-btn').forEach(b => {
                    if (b.dataset.mode === template.mode) {
                        b.click();
                    }
                });

                // Set values
                Object.keys(template).forEach(key => {
                    const el = document.getElementById(key.charAt(0).toLowerCase() + key.slice(1).replace(/_([a-z])/g, g => g[1].toUpperCase()));
                    if (el) {
                        if (typeof template[key] === 'boolean') {
                            el.checked = template[key];
                        } else {
                            el.value = template[key];
                        }
                    }
                });

                updateJSON();
            }
        }

        async function loadConfig() {
            try {
                const resp = await fetch(addApiKey('/api/v1/config'));
                const config = await resp.json();
                console.log('Loaded config:', config);
                alert('Config loaded (check console)');
            } catch (e) {
                console.error(e);
                alert('Failed to load config');
            }
        }

        async function saveConfig() {
            const config = getConfig();
            try {
                const resp = await fetch(addApiKey('/api/v1/config'), {
                    method: 'PUT',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                if (resp.ok) {
                    alert('Config saved!');
                } else {
                    alert('Failed to save config');
                }
            } catch (e) {
                console.error(e);
                alert('Error saving config');
            }
        }

        function exportJSON() {
            const config = getConfig();
            const blob = new Blob([JSON.stringify(config, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'openclaw-config.json';
            a.click();
        }

        // Initialize
        updateJSON();

        // Slider value displays
        document.getElementById('templateThreshold').addEventListener('input', e => {
            document.getElementById('thresholdValue').textContent = e.target.value + '%';
        });
        document.getElementById('changeThreshold').addEventListener('input', e => {
            document.getElementById('changeValue').textContent = e.target.value + '%';
        });
        document.getElementById('yoloConfidence').addEventListener('input', e => {
            document.getElementById('confidenceValue').textContent = e.target.value + '%';
        });
    </script>
</body>
</html>
"""


class ConfigEditorHandler(BaseHTTPRequestHandler):
    """HTTP handler for config editor"""

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.debug(f"ConfigEditor: {args[0]}")

    def do_GET(self):
        """Handle GET requests"""
        path = self.path.split("?")[0]

        if path == "/config-editor" or path == "/config-editor.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(CONFIG_EDITOR_HTML.encode())
        else:
            self.send_error(404)


class ConfigEditorServer:
    """Config Editor HTTP Server"""

    def __init__(self, port: int = 8767):
        self.port = port
        self.server: Optional[HTTPServer] = None

    def start(self):
        """Start the config editor server"""
        self.server = HTTPServer(("", self.port), ConfigEditorHandler)
        logger.info(f"Config Editor server started on port {self.port}")
        logger.info(f"Open http://localhost:{self.port}/config-editor in your browser")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Config Editor server stopped")
            self.stop()

    def stop(self):
        """Stop the server"""
        if self.server:
            self.server.shutdown()
            self.server = None


__all__ = [
    "CONFIG_EDITOR_HTML",
    "ConfigEditorHandler",
    "ConfigEditorServer",
]
