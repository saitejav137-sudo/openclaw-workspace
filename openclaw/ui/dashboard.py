"""
Modern Dashboard for OpenClaw

Enhanced HTML/JavaScript dashboard with real-time updates.
"""

import json
import time
from typing import Dict, Any, Optional


MODERN_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenClaw Vision Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { font-family: 'Inter', sans-serif; }
        .gradient-bg {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .status-dot.online { background: #10b981; }
        .status-dot.offline { background: #ef4444; }
        .status-dot.warning { background: #f59e0b; }
    </style>
</head>
<body class="gradient-bg min-h-screen text-white">
    <!-- Header -->
    <header class="border-b border-white/10">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-green-400 to-blue-500 rounded-lg flex items-center justify-center">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                        </svg>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold">OpenClaw</h1>
                        <p class="text-xs text-gray-400">Vision Automation Platform</p>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <div class="flex items-center space-x-2">
                        <span id="statusDot" class="status-dot online"></span>
                        <span id="statusText" class="text-sm text-gray-300">Connected</span>
                    </div>
                    <div class="text-right">
                        <p class="text-xs text-gray-400">Version</p>
                        <p class="text-sm font-semibold">2.0.0</p>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container mx-auto px-6 py-8">
        <!-- Stats Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <!-- Total Triggers -->
            <div class="card rounded-xl p-6">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-gray-400 text-sm">Total Triggers</h3>
                    <span class="text-2xl">📊</span>
                </div>
                <p id="totalTriggers" class="text-3xl font-bold">0</p>
                <p class="text-xs text-gray-500 mt-2">All time</p>
            </div>

            <!-- Success Rate -->
            <div class="card rounded-xl p-6">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-gray-400 text-sm">Success Rate</h3>
                    <span class="text-2xl">✅</span>
                </div>
                <p id="successRate" class="text-3xl font-bold">0%</p>
                <p class="text-xs text-gray-500 mt-2">Last 24 hours</p>
            </div>

            <!-- Active Jobs -->
            <div class="card rounded-xl p-6">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-gray-400 text-sm">Active Jobs</h3>
                    <span class="text-2xl">⏰</span>
                </div>
                <p id="activeJobs" class="text-3xl font-bold">0</p>
                <p class="text-xs text-gray-500 mt-2">Scheduled</p>
            </div>

            <!-- Detection Mode -->
            <div class="card rounded-xl p-6">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-gray-400 text-sm">Detection Mode</h3>
                    <span class="text-2xl">👁️</span>
                </div>
                <p id="detectionMode" class="text-xl font-bold">OCR</p>
                <p id="detectionStatus" class="text-xs text-green-400 mt-2">Active</p>
            </div>
        </div>

        <!-- Action Bar -->
        <div class="card rounded-xl p-6 mb-8">
            <div class="flex flex-wrap items-center justify-between gap-4">
                <div class="flex items-center space-x-4">
                    <button onclick="triggerManual()" class="bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        </svg>
                        <span>Trigger Now</span>
                    </button>
                    <button onclick="refreshStats()" class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg font-medium transition-colors">
                        Refresh
                    </button>
                </div>
                <div class="flex items-center space-x-4">
                    <label class="flex items-center space-x-2">
                        <span class="text-sm text-gray-400">Auto-refresh:</span>
                        <input type="checkbox" id="autoRefresh" checked class="w-4 h-4 rounded">
                    </label>
                </div>
            </div>
        </div>

        <!-- Two Column Layout -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Left Column - Stream & Controls -->
            <div class="lg:col-span-2 space-y-8">
                <!-- Stream -->
                <div class="card rounded-xl overflow-hidden">
                    <div class="p-4 border-b border-white/10 flex items-center justify-between">
                        <h2 class="font-semibold">Live Screen</h2>
                        <span id="streamStatus" class="text-xs text-gray-400">● Live</span>
                    </div>
                    <div class="aspect-video bg-black flex items-center justify-center">
                        <img id="streamFrame" src="/api/screenshot?api_key=" + getApiKey() alt="Screen" class="max-w-full max-h-full object-contain">
                    </div>
                </div>

                <!-- Recent Activity -->
                <div class="card rounded-xl p-6">
                    <h2 class="font-semibold mb-4">Recent Activity</h2>
                    <div id="activityLog" class="space-y-3 max-h-64 overflow-y-auto">
                        <p class="text-gray-500 text-sm">No activity yet</p>
                    </div>
                </div>
            </div>

            <!-- Right Column - Info -->
            <div class="space-y-8">
                <!-- Quick Actions -->
                <div class="card rounded-xl p-6">
                    <h2 class="font-semibold mb-4">Quick Actions</h2>
                    <div class="space-y-3">
                        <button onclick="showConfig()" class="w-full text-left px-4 py-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors flex items-center space-x-3">
                            <span>⚙️</span>
                            <span>Configuration</span>
                        </button>
                        <button onclick="showMetrics()" class="w-full text-left px-4 py-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors flex items-center space-x-3">
                            <span>📈</span>
                            <span>View Metrics</span>
                        </button>
                        <button onclick="showAlerts()" class="w-full text-left px-4 py-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors flex items-center space-x-3">
                            <span>🔔</span>
                            <span>Alert History</span>
                        </button>
                    </div>
                </div>

                <!-- System Info -->
                <div class="card rounded-xl p-6">
                    <h2 class="font-semibold mb-4">System Info</h2>
                    <div class="space-y-3 text-sm">
                        <div class="flex justify-between">
                            <span class="text-gray-400">API Status</span>
                            <span id="apiStatus" class="text-green-400">Online</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Rate Limit</span>
                            <span>60/min</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Auth</span>
                            <span id="authStatus">Enabled</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let apiKey = '';

        function getApiKey() {
            const params = new URLSearchParams(window.location.search);
            return params.get('api_key') || '';
        }

        function updateStatus(online) {
            const dot = document.getElementById('statusDot');
            const text = document.getElementById('statusText');
            if (online) {
                dot.className = 'status-dot online';
                text.textContent = 'Connected';
            } else {
                dot.className = 'status-dot offline';
                text.textContent = 'Disconnected';
            }
        }

        async function fetchStats() {
            try {
                const res = await fetch('/api/stats?api_key=' + getApiKey());
                const data = await res.json();

                document.getElementById('totalTriggers').textContent = data.total || 0;
                document.getElementById('successRate').textContent = (data.success_rate || 0).toFixed(1) + '%';

                updateStatus(true);
            } catch (e) {
                updateStatus(false);
            }
        }

        async function triggerManual() {
            try {
                const res = await fetch('/?api_key=' + getApiKey());
                const data = await res.json();
                console.log('Trigger:', data);
                addActivity('Manual trigger executed', data.triggered ? 'success' : 'info');
                refreshStats();
            } catch (e) {
                console.error('Trigger error:', e);
            }
        }

        async function refreshStats() {
            await fetchStats();
            // Refresh screenshot
            document.getElementById('streamFrame').src = '/api/screenshot?api_key=' + getApiKey() + '&t=' + Date.now();
        }

        function addActivity(message, type = 'info') {
            const log = document.getElementById('activityLog');
            const time = new Date().toLocaleTimeString();
            const colors = {
                success: 'text-green-400',
                error: 'text-red-400',
                info: 'text-blue-400',
                warning: 'text-yellow-400'
            };

            const entry = document.createElement('div');
            entry.className = 'flex items-center space-x-3 text-sm';
            entry.innerHTML = `<span class="text-gray-500">${time}</span><span class="${colors[type] || colors.info}">${message}</span>`;

            log.insertBefore(entry, log.firstChild);

            // Keep only last 20 entries
            while (log.children.length > 20) {
                log.removeChild(log.lastChild);
            }
        }

        function showConfig() {
            alert('Configuration panel coming soon!');
        }

        function showMetrics() {
            alert('Metrics panel coming soon!');
        }

        function showAlerts() {
            alert('Alerts panel coming soon!');
        }

        // Auto-refresh
        document.getElementById('autoRefresh').addEventListener('change', function(e) {
            if (e.target.checked) {
                startAutoRefresh();
            } else {
                stopAutoRefresh();
            }
        });

        let refreshInterval;

        function startAutoRefresh() {
            refreshInterval = setInterval(refreshStats, 5000);
        }

        function stopAutoRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        }

        // Initialize
        refreshStats();
        startAutoRefresh();
        addActivity('Dashboard loaded', 'success');
    </script>
</body>
</html>
"""


def get_modern_dashboard_html(api_key: str = "") -> str:
    """Get modern dashboard HTML with optional API key"""
    return MODERN_DASHBOARD_HTML


# Export
__all__ = ["get_modern_dashboard_html", "MODERN_DASHBOARD_HTML"]
