"""
Environment Workspaces for OpenClaw

Multi-environment configuration management for development,
staging, and production deployments.
"""

import os
import json
import yaml
from typing import Any, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

from .logger import get_logger
from .config import ConfigManager

logger = get_logger("workspace")


class EnvironmentType(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class WorkspaceConfig:
    """Workspace-specific configuration"""
    name: str
    environment: EnvironmentType
    log_level: str = "INFO"
    debug_mode: bool = False

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Database settings
    db_enabled: bool = False
    db_type: str = "sqlite"  # sqlite, postgres, mysql
    db_path: str = "~/.openclaw/openclaw.db"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "openclaw"
    db_user: str = "openclaw"
    db_password: str = ""

    # Redis settings
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379"

    # Security
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_required: bool = False

    # Features
    hot_reload_enabled: bool = True
    profiling_enabled: bool = False

    # Rate limiting
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Monitoring
    metrics_enabled: bool = False
    metrics_port: int = 9090
    health_check_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "environment": self.environment.value,
            "log_level": self.log_level,
            "debug_mode": self.debug_mode,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "db_enabled": self.db_enabled,
            "db_type": self.db_type,
            "db_path": self.db_path,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "redis_enabled": self.redis_enabled,
            "redis_url": self.redis_url,
            "cors_enabled": self.cors_enabled,
            "cors_origins": self.cors_origins,
            "api_key_required": self.api_key_required,
            "hot_reload_enabled": self.hot_reload_enabled,
            "profiling_enabled": self.profiling_enabled,
            "rate_limit_enabled": self.rate_limit_enabled,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window": self.rate_limit_window,
            "metrics_enabled": self.metrics_enabled,
            "metrics_port": self.metrics_port,
            "health_check_enabled": self.health_check_enabled,
        }


# Default workspace configurations
DEFAULT_WORKSPACES = {
    "development": WorkspaceConfig(
        name="development",
        environment=EnvironmentType.DEVELOPMENT,
        log_level="DEBUG",
        debug_mode=True,
        api_port=8080,
        db_type="sqlite",
        db_path="~/.openclaw/openclaw_dev.db",
        redis_enabled=False,
        cors_enabled=True,
        cors_origins=["http://localhost:3000", "http://localhost:8080"],
        api_key_required=False,
        hot_reload_enabled=True,
        profiling_enabled=True,
        rate_limit_enabled=False,
        metrics_enabled=True,
    ),
    "staging": WorkspaceConfig(
        name="staging",
        environment=EnvironmentType.STAGING,
        log_level="INFO",
        debug_mode=False,
        api_port=8080,
        db_type="postgres",
        db_host="staging-db.internal",
        db_port=5432,
        db_name="openclaw_staging",
        redis_enabled=True,
        redis_url="redis://staging-redis.internal:6379",
        cors_enabled=True,
        cors_origins=["https://staging.openclaw.io"],
        api_key_required=True,
        hot_reload_enabled=False,
        profiling_enabled=True,
        rate_limit_enabled=True,
        rate_limit_requests=1000,
        metrics_enabled=True,
    ),
    "production": WorkspaceConfig(
        name="production",
        environment=EnvironmentType.PRODUCTION,
        log_level="WARNING",
        debug_mode=False,
        api_port=8080,
        db_type="postgres",
        db_host="prod-db.internal",
        db_port=5432,
        db_name="openclaw",
        redis_enabled=True,
        redis_url="redis://prod-redis.internal:6379",
        cors_enabled=True,
        cors_origins=["https://openclaw.io"],
        api_key_required=True,
        hot_reload_enabled=False,
        profiling_enabled=False,
        rate_limit_enabled=True,
        rate_limit_requests=100,
        metrics_enabled=True,
    ),
}


class WorkspaceManager:
    """
    Manages environment workspaces for OpenClaw.

    Supports switching between development, staging, and production
    configurations with environment-specific defaults.
    """

    _instance: Optional['WorkspaceManager'] = None
    _current_workspace: Optional[WorkspaceConfig] = None

    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = workspace_dir or os.path.expanduser("~/.openclaw/workspaces")
        self.workspaces: Dict[str, WorkspaceConfig] = {}
        self._config_manager: Optional[ConfigManager] = None

    @classmethod
    def get_instance(cls) -> 'WorkspaceManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_current_workspace(cls) -> Optional[WorkspaceConfig]:
        """Get current workspace"""
        return cls._current_workspace

    def initialize(self, config_manager: Optional[ConfigManager] = None) -> bool:
        """Initialize workspace manager"""
        self._config_manager = config_manager

        # Load default workspaces
        self.workspaces = deepcopy(DEFAULT_WORKSPACES)

        # Try to load custom workspaces from disk
        self._load_workspaces()

        # Detect and set environment
        env = self._detect_environment()
        self.set_workspace(env)

        logger.info(f"Workspace manager initialized: {env}")
        return True

    def _detect_environment(self) -> str:
        """Detect environment from environment variables"""
        # Check OPENCLAW_ENV environment variable
        env = os.environ.get("OPENCLAW_ENV", "").lower()
        if env in ["development", "dev", "staging", "stage", "production", "prod"]:
            return env

        # Check NODE_ENV (common in web frameworks)
        env = os.environ.get("NODE_ENV", "").lower()
        if env in ["development", "dev"]:
            return "development"
        if env in ["staging", "stage"]:
            return "staging"
        if env in ["production", "prod"]:
            return "production"

        # Default to development
        return "development"

    def _load_workspaces(self):
        """Load custom workspaces from disk"""
        workspace_path = Path(self.workspace_dir)

        if not workspace_path.exists():
            return

        for yaml_file in workspace_path.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data:
                        name = yaml_file.stem
                        self.workspaces[name] = WorkspaceConfig(**data)
                        logger.info(f"Loaded workspace: {name}")
            except Exception as e:
                logger.error(f"Failed to load workspace {yaml_file}: {e}")

    def save_workspace(self, workspace: WorkspaceConfig) -> bool:
        """Save workspace configuration to disk"""
        try:
            workspace_path = Path(self.workspace_dir)
            workspace_path.mkdir(parents=True, exist_ok=True)

            yaml_file = workspace_path / f"{workspace.name}.yaml"
            with open(yaml_file, 'w') as f:
                yaml.dump(workspace.to_dict(), f, default_flow_style=False)

            self.workspaces[workspace.name] = workspace
            logger.info(f"Saved workspace: {workspace.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save workspace: {e}")
            return False

    def set_workspace(self, name: str) -> bool:
        """Set current workspace"""
        if name not in self.workspaces:
            logger.warning(f"Workspace '{name}' not found, using development")
            name = "development"

        self._current_workspace = self.workspaces[name]

        # Apply workspace settings to environment
        self._apply_workspace_settings(self._current_workspace)

        logger.info(f"Switched to workspace: {name}")
        return True

    def _apply_workspace_settings(self, workspace: WorkspaceConfig) -> Dict[str, Any]:
        """Apply workspace settings - returns dict instead of modifying os.environ"""
        # Return settings as a dictionary instead of modifying global environment
        return {
            "OPENCLAW_LOG_LEVEL": workspace.log_level,
            "OPENCLAW_API_HOST": workspace.api_host,
            "OPENCLAW_API_PORT": str(workspace.api_port),
            "OPENCLAW_DB_ENABLED": str(workspace.db_enabled),
            "OPENCLAW_DB_TYPE": workspace.db_type,
            "OPENCLAW_DB_PATH": workspace.db_path,
            "OPENCLAW_REDIS_ENABLED": str(workspace.redis_enabled),
            "OPENCLAW_REDIS_URL": workspace.redis_url,
            "OPENCLAW_DEBUG": str(workspace.debug_mode).lower(),
        }

    def get_workspace_settings(self) -> Dict[str, Any]:
        """Get current workspace settings without modifying environment"""
        if self._current_workspace:
            return self._apply_workspace_settings(self._current_workspace)
        return {}

    def get_workspace(self, name: str) -> Optional[WorkspaceConfig]:
        """Get workspace by name"""
        return self.workspaces.get(name)

    def list_workspaces(self) -> List[str]:
        """List all available workspaces"""
        return list(self.workspaces.keys())

    def create_workspace(
        self,
        name: str,
        base_workspace: str = "development",
        **overrides
    ) -> WorkspaceConfig:
        """Create a new workspace from base"""
        if base_workspace not in self.workspaces:
            base_workspace = "development"

        base = deepcopy(self.workspaces[base_workspace])
        base.name = name

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(base, key):
                setattr(base, key, value)

        self.workspaces[name] = base
        return base


def get_workspace_manager() -> WorkspaceManager:
    """Get global workspace manager"""
    return WorkspaceManager.get_instance()


def get_current_workspace() -> Optional[WorkspaceConfig]:
    """Get current workspace configuration"""
    return WorkspaceManager.get_current_workspace()


def switch_workspace(name: str) -> bool:
    """Switch to a different workspace"""
    manager = get_workspace_manager()
    return manager.set_workspace(name)


def is_development() -> bool:
    """Check if running in development mode"""
    workspace = get_current_workspace()
    return workspace.environment == EnvironmentType.DEVELOPMENT if workspace else True


def is_staging() -> bool:
    """Check if running in staging mode"""
    workspace = get_current_workspace()
    return workspace.environment == EnvironmentType.STAGING if workspace else False


def is_production() -> bool:
    """Check if running in production mode"""
    workspace = get_current_workspace()
    return workspace.environment == EnvironmentType.PRODUCTION if workspace else False


__all__ = [
    "EnvironmentType",
    "WorkspaceConfig",
    "WorkspaceManager",
    "get_workspace_manager",
    "get_current_workspace",
    "switch_workspace",
    "is_development",
    "is_staging",
    "is_production",
]
