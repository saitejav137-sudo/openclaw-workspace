"""
Multi-User Support with Role-Based Access Control (RBAC)

Supports multiple users with different roles:
- admin: Full access to all features
- operator: Can manage triggers and automations
- viewer: Read-only access to dashboard and stats
"""

import hashlib
import secrets
import time
import threading
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, parse_qs

from ..core.logger import get_logger
from ..storage import DatabaseManager

logger = get_logger("auth")


class UserRole(Enum):
    """User roles"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


@dataclass
class User:
    """User account"""
    id: str
    username: str
    email: Optional[str]
    password_hash: str
    role: UserRole
    api_key: Optional[str]
    created_at: float
    last_login: Optional[float]
    is_active: bool = True
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "api_key": self.api_key[:8] + "..." if self.api_key else None,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active
        }


@dataclass
class Session:
    """User session"""
    id: str
    user_id: str
    token: str
    created_at: float
    expires_at: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class Permission(Enum):
    """Permission types"""
    # Triggers
    TRIGGER_CREATE = "trigger:create"
    TRIGGER_READ = "trigger:read"
    TRIGGER_UPDATE = "trigger:update"
    TRIGGER_DELETE = "trigger:delete"
    TRIGGER_EXECUTE = "trigger:execute"

    # Config
    CONFIG_READ = "config:read"
    CONFIG_UPDATE = "config:update"

    # Users
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"

    # Stats
    STATS_READ = "stats:read"

    # Automation
    AUTOMATION_EXECUTE = "automation:execute"

    # Admin
    ADMIN_ALL = "admin:all"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: {
        Permission.TRIGGER_CREATE,
        Permission.TRIGGER_READ,
        Permission.TRIGGER_UPDATE,
        Permission.TRIGGER_DELETE,
        Permission.TRIGGER_EXECUTE,
        Permission.CONFIG_READ,
        Permission.CONFIG_UPDATE,
        Permission.USER_CREATE,
        Permission.USER_READ,
        Permission.USER_UPDATE,
        Permission.USER_DELETE,
        Permission.STATS_READ,
        Permission.AUTOMATION_EXECUTE,
        Permission.ADMIN_ALL,
    },
    UserRole.OPERATOR: {
        Permission.TRIGGER_CREATE,
        Permission.TRIGGER_READ,
        Permission.TRIGGER_UPDATE,
        Permission.TRIGGER_EXECUTE,
        Permission.CONFIG_READ,
        Permission.STATS_READ,
        Permission.AUTOMATION_EXECUTE,
    },
    UserRole.VIEWER: {
        Permission.TRIGGER_READ,
        Permission.CONFIG_READ,
        Permission.STATS_READ,
    },
}


class AuthManager:
    """
    Authentication and Authorization Manager
    """

    def __init__(self, db_path: Optional[str] = None):
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._sessions_lock = threading.Lock()
        self._session_timeout = 3600 * 24  # 24 hours

        # Load users from database if available
        if db_path:
            self._load_users(db_path)
        else:
            # Create default admin user
            self._create_default_users()

    def _create_default_users(self):
        """Create default users"""
        # Admin user
        admin = self.create_user(
            username="admin",
            password="admin123",
            role=UserRole.ADMIN,
            email="admin@localhost"
        )

        # Operator user
        operator = self.create_user(
            username="operator",
            password="operator123",
            role=UserRole.OPERATOR,
            email="operator@localhost"
        )

        # Viewer user
        viewer = self.create_user(
            username="viewer",
            password="viewer123",
            role=UserRole.VIEWER,
            email="viewer@localhost"
        )

        logger.info("Default users created: admin, operator, viewer")

    def _load_users(self, db_path: str):
        """Load users from database"""
        # Would load from database
        self._create_default_users()

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        hash_value = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            100000
        )
        return f"{salt}:{hash_value.hex()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password"""
        try:
            salt, hash_value = password_hash.split(":")
            expected = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode(),
                salt.encode(),
                100000
            )
            return expected.hex() == hash_value
        except Exception:
            return False

    def _generate_api_key(self) -> str:
        """Generate API key"""
        return secrets.token_urlsafe(32)

    def _generate_session_id(self) -> str:
        """Generate session ID"""
        return secrets.token_urlsafe(32)

    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole = UserRole.VIEWER,
        email: Optional[str] = None
    ) -> User:
        """Create a new user"""
        # Check if username exists
        for user in self._users.values():
            if user.username == username:
                raise ValueError(f"Username {username} already exists")

        user_id = secrets.token_urlsafe(8)
        api_key = self._generate_api_key()

        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            role=role,
            api_key=api_key,
            created_at=time.time(),
            last_login=None,
            is_active=True
        )

        self._users[user_id] = user
        logger.info(f"User created: {username} ({role.value})")

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in self._users.values():
            if user.username == username:
                return user
        return None

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        for user in self._users.values():
            if user.api_key == api_key:
                return user
        return None

    def list_users(self) -> List[User]:
        """List all users"""
        return list(self._users.values())

    def update_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None
    ) -> Optional[User]:
        """Update user"""
        user = self._users.get(user_id)
        if not user:
            return None

        if email is not None:
            user.email = email
        if role is not None:
            user.role = role
        if is_active is not None:
            user.is_active = is_active

        logger.info(f"User updated: {user.username}")
        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        user = self._users.get(user_id)
        if not user:
            return False

        # Prevent deleting last admin
        if user.role == UserRole.ADMIN:
            admins = [u for u in self._users.values() if u.role == UserRole.ADMIN]
            if len(admins) <= 1:
                raise ValueError("Cannot delete last admin user")

        del self._users[user_id]
        logger.info(f"User deleted: {user.username}")
        return True

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user"""
        user = self.get_user_by_username(username)

        if not user:
            return None

        if not user.is_active:
            return None

        if not self._verify_password(password, user.password_hash):
            return None

        user.last_login = time.time()
        return user

    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key"""
        user = self.get_user_by_api_key(api_key)

        if not user or not user.is_active:
            return None

        user.last_login = time.time()
        return user

    def create_session(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[Session]:
        """Create session"""
        user = self._users.get(user_id)
        if not user or not user.is_active:
            return None

        session_id = self._generate_session_id()
        token = secrets.token_urlsafe(32)

        session = Session(
            id=session_id,
            user_id=user_id,
            token=token,
            created_at=time.time(),
            expires_at=time.time() + self._session_timeout,
            ip_address=ip_address,
            user_agent=user_agent
        )

        with self._sessions_lock:
            self._sessions[session_id] = session

        logger.info(f"Session created for user: {user.username}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session"""
        with self._sessions_lock:
            session = self._sessions.get(session_id)

            if session and session.expires_at > time.time():
                return session

            # Clean up expired session
            if session:
                del self._sessions[session_id]

        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        with self._sessions_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has permission"""
        if not user.is_active:
            return False

        # Admin has all permissions
        if user.role == UserRole.ADMIN:
            return True

        return permission in ROLE_PERMISSIONS.get(user.role, set())

    def require_permission(self, user: User, permission: Permission):
        """Raise exception if user lacks permission"""
        if not self.has_permission(user, permission):
            raise PermissionError(f"User lacks permission: {permission.value}")

    def require_role(self, user: User, role: UserRole):
        """Raise exception if user doesn't have required role"""
        if user.role != role and user.role != UserRole.ADMIN:
            raise PermissionError(f"User requires role: {role.value}")


class RBACMiddleware:
    """
    RBAC Middleware for HTTP requests
    """

    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager

    def extract_user(self, request) -> Optional[User]:
        """Extract user from request"""
        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
            return self.auth_manager.authenticate_api_key(api_key)

        # Check query param
        if hasattr(request, "path"):
            parsed = urlparse(request.path)
            params = parse_qs(parsed.query)
            if "api_key" in params:
                api_key = params["api_key"][0]
                return self.auth_manager.authenticate_api_key(api_key)

        return None

    def require_auth(self, request) -> User:
        """Require authentication"""
        user = self.extract_user(request)
        if not user:
            raise PermissionError("Authentication required")
        return user

    def require_permission(self, request, permission: Permission) -> User:
        """Require specific permission"""
        user = self.require_auth(request)
        self.auth_manager.require_permission(user, permission)
        return user


# Global auth manager
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> Optional[AuthManager]:
    """Get global auth manager"""
    return _auth_manager


def init_auth_manager(db_path: Optional[str] = None) -> AuthManager:
    """Initialize global auth manager"""
    global _auth_manager
    _auth_manager = AuthManager(db_path)
    return _auth_manager


__all__ = [
    "User",
    "UserRole",
    "Session",
    "Permission",
    "AuthManager",
    "RBACMiddleware",
    "ROLE_PERMISSIONS",
    "get_auth_manager",
    "init_auth_manager",
]
