"""
Sandboxed Code Execution for OpenClaw

Safely execute dynamically generated code:
- Subprocess sandbox with resource limits
- Timeout enforcement
- Output capture and parsing
- Support for Python, Bash, and Node.js
"""

import time
import subprocess
import tempfile
import os
import shutil
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .logger import get_logger

logger = get_logger("sandbox")


class Language(Enum):
    """Supported execution languages."""
    PYTHON = "python"
    BASH = "bash"
    NODEJS = "nodejs"


@dataclass
class ExecutionResult:
    """Result of sandbox execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    duration: float = 0.0
    timed_out: bool = False
    error: Optional[str] = None
    language: Language = Language.PYTHON


@dataclass
class SandboxConfig:
    """Sandbox configuration."""
    timeout_seconds: float = 30.0
    max_output_bytes: int = 1024 * 1024  # 1MB
    max_memory_mb: int = 512
    allowed_languages: List[Language] = field(
        default_factory=lambda: [Language.PYTHON, Language.BASH]
    )
    working_dir: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)
    # Safety flags
    allow_network: bool = False
    allow_file_write: bool = True
    restricted_imports: List[str] = field(
        default_factory=lambda: ["os.system", "subprocess.call", "eval", "exec"]
    )


class CodeValidator:
    """Validates code before execution for safety."""

    # Dangerous patterns to block
    BLOCKED_PATTERNS = [
        "import shutil",
        "os.remove",
        "os.rmdir",
        "os.unlink",
        "shutil.rmtree",
        "__import__",
        "globals()",
        "locals()",
        "compile(",
    ]

    def validate(self, code: str, language: Language, config: SandboxConfig) -> Dict:
        """
        Validate code for safety.
        Returns {"safe": bool, "issues": [str]}
        """
        issues = []

        if language == Language.PYTHON:
            issues.extend(self._check_python(code, config))
        elif language == Language.BASH:
            issues.extend(self._check_bash(code, config))

        return {
            "safe": len(issues) == 0,
            "issues": issues
        }

    def _check_python(self, code: str, config: SandboxConfig) -> List[str]:
        """Check Python code for safety issues."""
        issues = []

        for pattern in self.BLOCKED_PATTERNS:
            if pattern in code:
                issues.append(f"Blocked pattern: {pattern}")

        for restricted in config.restricted_imports:
            if restricted in code:
                issues.append(f"Restricted import/call: {restricted}")

        if not config.allow_network:
            net_patterns = ["urllib", "requests", "http.client", "socket"]
            for pat in net_patterns:
                if pat in code:
                    issues.append(f"Network access not allowed: {pat}")

        return issues

    def _check_bash(self, code: str, config: SandboxConfig) -> List[str]:
        """Check bash code for safety issues."""
        issues = []

        dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){:|:&};:", "chmod 777 /"]
        for danger in dangerous:
            if danger in code:
                issues.append(f"Dangerous command: {danger}")

        if not config.allow_network:
            net_cmds = ["curl", "wget", "nc ", "netcat"]
            for cmd in net_cmds:
                if cmd in code:
                    issues.append(f"Network access not allowed: {cmd}")

        return issues


class Sandbox:
    """
    Sandboxed code execution environment.

    Usage:
        sandbox = Sandbox()

        # Execute Python code
        result = sandbox.execute('''
            data = [1, 2, 3, 4, 5]
            print(f"Sum: {sum(data)}")
            print(f"Mean: {sum(data)/len(data)}")
        ''')

        print(result.stdout)  # "Sum: 15\nMean: 3.0\n"
    """

    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self.validator = CodeValidator()
        self._execution_count = 0
        self._execution_history: List[ExecutionResult] = []
        self._lock = threading.Lock()

        # Create working directory
        if not self.config.working_dir:
            self.config.working_dir = tempfile.mkdtemp(prefix="openclaw_sandbox_")

    def execute(
        self,
        code: str,
        language: Language = Language.PYTHON,
        timeout: float = None,
        env: Dict[str, str] = None
    ) -> ExecutionResult:
        """
        Execute code in a sandboxed environment.
        """
        if language not in self.config.allowed_languages:
            return ExecutionResult(
                success=False,
                error=f"Language {language.value} not allowed",
                language=language
            )

        # Validate code
        validation = self.validator.validate(code, language, self.config)
        if not validation["safe"]:
            return ExecutionResult(
                success=False,
                error=f"Code validation failed: {'; '.join(validation['issues'])}",
                language=language
            )

        timeout = timeout or self.config.timeout_seconds

        # Execute based on language
        if language == Language.PYTHON:
            result = self._execute_python(code, timeout, env)
        elif language == Language.BASH:
            result = self._execute_bash(code, timeout, env)
        elif language == Language.NODEJS:
            result = self._execute_nodejs(code, timeout, env)
        else:
            result = ExecutionResult(
                success=False,
                error=f"Unsupported language: {language.value}",
                language=language
            )

        result.language = language

        with self._lock:
            self._execution_count += 1
            self._execution_history.append(result)

        return result

    def _execute_python(
        self,
        code: str,
        timeout: float,
        env: Dict = None
    ) -> ExecutionResult:
        """Execute Python code."""
        # Write code to temp file
        script_path = os.path.join(self.config.working_dir, "script.py")
        try:
            with open(script_path, 'w') as f:
                f.write(code)

            exec_env = os.environ.copy()
            exec_env.update(self.config.env_vars)
            if env:
                exec_env.update(env)

            start = time.time()
            proc = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.config.working_dir,
                env=exec_env
            )

            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=proc.stdout[:self.config.max_output_bytes],
                stderr=proc.stderr[:self.config.max_output_bytes],
                return_code=proc.returncode,
                duration=time.time() - start
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
                timed_out=True,
                duration=timeout
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e)
            )
        finally:
            # Cleanup script
            try:
                os.remove(script_path)
            except OSError:
                pass

    def _execute_bash(
        self,
        code: str,
        timeout: float,
        env: Dict = None
    ) -> ExecutionResult:
        """Execute Bash code."""
        script_path = os.path.join(self.config.working_dir, "script.sh")
        try:
            with open(script_path, 'w') as f:
                f.write("#!/bin/bash\nset -e\n" + code)
            os.chmod(script_path, 0o755)

            exec_env = os.environ.copy()
            exec_env.update(self.config.env_vars)
            if env:
                exec_env.update(env)

            start = time.time()
            proc = subprocess.run(
                ["bash", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.config.working_dir,
                env=exec_env
            )

            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=proc.stdout[:self.config.max_output_bytes],
                stderr=proc.stderr[:self.config.max_output_bytes],
                return_code=proc.returncode,
                duration=time.time() - start
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
                timed_out=True,
                duration=timeout
            )
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
        finally:
            try:
                os.remove(script_path)
            except OSError:
                pass

    def _execute_nodejs(
        self,
        code: str,
        timeout: float,
        env: Dict = None
    ) -> ExecutionResult:
        """Execute Node.js code."""
        script_path = os.path.join(self.config.working_dir, "script.js")
        try:
            with open(script_path, 'w') as f:
                f.write(code)

            exec_env = os.environ.copy()
            exec_env.update(self.config.env_vars)
            if env:
                exec_env.update(env)

            start = time.time()
            proc = subprocess.run(
                ["node", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.config.working_dir,
                env=exec_env
            )

            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=proc.stdout[:self.config.max_output_bytes],
                stderr=proc.stderr[:self.config.max_output_bytes],
                return_code=proc.returncode,
                duration=time.time() - start
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
                timed_out=True,
                duration=timeout
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                error="Node.js not installed"
            )
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
        finally:
            try:
                os.remove(script_path)
            except OSError:
                pass

    def cleanup(self):
        """Remove sandbox working directory."""
        try:
            if os.path.exists(self.config.working_dir):
                shutil.rmtree(self.config.working_dir)
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        with self._lock:
            total = self._execution_count
            successes = sum(1 for r in self._execution_history if r.success)
            timeouts = sum(1 for r in self._execution_history if r.timed_out)

            return {
                "total_executions": total,
                "successes": successes,
                "failures": total - successes,
                "timeouts": timeouts,
                "success_rate": round(successes / total, 3) if total > 0 else 0
            }


# ============== Global Instance ==============

_sandbox: Optional[Sandbox] = None


def get_sandbox(config: SandboxConfig = None) -> Sandbox:
    """Get global sandbox."""
    global _sandbox
    if _sandbox is None:
        _sandbox = Sandbox(config)
    return _sandbox


def run_python(code: str, timeout: float = 30.0) -> ExecutionResult:
    """Quick-run Python code in sandbox."""
    return get_sandbox().execute(code, Language.PYTHON, timeout)


def run_bash(code: str, timeout: float = 30.0) -> ExecutionResult:
    """Quick-run Bash code in sandbox."""
    return get_sandbox().execute(code, Language.BASH, timeout)


__all__ = [
    "Language",
    "ExecutionResult",
    "SandboxConfig",
    "CodeValidator",
    "Sandbox",
    "get_sandbox",
    "run_python",
    "run_bash",
]
