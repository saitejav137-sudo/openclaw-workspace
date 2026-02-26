"""
Code Generation for OpenClaw Agents

Integrates Qwen3-Coder and other code generation models.
Supports multiple providers and local inference.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .logger import get_logger
from .blip import get_minimax_ai

logger = get_logger("code_gen")


class CodeModel(Enum):
    """Supported code generation models"""
    QWEN_CODER = "qwen3-coder"
    CODE_LLAMA = "codellama"
    STABLE_CODE = "stable-code"
    DEEPSEEK_CODER = "deepseek-coder"


class Provider(Enum):
    """Inference providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MINIMAX = "minimax"
    LOCAL = "local"


@dataclass
class CodeGenerationRequest:
    """Code generation request"""
    prompt: str
    language: str = "python"
    max_tokens: int = 2048
    temperature: float = 0.7
    model: CodeModel = CodeModel.QWEN_CODER
    context: Optional[str] = None


@dataclass
class CodeGenerationResult:
    """Code generation result"""
    code: str
    language: str
    model: str
    generation_time: float
    success: bool
    error: Optional[str] = None


class CodeGenerator:
    """
    Code generation engine for agents.

    Supports:
    - Qwen3-Coder (recommended for 2026)
    - Code Llama
    - Stable Code
    - DeepSeek Coder

    Providers:
    - Ollama (local)
    - OpenAI
    - Anthropic
    - MiniMax
    """

    def __init__(
        self,
        provider: Provider = Provider.OLLAMA,
        model: CodeModel = CodeModel.QWEN_CODER,
        api_key: str = None,
        base_url: str = None
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Get appropriate client based on provider"""
        if self.provider == Provider.OLLAMA:
            try:
                import ollama
                return ollama
            except ImportError:
                logger.warning("ollama not installed")

        elif self.provider == Provider.OPENAI:
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                logger.warning("openai not installed")

        elif self.provider == Provider.MINIMAX:
            return get_minimax_ai(self.api_key)

        return None

    def generate(
        self,
        prompt: str,
        language: str = "python",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        context: str = None
    ) -> CodeGenerationResult:
        """Generate code"""
        start_time = time.time()

        # Build full prompt
        full_prompt = self._build_prompt(prompt, language, context)

        try:
            if self.provider == Provider.OLLAMA:
                code = self._generate_ollama(full_prompt, language)
            elif self.provider == Provider.OPENAI:
                code = self._generate_openai(full_prompt, max_tokens, temperature)
            elif self.provider == Provider.MINIMAX:
                code = self._generate_minimax(full_prompt)
            else:
                code = self._generate_local(full_prompt)

            return CodeGenerationResult(
                code=code,
                language=language,
                model=self.model.value,
                generation_time=time.time() - start_time,
                success=True
            )

        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return CodeGenerationResult(
                code="",
                language=language,
                model=self.model.value,
                generation_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

    def _build_prompt(self, prompt: str, language: str, context: Optional[str]) -> str:
        """Build full prompt with context"""
        parts = [
            f"You are an expert {language} programmer.",
            ""
        ]

        if context:
            parts.append(f"Context:\n{context}")
            parts.append("")

        parts.append(f"Task:\n{prompt}")
        parts.append("")

        if language == "python":
            parts.append("Write clean, well-documented Python code. Use type hints where appropriate.")
        elif language == "javascript":
            parts.append("Write modern ES6+ JavaScript code.")
        elif language == "bash":
            parts.append("Write efficient bash scripts.")

        parts.append("\nCode:")

        return "\n".join(parts)

    def _generate_ollama(self, prompt: str, language: str) -> str:
        """Generate using Ollama"""
        model_map = {
            CodeModel.QWEN_CODER: "qwen2.5-coder",
            CodeModel.CODE_LLAMA: "codellama",
            CodeModel.STABLE_CODE: "stable-code",
        }

        model_name = model_map.get(self.model, "qwen2.5-coder")

        import ollama
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": 0.7,
                "num_predict": 2048,
            }
        )

        return response["response"]

    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using OpenAI"""
        client = self._get_client()
        if not client:
            raise RuntimeError("OpenAI client not available")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content

    def _generate_minimax(self, prompt: str) -> str:
        """Generate using MiniMax"""
        client = self._get_client()
        if not client:
            raise RuntimeError("MiniMax client not available")

        messages = [
            {"role": "system", "content": "You are an expert programmer."},
            {"role": "user", "content": prompt}
        ]

        return client.chat(messages) or ""

    def _generate_local(self, prompt: str) -> str:
        """Fallback local generation"""
        return "# Local generation not implemented. Install ollama or configure API."

    def generate_automation_code(
        self,
        task_description: str,
        context: Dict[str, Any] = None
    ) -> CodeGenerationResult:
        """Generate automation code for OpenClaw"""
        prompt = f"""
Task: {task_description}

Generate Python automation code using OpenClaw framework.
Available modules:
- openclaw.core.vision: ScreenCapture, OCREngine, VisionEngine
- openclaw.core.automation: click, type_text, press, get_screen_size
- openclaw.core.actions: TriggerAction

Example:
```python
from openclaw.core.automation import click, type_text
from openclaw.core.vision import ScreenCapture

# Capture screen
img = ScreenCapture.capture_full()

# Click at position
click(100, 200)

# Type text
type_text("Hello World")
```

Now generate code for: {task_description}
"""

        return self.generate(prompt, language="python")


# Code analysis tools

def analyze_code(code: str, language: str = "python") -> Dict[str, Any]:
    """Analyze code for issues and improvements"""
    issues = []
    suggestions = []

    # Simple analysis - can extend with AST
    if len(code) > 1000:
        suggestions.append("Consider breaking into smaller functions")

    if "except:" in code:
        issues.append("Avoid bare except clauses")

    if "time.sleep" in code:
        suggestions.append("Consider using async instead of blocking sleep")

    return {
        "issues": issues,
        "suggestions": suggestions,
        "line_count": len(code.split("\n")),
        "language": language
    }


# Global code generator
_code_generator: Optional[CodeGenerator] = None


def get_code_generator(
    provider: Provider = Provider.OLLAMA,
    model: CodeModel = CodeModel.QWEN_CODER
) -> CodeGenerator:
    """Get global code generator"""
    global _code_generator
    if _code_generator is None:
        _code_generator = CodeGenerator(provider, model)
    return _code_generator


def generate_code(
    prompt: str,
    language: str = "python"
) -> CodeGenerationResult:
    """Quick code generation"""
    return get_code_generator().generate(prompt, language)


def generate_automation(
    task: str,
    context: Dict = None
) -> CodeGenerationResult:
    """Quick automation code generation"""
    return get_code_generator().generate_automation_code(task, context)


__all__ = [
    "CodeModel",
    "Provider",
    "CodeGenerationRequest",
    "CodeGenerationResult",
    "CodeGenerator",
    "analyze_code",
    "get_code_generator",
    "generate_code",
    "generate_automation",
]
