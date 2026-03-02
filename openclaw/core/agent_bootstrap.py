"""
Agent Bootstrap for OpenClaw

Wires up the dormant agent infrastructure:
- Registers AI-powered tools in the ToolRegistry
- Sets up pre-configured agents in the Swarm
- Registers handler functions in the Orchestrator
- Provides a single bootstrap_agents() entry point
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

from .logger import get_logger
from .agent_tools import Tool, ToolRegistry, get_tool_registry, ToolResult, ToolCallStatus

logger = get_logger("agent_bootstrap")


# ============== AI Helper ==============

def _get_api_key() -> Optional[str]:
    """Get MiniMax API key from environment or credentials file."""
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        key_path = os.path.expanduser("~/.openclaw/credentials/keys.json")
        if os.path.exists(key_path):
            try:
                with open(key_path) as f:
                    d = json.load(f)
                api_key = d.get("providers", {}).get("minimax", {}).get("default", {}).get("api_key")
            except Exception:
                pass
    return api_key


def _call_llm(prompt: str, system: str = "You are a helpful AI assistant.", max_tokens: int = 1024) -> str:
    """Call MiniMax LLM and return text response."""
    import requests

    api_key = _get_api_key()
    if not api_key:
        return "Error: AI API key not configured."

    url = "https://api.minimax.io/anthropic/v1/messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    data = {
        "model": "MiniMax-M2.5-Lightning",
        "max_tokens": max_tokens,
        "stream": False,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            # Try to grab the final text block
            for block in result.get("content", []):
                if block.get("type") == "text":
                    return block.get("text", "No response.")
                    
            # Fallback: model might have only produced thinking blocks due to max_tokens limits
            thinking = []
            for block in result.get("content", []):
                if block.get("type") == "thinking" and "thinking" in block:
                    thinking.append(block["thinking"])
            if thinking:
                logger.warning(f"AI response contained only thinking blocks (token limit reached?)")
                return "\n".join(thinking)
                
            return f"AI Error: Empty response body (HTTP 200)"
        return f"AI Error: HTTP {response.status_code} - {response.text}"
    except Exception as e:
        return f"AI Error: {e}"


# ============== LLM-Powered Intelligence ==============

def _llm_decompose_task(task: str, available_capabilities: List[str]) -> List[Dict]:
    """
    Use the LLM to intelligently decompose any task into subtasks.
    Returns a list of dicts with 'name', 'description', 'capabilities'.
    """
    caps_str = ', '.join(available_capabilities)

    prompt = f"""You are a task planning AI. Decompose the following task into 2-5 concrete subtasks.

Task: {task}

Available agent capabilities: {caps_str}

For each subtask, specify:
- name: short identifier (snake_case, e.g. "search_info")
- description: what the subtask should accomplish (be specific!)
- capabilities: which capabilities from the list above are needed (pick the most relevant ones)

Respond with ONLY a JSON array, nothing else:
[
  {{"name": "step_name", "description": "What to do", "capabilities": ["capability1"]}},
  ...
]

Rules:
- Each subtask should be actionable and specific to the original task
- Order subtasks logically (research before analysis, analysis before summarization)
- Use "general" capability as fallback if no specific capability fits
- Be practical — break the task into real work steps, not generic phases"""

    response = _call_llm(
        prompt,
        system="You are a task decomposition expert. Always respond with valid JSON arrays only.",
        max_tokens=512
    )

    try:
        import re
        # Find JSON array in response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            steps = json.loads(json_match.group())
            if isinstance(steps, list) and len(steps) > 0:
                # Validate each step has required fields
                valid_steps = []
                for s in steps:
                    if isinstance(s, dict) and 'name' in s and 'description' in s:
                        s.setdefault('capabilities', ['general'])
                        valid_steps.append(s)
                if valid_steps:
                    logger.info(f"LLM decomposed task into {len(valid_steps)} subtasks")
                    return valid_steps
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM decomposition parse error: {e}")

    # Fallback: simple 3-step plan
    logger.info("LLM decomposition failed, using intelligent fallback")
    return [
        {"name": "research", "description": f"Research and gather information about: {task}", "capabilities": ["web_search"]},
        {"name": "analyze", "description": f"Analyze the gathered information for: {task}", "capabilities": ["analysis"]},
        {"name": "synthesize", "description": f"Synthesize findings into a comprehensive answer for: {task}", "capabilities": ["summarization"]},
    ]


def _llm_synthesize_results(task: str, subtask_results: List[Dict[str, str]]) -> str:
    """
    Use the LLM to intelligently synthesize results from multiple subtasks
    into a coherent final answer.
    """
    results_text = ""
    for r in subtask_results:
        name = r.get('name', 'step')
        result = r.get('result', 'No result')
        results_text += f"### {name}\n{result}\n\n"

    prompt = f"""You are synthesizing results from multiple AI agents that worked on a task.

Original Task: {task}

Agent Results:
{results_text}

Create a clear, comprehensive final answer that:
1. Combines insights from all agents
2. Removes redundancy
3. Presents information in a logical flow
4. Highlights key findings and actionable insights
5. Is well-structured with clear sections

Write the synthesized answer directly, no meta-commentary."""

    return _call_llm(
        prompt,
        system="You are an expert at synthesizing information from multiple sources into clear, actionable summaries.",
        max_tokens=1500
    )


def _llm_decompose_swarm_task(task_description: str, available_roles: List[str]) -> List[Dict]:
    """
    Use the LLM to decompose a task for the swarm, assigning roles.
    """
    roles_str = ', '.join(available_roles)

    prompt = f"""Decompose this task for a team of AI agents with these available roles: {roles_str}

Task: {task_description}

For each subtask, specify which role should handle it. Respond with ONLY a JSON array:
[
  {{"description": "Specific subtask description", "role": "role_name", "priority": 1}},
  ...
]

Rules:
- Assign the most appropriate role for each subtask
- Higher priority number = should run first
- Make subtask descriptions specific and actionable
- Use 2-4 subtasks
- CRITICAL: Subtask 'description' should ONLY contain the core topic or query (e.g. "quantum computing breakthroughs 2026"). Do NOT use prefixes like "Research:" or "Analyze:"."""

    response = _call_llm(
        prompt,
        system="You are a team coordination expert. Always respond with valid JSON arrays only.",
        max_tokens=600
    )

    try:
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            steps = json.loads(json_match.group())
            if isinstance(steps, list) and len(steps) > 0:
                return steps
    except Exception:
        pass

    # Fallback
    return [
        {"description": task_description, "role": "researcher", "priority": 2},
        {"description": task_description, "role": "reviewer", "priority": 1},
    ]


# ============== AI-Powered Tools ==============

def _create_web_search_tool() -> Tool:
    """Create web search tool using DuckDuckGo."""
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the internet for information."""
        try:
            from openclaw.integrations.search import get_search_engine
            engine = get_search_engine("duckduckgo")
            response = engine.search(query, max_results=max_results)

            if not response.results:
                return f"No results found for: {query}"

            lines = [f"Search results for: {query}\n"]
            for i, r in enumerate(response.results, 1):
                lines.append(f"{i}. {r.title}")
                lines.append(f"   {r.url}")
                if r.snippet:
                    lines.append(f"   {r.snippet[:200]}")
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            return f"Search error: {e}"

    return Tool(
        name="web_search",
        description="Search the internet for information. Args: query (str), max_results (int, optional)",
        func=web_search,
        parameters={"query": "string", "max_results": "int (optional, default 5)"}
    )


def _create_ai_ask_tool() -> Tool:
    """Create AI question-answering tool."""
    def ai_ask(question: str) -> str:
        """Ask the AI a question and get a direct answer."""
        return _call_llm(
            prompt=question,
            system="You are a knowledgeable AI assistant. Give concise, factual answers.",
            max_tokens=512
        )

    return Tool(
        name="ai_ask",
        description="Ask the AI a question. Args: question (str)",
        func=ai_ask,
        parameters={"question": "string"}
    )


def _create_fetch_url_tool() -> Tool:
    """Create URL fetching tool."""
    def fetch_url(url: str) -> str:
        """Fetch and extract text content from a URL."""
        import requests
        from bs4 import BeautifulSoup

        if not url.startswith("http"):
            url = "https://" + url

        try:
            response = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts, styles, and common boilerplate elements
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                element.decompose()
                
            text = soup.get_text(separator=' ', strip=True)
            return text[:4000]
        except Exception as e:
            return f"Fetch error: {e}"

    return Tool(
        name="fetch_url",
        description="Fetch text content from a URL. Args: url (str)",
        func=fetch_url,
        parameters={"url": "string"}
    )


def _create_summarize_tool() -> Tool:
    """Create text summarization tool."""
    def summarize(text: str, style: str = "concise") -> str:
        """Summarize long text."""
        return _call_llm(
            prompt=f"Summarize the following text in a {style} manner:\n\n{text[:4000]}",
            system="You are a summarization expert. Create clear, accurate summaries.",
            max_tokens=512
        )

    return Tool(
        name="summarize",
        description="Summarize long text. Args: text (str), style (str, optional: 'concise'|'detailed'|'bullets')",
        func=summarize,
        parameters={"text": "string", "style": "string (optional)"}
    )


def _create_translate_tool() -> Tool:
    """Create translation tool."""
    def translate(text: str, target_language: str = "English") -> str:
        """Translate text to target language."""
        return _call_llm(
            prompt=f"Translate the following text to {target_language}. Only return the translation, no explanations.\n\n{text}",
            system="You are an expert translator. Translate accurately and naturally.",
            max_tokens=1024
        )

    return Tool(
        name="translate",
        description="Translate text to another language. Args: text (str), target_language (str)",
        func=translate,
        parameters={"text": "string", "target_language": "string"}
    )


def _create_analyze_tool() -> Tool:
    """Create analysis tool for structured analysis of topics."""
    def analyze(topic: str, depth: str = "medium") -> str:
        """Analyze a topic in depth."""
        depth_prompts = {
            "brief": "Give a brief 2-3 sentence analysis.",
            "medium": "Provide a thorough analysis with key points and insights.",
            "deep": "Provide an extensive, multi-faceted analysis with pros/cons, implications, and recommendations.",
        }
        return _call_llm(
            prompt=f"Analyze the following: {topic}",
            system=f"You are an expert analyst. {depth_prompts.get(depth, depth_prompts['medium'])}",
            max_tokens=1024
        )

    return Tool(
        name="analyze",
        description="Analyze a topic. Args: topic (str), depth (str, optional: 'brief'|'medium'|'deep')",
        func=analyze,
        parameters={"topic": "string", "depth": "string (optional)"}
    )


def _create_code_tool() -> Tool:
    """Create code generation/analysis tool."""
    def generate_code(task: str, language: str = "python") -> str:
        """Generate or explain code."""
        return _call_llm(
            prompt=f"Task: {task}\nLanguage: {language}",
            system="You are an expert programmer. Write clean, well-commented code. If the task is about explaining code, provide clear explanations.",
            max_tokens=1024
        )

    return Tool(
        name="generate_code",
        description="Generate or explain code. Args: task (str), language (str, optional)",
        func=generate_code,
        parameters={"task": "string", "language": "string (optional, default 'python')"}
    )


# ============== Orchestrator Handlers ==============

def _create_search_handler():
    """Handler for web_search capability."""
    def handler(subtask, context):
        registry = get_tool_registry()
        result = registry.execute("web_search", query=subtask.description)
        return result.result if result.status == ToolCallStatus.SUCCESS else f"Search failed: {result.error}"
    return handler


def _create_analysis_handler():
    """Handler for analysis capability."""
    def handler(subtask, context):
        # Use previous results as context if available
        prev_results = context.get("previous_results", "")
        text = subtask.description
        if prev_results:
            text = f"{text}\n\nContext from previous steps:\n{prev_results}"
        registry = get_tool_registry()
        result = registry.execute("analyze", topic=text)
        return result.result if result.status == ToolCallStatus.SUCCESS else f"Analysis failed: {result.error}"
    return handler


def _create_summarization_handler():
    """Handler for summarization capability."""
    def handler(subtask, context):
        prev_results = context.get("previous_results", "")
        text = prev_results if prev_results else subtask.description
        registry = get_tool_registry()
        result = registry.execute("summarize", text=text)
        return result.result if result.status == ToolCallStatus.SUCCESS else f"Summary failed: {result.error}"
    return handler


def _create_coding_handler():
    """Handler for coding capability."""
    def handler(subtask, context):
        registry = get_tool_registry()
        result = registry.execute("generate_code", task=subtask.description)
        return result.result if result.status == ToolCallStatus.SUCCESS else f"Code gen failed: {result.error}"
    return handler


def _create_general_handler():
    """Handler for general tasks."""
    def handler(subtask, context):
        return _call_llm(
            prompt=subtask.description,
            system="You are a helpful assistant. Complete the given task thoroughly.",
            max_tokens=1024
        )
    return handler


# ============== Memory & Knowledge Tools ==============


def _create_memory_search_tool() -> Tool:
    """Create tool for agents to search their own memory."""
    def search_memory(query: str, memory_type: str = "all", limit: int = 5) -> str:
        """Search past conversations and stored knowledge from memory.
        Use this when:
        - The user references something from a previous conversation
        - You need context about past interactions or decisions
        - You want to recall previously stored facts or preferences
        """
        try:
            from .agent_memory import get_agent_memory, MemoryQuery, MemoryType
            memory = get_agent_memory()

            mem_type = None
            if memory_type != "all":
                type_map = {
                    "episodic": MemoryType.EPISODIC,
                    "semantic": MemoryType.SEMANTIC,
                    "procedural": MemoryType.PROCEDURAL,
                    "working": MemoryType.WORKING,
                }
                mem_type = type_map.get(memory_type.lower())

            results = memory.query_memories(MemoryQuery(
                text=query, memory_type=mem_type, limit=limit
            ))

            if not results:
                return f"No memories found for: {query}"

            lines = [f"Found {len(results)} relevant memories:\n"]
            for i, m in enumerate(results, 1):
                from datetime import datetime
                when = datetime.fromtimestamp(m.timestamp).strftime('%Y-%m-%d %H:%M')
                lines.append(f"{i}. [{when}] (importance: {m.importance:.1f}) {m.content[:300]}")
            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Memory search error: {e}")
            # Fallback: check file-based memories
            try:
                memory_file = os.path.expanduser("~/.openclaw/memories/user_memories.jsonl")
                if os.path.exists(memory_file):
                    import json as _json
                    matches = []
                    with open(memory_file) as f:
                        for line in f:
                            entry = _json.loads(line.strip())
                            content = entry.get("content", "")
                            if query.lower() in content.lower():
                                matches.append(content[:200])
                    if matches:
                        return f"Found {len(matches)} memories:\n" + "\n".join(f"- {m}" for m in matches[:limit])
                return f"No memories found for: {query}"
            except Exception:
                return f"Memory system unavailable. No results for: {query}"

    return Tool(
        name="search_memory",
        description="Search your memory for past conversations, user preferences, stored facts. Args: query (str), memory_type (str, optional: 'all'|'episodic'|'semantic'), limit (int, optional)",
        func=search_memory,
        parameters={"query": "string", "memory_type": "string (optional)", "limit": "int (optional, default 5)"}
    )


def _create_memory_store_tool() -> Tool:
    """Create tool for agents to store important information in memory."""
    def store_memory(content: str, importance: float = 0.7, category: str = "episodic") -> str:
        """Store important information in long-term memory.
        Use this when:
        - The user shares important preferences or personal info
        - You discover a key fact that should be remembered
        - The user explicitly asks you to remember something
        """
        try:
            from .agent_memory import get_agent_memory, MemoryType
            memory = get_agent_memory()

            type_map = {
                "episodic": MemoryType.EPISODIC,
                "semantic": MemoryType.SEMANTIC,
                "procedural": MemoryType.PROCEDURAL,
            }
            mem_type = type_map.get(category.lower(), MemoryType.EPISODIC)

            memory.add_memory(
                content=content,
                memory_type=mem_type,
                importance=min(max(importance, 0.0), 1.0),
                metadata={"source": "agent", "auto_stored": True}
            )
            return f"✅ Stored in {category} memory (importance: {importance}): {content[:100]}"

        except Exception as e:
            # Fallback: file-based storage
            try:
                memory_dir = os.path.expanduser("~/.openclaw/memories")
                os.makedirs(memory_dir, exist_ok=True)
                memory_file = os.path.join(memory_dir, "user_memories.jsonl")
                import json as _json
                with open(memory_file, "a") as f:
                    f.write(_json.dumps({
                        "content": content,
                        "importance": importance,
                        "category": category,
                        "source": "agent",
                        "timestamp": time.time()
                    }) + "\n")
                return f"✅ Stored: {content[:100]}"
            except Exception as e2:
                return f"Failed to store memory: {e2}"

    return Tool(
        name="store_memory",
        description="Store important information in long-term memory for future recall. Args: content (str), importance (float 0-1, optional), category (str, optional: 'episodic'|'semantic'|'procedural')",
        func=store_memory,
        parameters={"content": "string", "importance": "float (optional)", "category": "string (optional)"}
    )


def _create_rag_search_tool() -> Tool:
    """Create tool for agents to search the RAG knowledge base."""
    def search_knowledge(query: str, top_k: int = 5) -> str:
        """Search the indexed knowledge base for specific information.
        Use this when:
        - You need factual information from indexed documents
        - The user asks about topics that might be in stored files
        - You want to find relevant documentation or notes
        """
        try:
            from .rag_engine import RAGEngine, RAGConfig
            rag = RAGEngine(RAGConfig(vector_db_backend="memory"))

            # Auto-index common paths if not already done
            if rag.store.count() == 0:
                default_paths = [
                    os.path.expanduser("~/.openclaw/notes"),
                    os.path.expanduser("~/.openclaw/memories"),
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        try:
                            rag.index_directory(path)
                        except Exception:
                            pass

            results = rag.query(query, top_k=top_k)

            if not results:
                return f"No knowledge base results for: {query}"

            lines = [f"Found {len(results)} relevant documents:\n"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. [score: {r.similarity:.2f}] Source: {r.source}")
                lines.append(f"   {r.text[:300]}")
            return "\n".join(lines)

        except Exception as e:
            return f"Knowledge base search unavailable: {e}"

    return Tool(
        name="search_knowledge",
        description="Search the indexed knowledge base (documents, notes, files). Args: query (str), top_k (int, optional)",
        func=search_knowledge,
        parameters={"query": "string", "top_k": "int (optional, default 5)"}
    )


# ============== AskUser Tool ==============

# Global reference to TelegramBot for AskUser tool interaction.
# Set by TelegramBot._bootstrap_agents() after bootstrapping.
_telegram_bot_ref = None


def set_telegram_bot_ref(bot):
    """Set the global reference to TelegramBot for AskUser tool.
    Called by TelegramBot after bootstrap to connect the tool to the bot."""
    global _telegram_bot_ref
    _telegram_bot_ref = bot
    logger.info("✅ AskUser tool connected to Telegram bot")


def _create_ask_user_tool() -> Tool:
    """Create tool for agents to ask the user structured questions."""
    def ask_user(question: str, options: str = "") -> str:
        """Ask the user a question and wait for their response.
        Use this when:
        - You need clarification before proceeding with a task
        - The task has multiple valid approaches and user preference matters
        - You want to confirm before taking an irreversible action
        
        Args:
            question: The question to ask the user
            options: Comma-separated list of options (e.g., "Python,Rust,Go").
                     If empty, provides Yes/No by default.
        """
        global _telegram_bot_ref
        if not _telegram_bot_ref:
            return f"(Cannot ask user — no active chat channel. Proceeding with best judgment for: {question})"

        # Parse options
        if options and options.strip():
            option_list = [o.strip() for o in options.split(",") if o.strip()]
        else:
            option_list = ["Yes", "No"]

        # Limit to 6 options (Telegram inline keyboard practical limit)
        option_list = option_list[:6]

        # Generate unique request ID
        import hashlib
        request_id = hashlib.sha256(f"{question}{time.time()}".encode()).hexdigest()[:12]

        logger.info(f"AskUser: '{question}' with options {option_list} (id: {request_id})")

        # This blocks until user responds or timeout (120s)
        response = _telegram_bot_ref.wait_for_user_response(
            request_id=request_id,
            question=question,
            options=option_list,
            timeout=120.0
        )

        return f"User responded: {response}"

    return Tool(
        name="ask_user",
        description="Ask the user a question with clickable options. Use for clarification or confirmation. Args: question (str), options (str, optional, comma-separated choices e.g. 'Python,Rust,Go')",
        func=ask_user,
        parameters={"question": "string", "options": "string (optional, comma-separated)"}
    )


def _create_relay_tool() -> Tool:
    """Create tool for agents to delegate tasks to the other bot."""
    def relay_to_bot(task: str, wait_for_response: str = "no") -> str:
        """Delegate a task to the other bot (Ellora) for processing.
        Use this when:
        - The task needs capabilities you don't have
        - You want a second opinion or cross-validation
        - The user explicitly asks you to coordinate with the other bot

        Args:
            task: The task or question to send to the other bot
            wait_for_response: 'yes' to wait for response, 'no' to fire-and-forget
        """
        try:
            from .interbot import get_interbot_bridge
            bridge = get_interbot_bridge()
            other = bridge.get_other_bot()

            if wait_for_response.lower() in ("yes", "true", "1"):
                response = bridge.send_query(other, task, timeout=90.0)
                return f"Response from {other.title()}: {response}"
            else:
                msg_id = bridge.send_task(other, task)
                return f"Task delegated to {other.title()} (ID: {msg_id[:8]}). Response will arrive asynchronously."
        except Exception as e:
            return f"Relay failed: {e}"

    return Tool(
        name="relay_to_bot",
        description="Delegate a task to the other bot (Ellora). Args: task (str), wait_for_response (str, 'yes'|'no')",
        func=relay_to_bot,
        parameters={"task": "string", "wait_for_response": "string (optional, 'yes' or 'no')"}
    )


# ============== Bootstrap ==============

_bootstrapped = False


def _create_time_tool() -> Tool:
    """Create time awareness tool."""
    def get_time() -> str:
        """Get the current system date and time."""
        from datetime import datetime
        return f"Current runtime string: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p %Z')}"

    return Tool(
        name="get_time",
        description="Get exact current time and date.",
        func=get_time,
        parameters={}
    )


def bootstrap_agents() -> Dict[str, Any]:
    """
    Bootstrap the entire agent system.

    Call this once at startup to:
    1. Register AI-powered tools
    2. Set up the orchestrator with agents and handlers
    3. Configure the agent swarm

    Returns dict with references to initialized systems.
    """
    global _bootstrapped
    if _bootstrapped:
        logger.info("Agents already bootstrapped, skipping")
        return {}

    logger.info("🚀 Bootstrapping agent system...")

    # 1. Register AI-powered tools
    registry = get_tool_registry()
    new_tools = [
        _create_web_search_tool(),
        _create_ai_ask_tool(),
        _create_fetch_url_tool(),
        _create_summarize_tool(),
        _create_translate_tool(),
        _create_analyze_tool(),
        _create_code_tool(),
        _create_time_tool(),
        _create_memory_search_tool(),
        _create_memory_store_tool(),
        _create_rag_search_tool(),
        _create_ask_user_tool(),
        _create_relay_tool(),
    ]
    for tool in new_tools:
        registry.register(tool)
    logger.info(f"✅ Registered {len(new_tools)} AI-powered tools (total: {len(registry.list_tools())})")

    # 2. Set up the Orchestrator with LLM-powered intelligence
    from .orchestrator import AgentOrchestrator, SubTask, TaskPlan, ExecutionStrategy, TaskStatus
    orchestrator = AgentOrchestrator(max_parallel=4, default_timeout=120.0)

    # Register agents
    orchestrator.registry.register("searcher", "Web Searcher", ["web_search"])
    orchestrator.registry.register("analyst", "Data Analyst", ["analysis"])
    orchestrator.registry.register("summarizer", "Summarizer", ["summarization"])
    orchestrator.registry.register("coder", "Code Expert", ["coding"])
    orchestrator.registry.register("generalist", "General Agent", ["general"], max_concurrent=3)

    # Register handlers
    orchestrator.register_handler("web_search", _create_search_handler())
    orchestrator.register_handler("analysis", _create_analysis_handler())
    orchestrator.register_handler("summarization", _create_summarization_handler())
    orchestrator.register_handler("coding", _create_coding_handler())
    orchestrator.register_handler("general", _create_general_handler())

    # === INJECT LLM-POWERED DECOMPOSITION ===
    available_caps = ["web_search", "analysis", "summarization", "coding", "general"]
    original_decompose = orchestrator.decomposer.decompose

    def llm_decompose(task, strategy=ExecutionStrategy.SEQUENTIAL, template=None):
        """LLM-powered task decomposition — replaces keyword matching."""
        # If a specific template is requested, use original behavior
        if template:
            return original_decompose(task, strategy, template)

        # Use LLM to decompose the task
        llm_steps = _llm_decompose_task(task, available_caps)

        plan = TaskPlan(original_task=task, strategy=strategy)
        subtasks = []
        for i, step in enumerate(llm_steps):
            subtask = SubTask(
                name=step["name"],
                description=step["description"],
                required_capabilities=step.get("capabilities", ["general"]),
                dependencies=[subtasks[i-1].id] if i > 0 else [],
            )
            subtasks.append(subtask)
        plan.subtasks = subtasks
        logger.info(f"🧠 LLM decomposed task into {len(subtasks)} subtasks: {[s.name for s in subtasks]}")
        return plan

    orchestrator.decomposer.decompose = llm_decompose

    # === INJECT LLM-POWERED RESULT SYNTHESIS ===
    original_merge = orchestrator.aggregator._merge_results

    def llm_merge_results(plan):
        """LLM-powered result synthesis — replaces simple concatenation."""
        subtask_results = []
        for subtask in plan.subtasks:
            if subtask.status == TaskStatus.COMPLETED and subtask.result is not None:
                subtask_results.append({
                    "name": subtask.name,
                    "result": str(subtask.result)[:1000]
                })

        if not subtask_results:
            return original_merge(plan)

        # Use LLM to synthesize
        try:
            synthesized = _llm_synthesize_results(plan.original_task, subtask_results)
            if synthesized and not synthesized.startswith("Error"):
                return synthesized
        except Exception as e:
            logger.warning(f"LLM synthesis failed, falling back: {e}")

        return original_merge(plan)

    orchestrator.aggregator._merge_results = llm_merge_results
    logger.info("✅ Orchestrator upgraded with LLM-powered decomposition & synthesis")

    logger.info("✅ Orchestrator configured with 5 agents and handlers")

    # 3. Set up the Swarm with LLM-powered intelligence
    from .agent_swarm import AgentSwarm, AgentRole, SwarmTask
    swarm = AgentSwarm()

    def researcher_handler(task_desc):
        """Research handler — generates optimal search query, searches, and analyzes top 10 full URLs."""
        # Ask LLM to extract a clean search query
        query_prompt = f"Convert this task into a 3-5 word optimized Google search query. Output ONLY the search keywords, nothing else. Task: {task_desc}"
        search_query = _call_llm(query_prompt, system="You are an expert search engine query optimizer. Return only the raw keywords.", max_tokens=256)
        search_query = search_query.strip().strip('"\'')
        
        logger.info(f"🔍 Researcher Agent generated search query: '{search_query}' from task: '{task_desc}'")
        
        from openclaw.integrations.search import get_search_engine
        engine = get_search_engine()
        
        # 1. Search for top 10 results
        response = engine.search(search_query, max_results=10)
        
        if not response.results:
            return f"Research error: No results found for '{search_query}'"
            
        logger.info(f"🔍 Researcher Agent found {len(response.results)} results, fetching full text for top URLs...")
        
        # 2. Fetch full text for every URL
        combined_text = f"Search Query: {search_query}\n\n"
        for i, r in enumerate(response.results):
            url = r.url
            if url:
                fetch_res = registry.execute("fetch_url", url=url)
                fetched_text = fetch_res.result if fetch_res.status == ToolCallStatus.SUCCESS else "(Failed to fetch content)"
                
                # Take up to 2500 characters from EACH webpage to build a massive context (but stay within LLM limits)
                combined_text += f"---\nSource {i+1}: {r.title}\nURL: {url}\nSnippet: {r.snippet}\nContent:\n{str(fetched_text)[:2500]}\n\n"
                
        logger.info(f"🔍 Researcher Agent compiling {len(combined_text)} characters of scraped web content for deep analysis...")
        
        # 3. Analyze the deeply scraped content
        analysis = registry.execute(
            "analyze", 
            topic=f"Based on the following scraped webpage contents from 10 sources, analyze: {task_desc}\n\nWebpage Data:\n{combined_text}",
            depth="deep"
        )
        
        return analysis.result if analysis.status == ToolCallStatus.SUCCESS else f"Analysis failed, raw data snippet:\n{combined_text[:4000]}"

    def coder_handler(task_desc):
        """Code handler — generates/explains code."""
        result = registry.execute("generate_code", task=task_desc)
        return result.result if result.status == ToolCallStatus.SUCCESS else f"Code error: {result.error}"

    def analyst_handler(task_desc):
        """Analyst handler — deep analysis."""
        result = registry.execute("analyze", topic=task_desc, depth="deep")
        return result.result if result.status == ToolCallStatus.SUCCESS else f"Analysis error: {result.error}"

    def writer_handler(task_desc):
        """Writer handler — summarizes and writes."""
        result = registry.execute("summarize", text=task_desc, style="detailed")
        return result.result if result.status == ToolCallStatus.SUCCESS else f"Writing error: {result.error}"

    # Deploy multiple agents per role to prevent bottlenecking on parallel tasks
    for i in range(1, 6):  # 5 Researchers
        swarm.add_agent(f"Researcher-{i}", AgentRole.RESEARCHER, researcher_handler, ["web_search", "analysis"])
        
    for i in range(1, 4):  # 3 Coders
        swarm.add_agent(f"Coder-{i}", AgentRole.CODER, coder_handler, ["coding", "debugging"])
        
    for i in range(1, 4):  # 3 Analysts
        swarm.add_agent(f"Analyst-{i}", AgentRole.REVIEWER, analyst_handler, ["analysis", "review"])
        
    for i in range(1, 3):  # 2 Writers
        swarm.add_agent(f"Writer-{i}", AgentRole.COMMUNICATOR, writer_handler, ["summarization", "writing"])

    # === INJECT LLM-POWERED SWARM DECOMPOSITION ===
    original_swarm_decompose = swarm.decomposer.decompose

    def llm_swarm_decompose(task_obj, available_roles):
        """LLM-powered swarm task decomposition."""
        role_names = [r.value for r in available_roles]
        llm_steps = _llm_decompose_swarm_task(task_obj.description, role_names)

        subtasks = []
        for step in llm_steps:
            role_str = step.get("role", "executor")
            try:
                role = AgentRole(role_str)
            except ValueError:
                role = AgentRole.EXECUTOR

            subtasks.append(SwarmTask(
                description=step["description"],
                required_role=role,
                parent_task=task_obj.id,
                priority=step.get("priority", task_obj.priority)
            ))

        if not subtasks:
            return original_swarm_decompose(task_obj, available_roles)

        task_obj.subtasks = [st.id for st in subtasks]
        logger.info(f"🧠 LLM decomposed swarm task into {len(subtasks)} subtasks")
        return subtasks

    swarm.decomposer.decompose = llm_swarm_decompose
    logger.info("✅ Swarm upgraded with LLM-powered decomposition")

    # 4. Set up the ReAct agent
    from .react_agent import ReActAgent

    def llm_think(goal, history, context):
        """LLM-powered thinking function for ReAct agent."""
        history_text = ""
        for step in history[-5:]:  # Last 5 steps for context
            if step.step_type.value == "thought":
                history_text += f"Thought: {step.content}\n"
            elif step.step_type.value == "action":
                history_text += f"Action: {step.tool_name}({step.tool_args})\n"
            elif step.step_type.value == "observation":
                history_text += f"Observation: {step.content[:300]}\n"

        tools = get_tool_registry().list_tools()
        tool_names = [t["name"] for t in tools]

        # Build tool descriptions for better LLM awareness
        tool_descs = []
        for t in tools:
            tool_descs.append(f"- {t['name']}: {t.get('description', 'No description')}")
        tools_section = '\n'.join(tool_descs)

        from datetime import datetime
        current_time_str = datetime.now().strftime('%A, %B %d, %Y %I:%M %p %Z')

        prompt = f"""You are an advanced autonomous AI agent using the ReAct framework (Reason + Act).
You solve tasks step-by-step by thinking carefully, using tools, and observing results.

CURRENT SYSTEM TIME: {current_time_str}
(Use this to answer questions about 'today', 'now', or relative dates)

GOAL: {goal}

AVAILABLE TOOLS:
{tools_section}

PREVIOUS STEPS:
{history_text if history_text else '(none yet — this is the first step)'}

INSTRUCTIONS:
1. Think about what information you need and which tool would help
2. If you already have enough info from previous observations, give the final answer
3. Do NOT repeat actions you've already taken
4. Be specific in tool arguments — use exact queries, not vague ones
5. For web_search, use detailed search queries
6. For ai_ask, ask specific questions
7. CRITICAL: If a tool returns an error or 0 results, DO NOT get stuck in an infinite loop retrying. If search fails twice, STOP searching and formulate a final answer using your own knowledge.
8. Never say "I don't have real-time access" since you DO have access. If searches fail, just give the best answer you can.

Respond with ONLY valid JSON in one of these formats:

To use a tool:
{{
    "reasoning": "I need to [specific reason]. The best tool is [tool] because [why].",
    "action": "tool_name",
    "action_args": {{"arg_name": "value"}}
}}

To give the final answer (only when you have enough information):
{{
    "reasoning": "I have gathered [what]. The answer is clear because [why].",
    "action": "final_answer",
    "answer": "Your comprehensive, well-structured answer here"
}}"""

        response = _call_llm(
            prompt,
            system="You are an autonomous AI agent that solves tasks using tools. Think step-by-step. Always respond with valid JSON only — no markdown, no explanations outside JSON.",
            max_tokens=700
        )

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # Normalize: ensure 'reasoning' key exists
                if 'thought' in data and 'reasoning' not in data:
                    data['reasoning'] = data.pop('thought')
                # Normalize: if 'is_final' is True and no 'action', set action to final_answer
                if data.get('is_final') and 'action' not in data:
                    data['action'] = 'final_answer'
                
                # Robust extraction of the actual answer string
                if data.get('action') == 'final_answer' and not data.get('answer'):
                    possible_keys = ['response', 'output', 'final_answer', 'text', 'message']
                    for k in possible_keys:
                        if data.get(k):
                            data['answer'] = data[k]
                            break
                    if not data.get('answer'):
                        data['answer'] = data.get('reasoning', 'Done.')
                
                return data
        except (json.JSONDecodeError, Exception):
            pass

        # Fallback
        return {
            "reasoning": response[:200],
            "action": "final_answer",
            "answer": response,
        }

    react_agent = ReActAgent(
        think_fn=llm_think,
        tool_registry=get_tool_registry(),
        max_steps=8,
        max_retries=2,
        verbose=True,
    )

    logger.info("✅ ReAct agent configured with LLM-powered reasoning")

    _bootstrapped = True
    logger.info("🎉 Agent system fully bootstrapped!")

    return {
        "orchestrator": orchestrator,
        "swarm": swarm,
        "react_agent": react_agent,
        "tool_registry": registry,
    }


def get_tool_list() -> str:
    """Get formatted list of all registered tools."""
    tools = get_tool_registry().list_tools()
    if not tools:
        return "No tools registered."
    lines = ["📦 Available Tools:\n"]
    for t in tools:
        lines.append(f"  🔧 {t['name']}")
        lines.append(f"     {t['description']}")
    return "\n".join(lines)


__all__ = [
    "bootstrap_agents",
    "get_tool_list",
    "set_telegram_bot_ref",
]
