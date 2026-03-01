"""
Autonomous Research Agent for OpenClaw

Multi-step web research:
- Search → Read → Extract → Synthesize
- Source credibility scoring
- Citation tracking
- Research session management
"""

import time
import hashlib
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .logger import get_logger

logger = get_logger("research_agent")


class SourceType(Enum):
    """Types of information sources."""
    WEB_SEARCH = "web_search"
    WEB_PAGE = "web_page"
    FILE = "file"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"


@dataclass
class Source:
    """An information source."""
    url: str
    title: str = ""
    source_type: SourceType = SourceType.WEB_PAGE
    credibility: float = 0.5  # 0-1
    accessed_at: float = field(default_factory=time.time)


@dataclass
class Finding:
    """A piece of information found during research."""
    id: str
    content: str
    source: Source
    relevance: float = 0.5
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)


@dataclass
class ResearchQuery:
    """A research query with parameters."""
    question: str
    max_sources: int = 5
    min_credibility: float = 0.3
    search_depth: int = 2  # How many levels deep to search
    include_types: List[SourceType] = field(
        default_factory=lambda: [SourceType.WEB_SEARCH, SourceType.MEMORY]
    )


@dataclass
class ResearchReport:
    """Final research report."""
    query: str
    findings: List[Finding] = field(default_factory=list)
    summary: str = ""
    sources_consulted: int = 0
    confidence: float = 0.0
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SearchProvider:
    """
    Abstract search provider.
    Integrates with SearXNG or other search backends.
    """

    def __init__(self, search_fn: Optional[Callable] = None, base_url: str = ""):
        self.search_fn = search_fn
        self.base_url = base_url

    def search(self, query: str, max_results: int = 5) -> List[Source]:
        """Execute a search query."""
        if self.search_fn:
            try:
                results = self.search_fn(query, max_results)
                return [
                    Source(
                        url=r.get("url", ""),
                        title=r.get("title", ""),
                        source_type=SourceType.WEB_SEARCH,
                        credibility=self._score_credibility(r.get("url", ""))
                    )
                    for r in results
                ]
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return []

        # Fallback: return empty
        logger.warning("No search function configured")
        return []

    def _score_credibility(self, url: str) -> float:
        """Score source credibility based on domain."""
        high_credibility = [
            "wikipedia.org", "github.com", "stackoverflow.com",
            "docs.python.org", "arxiv.org", "mozilla.org",
            ".edu", ".gov", "documentation"
        ]
        medium_credibility = [
            "medium.com", "dev.to", "blog", "tutorial"
        ]

        url_lower = url.lower()
        for domain in high_credibility:
            if domain in url_lower:
                return 0.9
        for domain in medium_credibility:
            if domain in url_lower:
                return 0.6
        return 0.4


class ContentExtractor:
    """
    Extract structured information from raw content.
    """

    def extract_facts(self, content: str, query: str) -> List[str]:
        """Extract relevant facts from content."""
        if not content:
            return []

        # Simple sentence-level extraction
        sentences = content.replace('\n', ' ').split('.')
        query_words = set(query.lower().split())

        relevant = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10 or len(sentence) > 500:
                continue

            # Score by keyword overlap
            sent_words = set(sentence.lower().split())
            overlap = len(query_words & sent_words)
            if overlap >= 2 or (overlap >= 1 and len(query_words) <= 3):
                relevant.append(sentence)

        return relevant[:10]  # Return top 10

    def summarize(self, findings: List[Finding], query: str) -> str:
        """Create a summary from findings."""
        if not findings:
            return "No relevant information found."

        # Sort by relevance
        sorted_findings = sorted(findings, key=lambda f: f.relevance, reverse=True)

        parts = [f"Research Summary for: {query}\n"]
        seen_content = set()

        for f in sorted_findings[:10]:
            # Deduplicate
            content_hash = hashlib.sha256(f.content[:100].encode()).hexdigest()[:8]
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            parts.append(f"• {f.content}")
            if f.source.title:
                parts.append(f"  Source: {f.source.title} ({f.source.url})")

        return "\n".join(parts)


class ResearchAgent:
    """
    Autonomous research agent that searches, reads, and synthesizes info.

    Usage:
        agent = ResearchAgent(search_fn=my_searxng_search)

        report = agent.research("What are the latest trends in AI agents?")
        print(report.summary)
    """

    def __init__(
        self,
        search_fn: Optional[Callable] = None,
        fetch_fn: Optional[Callable] = None,
        llm_fn: Optional[Callable] = None
    ):
        self.search_provider = SearchProvider(search_fn=search_fn)
        self.extractor = ContentExtractor()
        self.fetch_fn = fetch_fn  # Function to fetch page content
        self.llm_fn = llm_fn      # LLM for synthesis
        self._reports: List[ResearchReport] = []
        self._findings_cache: Dict[str, List[Finding]] = {}

    def research(self, query: str, config: ResearchQuery = None) -> ResearchReport:
        """
        Conduct autonomous research on a topic.

        Steps:
        1. Search for relevant sources
        2. Fetch and read each source
        3. Extract relevant information
        4. Synthesize into a report
        """
        if config is None:
            config = ResearchQuery(question=query)

        start = time.time()
        report = ResearchReport(query=query)

        logger.info(f"Starting research: {query}")

        # Step 1: Search
        sources = self.search_provider.search(query, config.max_sources)
        report.sources_consulted = len(sources)
        logger.info(f"Found {len(sources)} sources")

        # Step 2 & 3: Fetch and extract
        all_findings = []
        for source in sources:
            if source.credibility < config.min_credibility:
                continue

            findings = self._process_source(source, query)
            all_findings.extend(findings)

        # Check memory-based sources
        if SourceType.MEMORY in config.include_types:
            memory_findings = self._search_memory(query)
            all_findings.extend(memory_findings)

        # Step 4: Synthesize
        report.findings = all_findings
        report.summary = self.extractor.summarize(all_findings, query)
        report.confidence = self._calculate_confidence(all_findings)
        report.duration = time.time() - start

        self._reports.append(report)

        logger.info(
            f"Research complete: {len(all_findings)} findings, "
            f"confidence={report.confidence:.2f}, "
            f"duration={report.duration:.1f}s"
        )

        return report

    def _process_source(self, source: Source, query: str) -> List[Finding]:
        """Fetch and extract findings from a source."""
        try:
            content = ""
            if self.fetch_fn:
                content = self.fetch_fn(source.url)

            if not content:
                return []

            facts = self.extractor.extract_facts(content, query)
            findings = []

            for fact in facts:
                finding_id = hashlib.sha256(
                    f"{fact}{source.url}".encode()
                ).hexdigest()[:10]

                findings.append(Finding(
                    id=finding_id,
                    content=fact,
                    source=source,
                    relevance=source.credibility,
                    confidence=source.credibility * 0.8
                ))

            return findings

        except Exception as e:
            logger.error(f"Failed to process {source.url}: {e}")
            return []

    def _search_memory(self, query: str) -> List[Finding]:
        """Search agent memory for relevant information."""
        try:
            from .agent_memory import get_agent_memory, MemoryQuery
            memory = get_agent_memory()
            results = memory.query_memories(MemoryQuery(text=query, limit=5))

            return [
                Finding(
                    id=f"mem_{m.id}",
                    content=m.content,
                    source=Source(
                        url="memory",
                        title="Agent Memory",
                        source_type=SourceType.MEMORY,
                        credibility=0.7
                    ),
                    relevance=m.importance,
                    confidence=0.7
                )
                for m in results
            ]
        except Exception:
            return []

    def _calculate_confidence(self, findings: List[Finding]) -> float:
        """Calculate overall confidence in the research."""
        if not findings:
            return 0.0

        # Average confidence weighted by relevance
        total_weight = sum(f.relevance for f in findings)
        if total_weight == 0:
            return 0.0

        weighted_conf = sum(f.confidence * f.relevance for f in findings)
        return min(weighted_conf / total_weight, 1.0)

    def get_reports(self) -> List[ResearchReport]:
        """Get all research reports."""
        return self._reports


# ============== Global Instance ==============

_research_agent: Optional[ResearchAgent] = None


def get_research_agent(**kwargs) -> ResearchAgent:
    """Get global research agent."""
    global _research_agent
    if _research_agent is None:
        _research_agent = ResearchAgent(**kwargs)
    return _research_agent


__all__ = [
    "SourceType",
    "Source",
    "Finding",
    "ResearchQuery",
    "ResearchReport",
    "SearchProvider",
    "ContentExtractor",
    "ResearchAgent",
    "get_research_agent",
]
