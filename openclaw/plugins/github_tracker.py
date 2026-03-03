"""
GitHub Issues Task Tracker Plugin for OpenClaw

Implements the TaskTracker ABC from plugin_system.py.
Provides CRUD operations on GitHub Issues via the REST API.

Config:
    token: GitHub Personal Access Token (or GITHUB_TOKEN env)
    owner: Repository owner (e.g., "username")
    repo: Repository name (e.g., "openclaw")
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

from core.plugin_system import TaskTracker, PluginManifest, PluginModule, PluginSlot
from core.logger import get_logger

logger = get_logger("plugin.github_tracker")


# GitHub API base
GITHUB_API = "https://api.github.com"

# Map OpenClaw states to GitHub labels
STATE_LABEL_MAP = {
    "pending": "todo",
    "running": "in-progress",
    "completed": "done",
    "failed": "bug",
    "blocked": "blocked",
}


class GitHubTrackerPlugin(TaskTracker):
    """
    GitHub Issues task tracker.

    Maps OpenClaw tasks to GitHub issues:
    - create_issue → POST /repos/{owner}/{repo}/issues
    - get_issue → GET /repos/{owner}/{repo}/issues/{number}
    - update_issue → PATCH /repos/{owner}/{repo}/issues/{number}
    - list_issues → GET /repos/{owner}/{repo}/issues

    Config:
        token: PAT with 'repo' scope
        owner: GitHub user/org
        repo: Repository name
    """

    def __init__(self):
        self.token: Optional[str] = None
        self.owner: str = ""
        self.repo: str = ""
        self._total_created: int = 0
        self._total_updated: int = 0
        self._total_fetched: int = 0

    @property
    def name(self) -> str:
        return "github-tracker"

    def configure(self, config: Dict[str, Any]) -> None:
        self.token = config.get("token") or os.getenv("GITHUB_TOKEN")
        self.owner = config.get("owner", self.owner)
        self.repo = config.get("repo", self.repo)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _api_url(self, path: str) -> str:
        return f"{GITHUB_API}/repos/{self.owner}/{self.repo}/{path}"

    async def get_issue(self, identifier: str) -> Dict[str, Any]:
        """Fetch a GitHub issue by number."""
        import requests

        url = self._api_url(f"issues/{identifier}")
        try:
            resp = requests.get(url, headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                self._total_fetched += 1
                data = resp.json()
                return {
                    "id": str(data["number"]),
                    "title": data["title"],
                    "description": data.get("body", ""),
                    "state": data["state"],
                    "labels": [l["name"] for l in data.get("labels", [])],
                    "assignees": [a["login"] for a in data.get("assignees", [])],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "url": data["html_url"],
                }
            return {"error": f"HTTP {resp.status_code}", "body": resp.text[:200]}
        except Exception as e:
            return {"error": str(e)}

    async def create_issue(
        self,
        title: str,
        description: str,
        labels: List[str] = None,
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """Create a new GitHub issue."""
        import requests

        all_labels = list(labels or [])
        if priority in ("urgent", "high"):
            all_labels.append("priority:high")
        elif priority == "low":
            all_labels.append("priority:low")
        all_labels.append("openclaw")  # Tag all OpenClaw-created issues

        payload = {
            "title": title,
            "body": description,
            "labels": all_labels,
        }

        try:
            resp = requests.post(
                self._api_url("issues"),
                headers=self._headers(),
                json=payload,
                timeout=10,
            )
            if resp.status_code == 201:
                self._total_created += 1
                data = resp.json()
                logger.info("Created GitHub issue #%d: %s", data["number"], title)
                return {
                    "id": str(data["number"]),
                    "title": data["title"],
                    "url": data["html_url"],
                    "state": data["state"],
                }
            return {"error": f"HTTP {resp.status_code}", "body": resp.text[:200]}
        except Exception as e:
            return {"error": str(e)}

    async def update_issue(
        self,
        identifier: str,
        state: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        """Update a GitHub issue — change state and/or add a comment."""
        import requests

        # Update state
        if state:
            github_state = "closed" if state in ("completed", "done", "resolved") else "open"
            patch_data = {"state": github_state}

            # Add label for OpenClaw state
            label = STATE_LABEL_MAP.get(state)
            if label:
                patch_data["labels"] = [label, "openclaw"]

            try:
                resp = requests.patch(
                    self._api_url(f"issues/{identifier}"),
                    headers=self._headers(),
                    json=patch_data,
                    timeout=10,
                )
                if resp.status_code == 200:
                    self._total_updated += 1
                    logger.info("Updated issue #%s state → %s", identifier, state)
            except Exception as e:
                logger.error("Failed to update issue #%s: %s", identifier, e)

        # Add comment
        if comment:
            try:
                requests.post(
                    self._api_url(f"issues/{identifier}/comments"),
                    headers=self._headers(),
                    json={"body": comment},
                    timeout=10,
                )
            except Exception as e:
                logger.error("Failed to comment on issue #%s: %s", identifier, e)

    async def list_issues(
        self,
        state: str = "open",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List GitHub issues with filters."""
        import requests

        params = {
            "state": state,
            "per_page": min(limit, 100),
            "labels": "openclaw",  # Only show OpenClaw-managed issues
            "sort": "updated",
            "direction": "desc",
        }

        try:
            resp = requests.get(
                self._api_url("issues"),
                headers=self._headers(),
                params=params,
                timeout=10,
            )
            if resp.status_code == 200:
                return [
                    {
                        "id": str(issue["number"]),
                        "title": issue["title"],
                        "state": issue["state"],
                        "labels": [l["name"] for l in issue.get("labels", [])],
                        "url": issue["html_url"],
                        "updated_at": issue["updated_at"],
                    }
                    for issue in resp.json()
                ]
            return []
        except Exception:
            return []

    def get_stats(self) -> Dict[str, Any]:
        return {
            "owner": self.owner,
            "repo": self.repo,
            "configured": bool(self.token and self.owner and self.repo),
            "total_created": self._total_created,
            "total_updated": self._total_updated,
            "total_fetched": self._total_fetched,
        }


# ============== Plugin Module ==============

def create_plugin(config: Dict[str, Any] = None) -> GitHubTrackerPlugin:
    plugin = GitHubTrackerPlugin()
    if config:
        plugin.configure(config)
    return plugin


MANIFEST = PluginManifest(
    name="github-tracker",
    slot=PluginSlot.TASK_TRACKER,
    description="GitHub Issues task tracker",
    version="1.0.0",
)

github_module = PluginModule(manifest=MANIFEST, create=create_plugin)


__all__ = ["GitHubTrackerPlugin", "create_plugin", "MANIFEST", "github_module"]
