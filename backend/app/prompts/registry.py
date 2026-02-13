"""
Prompt registry for versioned prompt management.

Loads prompt templates from YAML files, supports A/B testing,
and provides comparison between prompt versions.
"""

import yaml
import structlog
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

logger = structlog.get_logger()
TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class PromptConfig:
    version: str
    name: str
    description: str
    author: str
    status: str  # "active", "archived", "testing"
    system_prompt: str
    user_prompt: str
    model_config: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    changelog: list[dict] = field(default_factory=list)
    file_path: str = ""


@lru_cache(maxsize=32)
def _load_prompt_cached(file_path: str) -> dict:
    """Cache YAML reads to avoid disk I/O on every call."""
    with open(file_path) as f:
        return yaml.safe_load(f)


def load_prompt(file_path: Path) -> PromptConfig:
    """Load a single prompt from a YAML file."""
    data = _load_prompt_cached(str(file_path))
    return PromptConfig(
        version=data.get("version", "1.0"),
        name=data.get("name", file_path.stem),
        description=data.get("description", ""),
        author=data.get("author", "system"),
        status=data.get("status", "active"),
        system_prompt=data.get("system_prompt", ""),
        user_prompt=data.get("user_prompt", ""),
        model_config=data.get("model_config", {}),
        tags=data.get("tags", []),
        changelog=data.get("changelog", []),
        file_path=str(file_path),
    )


def list_prompts() -> list[PromptConfig]:
    """List all available prompt templates."""
    prompts = []
    if not TEMPLATES_DIR.exists():
        logger.warning("Templates directory not found", path=str(TEMPLATES_DIR))
        return prompts

    for yaml_file in sorted(TEMPLATES_DIR.glob("*.yaml")):
        try:
            prompts.append(load_prompt(yaml_file))
        except Exception as e:
            logger.warning("Failed to load prompt", file=str(yaml_file), error=str(e))
    return prompts


def get_active_prompt(name_prefix: str) -> Optional[str]:
    """
    Get the active system prompt for a given name prefix.

    Args:
        name_prefix: Prefix to match against prompt names (e.g., "prediction_engine", "offline", "chat")

    Returns:
        The system_prompt string of the active prompt, or None if not found.
    """
    prompts = list_prompts()
    for prompt in prompts:
        if prompt.name.startswith(name_prefix) and prompt.status == "active":
            logger.debug("Using versioned prompt", name=prompt.name, version=prompt.version)
            return prompt.system_prompt
    return None


def get_prompt_by_version(name_prefix: str, version: str) -> Optional[PromptConfig]:
    """Get a specific prompt version."""
    prompts = list_prompts()
    for prompt in prompts:
        if prompt.name.startswith(name_prefix) and prompt.version == version:
            return prompt
    return None


def compare_prompts(name_prefix: str) -> list[dict]:
    """Compare all versions of a prompt."""
    prompts = list_prompts()
    matching = [p for p in prompts if p.name.startswith(name_prefix)]

    comparisons = []
    for prompt in matching:
        comparisons.append({
            "name": prompt.name,
            "version": prompt.version,
            "status": prompt.status,
            "description": prompt.description,
            "prompt_length": len(prompt.system_prompt),
            "tags": prompt.tags,
            "changelog": prompt.changelog,
        })
    return comparisons
