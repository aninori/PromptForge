"""Agent registry for the specialized orchestration layer.

Each agent is a constrained workflow with a fixed role, a display model label,
and a runtime model hint. The backend uses these definitions to route context
collection, retrieval, and prompt shaping.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import settings
from .schemas import AgentCard

# Store user-created agents alongside the other runtime data, not in the source tree.
_AGENTS_FILE = Path(settings.chroma_dir) / "agents.json"


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    name: str
    role: str
    purpose: str
    model: str
    focus: str
    color: str
    system_prompt: str
    output_format: str
    retrieval_bias: tuple[str, ...]
    default_top_k: int = 6
    use_expansion: bool = True


AGENTS: dict[str, AgentDefinition] = {
    "debug": AgentDefinition(
        id="debug",
        name="Debug Agent",
        role="Specialist",
        purpose="Pinpoints exceptions, memory spikes, and failed code paths, and proposes a fix.",
        model=f"Ollama / {settings.gen_model}",
        focus="Bug triage, stack traces, and suggested fixes",
        color="indigo",
        system_prompt=(
            "You are Debug Agent, a careful senior engineer. "
            "Identify the root cause, cite the most relevant files, and propose the smallest safe fix. "
            "Do not speculate beyond the evidence in the retrieved context."
        ),
        output_format=(
            "Return sections named Summary, Root Cause, Relevant Files, Fix Plan, Validation, and Patch Notes."
        ),
        retrieval_bias=("error", "exception", "traceback", "timeout", "stack", "failure", "test"),
        default_top_k=6,
        use_expansion=True,
    ),
    "refactor": AgentDefinition(
        id="refactor",
        name="Refactor Pro",
        role="Architect",
        purpose="Recommends restructuring for clarity, performance, and maintainability while preserving contracts.",
        model=f"Ollama / {settings.gen_model}",
        focus="Legacy cleanup, architecture, and modularization",
        color="emerald",
        system_prompt=(
            "You are Refactor Pro, an architecture-minded senior engineer. "
            "Focus on design smells, module boundaries, duplication, and safe API-preserving refactors."
        ),
        output_format=(
            "Return sections named Summary, Structural Risks, Refactor Plan, Files to Touch, Validation, and Rollout Notes."
        ),
        retrieval_bias=("refactor", "module", "service", "layer", "duplicate", "complexity", "coupling"),
        default_top_k=7,
        use_expansion=True,
    ),
    "docs": AgentDefinition(
        id="docs",
        name="Documentation Bot",
        role="Writer",
        purpose="Generates crisp markdown docs, API references, and onboarding notes from code.",
        model=f"Ollama / {settings.gen_model}",
        focus="Doc generation, summaries, and API descriptions",
        color="violet",
        system_prompt=(
            "You are Documentation Bot, a technical writer with engineering context. "
            "Summarize public APIs, module responsibilities, and usage examples clearly and concisely."
        ),
        output_format=(
            "Return sections named Overview, Public API, Important Files, Example Usage, and Suggested Doc Additions."
        ),
        retrieval_bias=("readme", "docs", "api", "usage", "example", "class", "function"),
        default_top_k=5,
        use_expansion=False,
    ),
    "security": AgentDefinition(
        id="security",
        name="Security Auditor",
        role="Security",
        purpose="Identifies vulnerabilities, unsafe resource access, and injection risks.",
        model=f"Ollama / {settings.gen_model}",
        focus="Auth, validation, secrets, and attack surface review",
        color="amber",
        system_prompt=(
            "You are Security Auditor, a paranoid application security reviewer. "
            "Prioritize injection risks, unsafe file access, auth flows, secrets handling, and over-permissive APIs."
        ),
        output_format=(
            "Return sections named Summary, Findings, Severity, Affected Files, Remediation, and Verification."
        ),
        retrieval_bias=("security", "auth", "token", "secret", "sanitize", "validate", "permission"),
        default_top_k=6,
        use_expansion=True,
    ),
}


def _load_custom_agents() -> None:
    if not _AGENTS_FILE.exists():
        return
    try:
        data = json.loads(_AGENTS_FILE.read_text())
        for item in data:
            agent = AgentDefinition(
                id=item["id"],
                name=item["name"],
                role=item.get("role", "Custom"),
                purpose=item.get("purpose", ""),
                model=item.get("model", f"Ollama / {settings.gen_model}"),
                focus=item.get("focus", ""),
                color=item.get("color", "slate"),
                system_prompt=item.get("system_prompt", ""),
                output_format=item.get("output_format", "Return a clear summary of your findings."),
                retrieval_bias=tuple(item.get("retrieval_bias", [])),
                default_top_k=int(item.get("default_top_k", 6)),
                use_expansion=bool(item.get("use_expansion", True)),
            )
            AGENTS[agent.id] = agent
    except Exception:
        pass


def _save_custom_agents() -> None:
    custom = [a for a in AGENTS.values() if a.id not in {"debug", "refactor", "docs", "security"}]
    _AGENTS_FILE.write_text(json.dumps([asdict(a) for a in custom], indent=2))


def create_agent(
    id: str,
    name: str,
    role: str = "Custom",
    purpose: str = "",
    focus: str = "",
    color: str = "slate",
    system_prompt: str = "",
    output_format: str = "Return a clear summary of your findings.",
    retrieval_bias: list[str] | None = None,
    default_top_k: int = 6,
    use_expansion: bool = True,
) -> AgentCard:
    if not id or not name:
        raise ValueError("Agent 'id' and 'name' are required.")
    agent = AgentDefinition(
        id=id,
        name=name,
        role=role,
        purpose=purpose,
        model=f"Ollama / {settings.gen_model}",
        focus=focus,
        color=color,
        system_prompt=system_prompt,
        output_format=output_format,
        retrieval_bias=tuple(retrieval_bias or []),
        default_top_k=default_top_k,
        use_expansion=use_expansion,
    )
    AGENTS[id] = agent
    _save_custom_agents()
    return AgentCard(
        id=agent.id,
        name=agent.name,
        role=agent.role,
        purpose=agent.purpose,
        model=agent.model,
        focus=agent.focus,
        color=agent.color,
        default_top_k=agent.default_top_k,
        use_expansion=agent.use_expansion,
    )


_load_custom_agents()


def list_agent_cards() -> list[AgentCard]:
    return [
        AgentCard(
            id=agent.id,
            name=agent.name,
            role=agent.role,
            purpose=agent.purpose,
            model=agent.model,
            focus=agent.focus,
            color=agent.color,
            default_top_k=agent.default_top_k,
            use_expansion=agent.use_expansion,
        )
        for agent in AGENTS.values()
    ]


def get_agent(agent_id: str) -> AgentDefinition:
    try:
        return AGENTS[agent_id]
    except KeyError as exc:
        raise ValueError(f"Unknown agent '{agent_id}'.") from exc
