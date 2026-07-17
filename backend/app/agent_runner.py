"""Agent orchestration for specialized engineering workflows.

This module sits on top of the RAG pipeline and turns the general retrieval
stack into role-based workflows: Debug, Refactor, Docs, and Security.
"""
from __future__ import annotations

import json
import os
import re
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from . import generation, indexing, optimizer, retrieval
from .agent_registry import AgentDefinition, get_agent, list_agent_cards
from .config import settings
from .schemas import (
    AgentFinding,
    AgentLogItem,
    AgentRunRequest,
    AgentRunResult,
    CodebaseSummary,
    RetrievedChunk,
)
from .store import collection, count_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SourceEvent = Callable[[str, Any], None]

CODE_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript/React",
    ".jsx": "JavaScript/React",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++",
    ".cs": "C#",
    ".php": "PHP",
    ".sql": "SQL",
    ".md": "Markdown",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
}

SKIP_DIRS = {"node_modules", ".git", "dist", "build", ".venv", "__pycache__", ".chroma"}


def list_agents():
    return list_agent_cards()


def _resolve_target_path(path: str, base: Path | None = None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (base or PROJECT_ROOT) / path
    return candidate.resolve()


def _relative_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix() or "."
    except ValueError:
        return path.as_posix()


def _infer_agent_id(request: AgentRunRequest) -> str:
    text = f"{request.user_request}\n{request.task_type or ''}".lower()
    logs = request.logs.lower()

    if re.search(r"\b(launch|run|start|setup|install|readme|document|docs|onboard)\b", text):
        return "docs"
    if re.search(r"\b(security|audit|vulnerab|auth|token|secret|permission|injection)\b", text):
        return "security"
    if re.search(r"\b(refactor|cleanup|architecture|modular|maintain|performance)\b", text):
        return "refactor"
    if re.search(r"\b(error|bug|fix|timeout|exception|crash|failed|failure|memory|504)\b", f"{text}\n{logs}"):
        return "debug"
    return "debug"


def _iter_source_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if Path(name).suffix.lower() in CODE_EXTENSIONS:
                yield Path(dirpath) / name


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _normalize_terms(*parts: str) -> list[str]:
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        tokens.extend(
            tok
            for tok in re.split(r"[^a-zA-Z0-9_]+", part.lower())
            if tok and len(tok) > 2
        )
    return list(dict.fromkeys(tokens))


def _path_matches(path: str, prefix: str) -> bool:
    if not prefix:
        return True
    norm_path = path.replace("\\", "/").lower()
    norm_prefix = prefix.replace("\\", "/").lower().rstrip("/")
    return norm_path == norm_prefix or norm_path.startswith(f"{norm_prefix}/")


def _detect_codebase_summary(target: Path, request: AgentRunRequest, agent: AgentDefinition) -> CodebaseSummary:
    languages: set[str] = set()
    frameworks: set[str] = set()
    services: set[str] = set()
    storage: set[str] = set()
    signals: set[str] = set()
    risk_flags: set[str] = set()
    file_count = 0

    request_terms = _normalize_terms(request.user_request, request.logs, request.task_type or "", agent.focus, *agent.retrieval_bias)
    targeted_terms = request_terms[:18]

    for fp in _iter_source_files(target):
        file_count += 1
        ext = fp.suffix.lower()
        if ext in CODE_EXTENSIONS:
            languages.add(CODE_EXTENSIONS[ext])

        text = _read_text(fp)
        lower = text.lower()
        rel = fp.relative_to(target).as_posix().lower()

        if "fastapi" in lower or "from fastapi" in lower:
            frameworks.add("FastAPI")
        if "react" in lower or "tsx" in fp.suffix.lower() or "vite" in lower:
            frameworks.add("React/Vite")
        if "sqlalchemy" in lower:
            storage.add("SQLAlchemy")
        if "postgres" in lower or "psycopg" in lower:
            storage.add("PostgreSQL")
        if "chromadb" in lower or "chroma" in lower:
            storage.add("Chroma")
        if "ollama" in lower:
            frameworks.add("Ollama")
        if "groq" in lower:
            frameworks.add("Groq")
        if "pytest" in lower:
            services.add("Tests")
        if "router" in rel or "route" in rel or "@app." in lower:
            services.add("API routes")
        if "middleware" in rel or "middleware" in lower:
            services.add("Middleware")
        if "parser" in rel or "parse" in rel:
            services.add("Parser")
        if "ingest" in rel or "ingestion" in rel:
            services.add("Ingestion")
        if "auth" in rel or "auth" in lower:
            services.add("Auth")
        if "db" in rel or "database" in lower or "models" in rel:
            services.add("Database layer")

        for term in targeted_terms:
            if term in rel or term in lower:
                signals.add(term)

        if agent.id == "security":
            if re.search(r"\beval\s*\(", text) or re.search(r"\bexec\s*\(", text):
                risk_flags.add("Dynamic code execution detected")
            if re.search(r"os\.system\s*\(", text) or re.search(r"subprocess\.(run|call|Popen)\s*\(", text):
                risk_flags.add("Shell execution surface detected")
            if re.search(r"open\s*\([^)]*user", text, re.IGNORECASE):
                risk_flags.add("Potential user-controlled file access")
            if re.search(r"api[_-]?key|secret|password|token", lower):
                risk_flags.add("Secrets or auth material referenced in code")
            if re.search(r"request\.(args|json|form|query)", lower) and re.search(r"exec|eval|sql", lower):
                risk_flags.add("Input validation should be reviewed")

        if len(signals) >= 8 and file_count >= 40:
            break

    if not signals:
        signals.update(
            term
            for term in _normalize_terms(request.user_request, request.logs)
            if term in {"timeout", "memory", "refactor", "security", "auth", "parser", "ingestion", "docs"}
        )

    return CodebaseSummary(
        root_path=str(target.relative_to(PROJECT_ROOT)) if target.is_relative_to(PROJECT_ROOT) else str(target),
        file_count=file_count,
        languages=sorted(languages),
        frameworks=sorted(frameworks),
        services=sorted(services),
        storage=sorted(storage),
        signals=sorted(signals),
        risk_flags=sorted(risk_flags),
    )


def _boost_chunks(
    chunks: list[RetrievedChunk],
    target_path: str,
    agent: AgentDefinition,
    extra_terms: list[str],
    limit: int,
) -> list[RetrievedChunk]:
    def score(chunk: RetrievedChunk) -> float:
        value = float(chunk.score)
        if _path_matches(chunk.path, target_path):
            value += 0.18
        lowered = f"{chunk.path}\n{chunk.snippet}".lower()
        for term in agent.retrieval_bias:
            if term in lowered:
                value += 0.03
        for term in extra_terms:
            if term in lowered:
                value += 0.02
        return value

    ranked = sorted(chunks, key=score, reverse=True)
    return ranked[:limit]


_PLAN_SYSTEM = (
    "You are a senior engineering lead reviewing a codebase. "
    "Given agent context and retrieved code chunks, output EXACTLY 3 numbered action steps "
    "that form an execution plan for the task. "
    "Each step must be a single sentence on its own line starting with the number and a period (e.g. '1. ...'). "
    "No extra commentary, headers, or blank lines between steps."
)

_FINDINGS_SYSTEM = (
    "You are a senior engineering lead reviewing a codebase. "
    "Given agent context and retrieved code chunks, output 1–3 specific findings as JSON. "
    "Each finding must have: severity (high|medium|low), title (short string), detail (1–2 sentence explanation). "
    "Output ONLY a valid JSON array, no markdown fences or commentary. "
    'Example: [{"severity":"high","title":"Blocking path","detail":"The code ..."}]'
)


def _chunk_context(chunks: list[RetrievedChunk], max_chars: int = 3000) -> str:
    parts: list[str] = []
    total = 0
    for c in chunks[:6]:
        block = f"// {c.path} ({c.lines})\n{c.snippet[:800]}"
        total += len(block)
        parts.append(block)
        if total >= max_chars:
            break
    return "\n\n".join(parts)


def _derive_agent_plan(
    agent: AgentDefinition,
    summary: CodebaseSummary,
    chunks: list[RetrievedChunk],
    request: AgentRunRequest,
    model: str,
) -> list[str]:
    context = _chunk_context(chunks)
    prompt = (
        f"Agent: {agent.name} ({agent.role})\n"
        f"Task: {request.user_request.strip()}\n"
        f"Target: {request.target_path}\n"
        f"Languages: {', '.join(summary.languages) or 'unknown'}\n"
        f"Risk flags: {', '.join(summary.risk_flags) or 'none'}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Output the 3-step execution plan now."
    )
    try:
        raw = generation.generate(prompt, model=model, system_prompt=_PLAN_SYSTEM).strip()
        steps = [
            re.sub(r"^\d+[.)]\s*", "", line).strip()
            for line in raw.splitlines()
            if re.match(r"^\d+[.)]\s+\S", line.strip())
        ]
        if len(steps) >= 2:
            return steps[:3]
    except Exception:
        pass
    # Fallback
    return [
        f"Review the {agent.focus} of the target path using the retrieved chunks.",
        "Identify the primary concern and isolate the minimal intervention.",
        "Validate the proposed change against the existing contract.",
    ]


def _derive_agent_findings(
    agent: AgentDefinition,
    summary: CodebaseSummary,
    chunks: list[RetrievedChunk],
    request: AgentRunRequest,
    model: str,
) -> list[AgentFinding]:
    top_paths = [c.path for c in chunks[:3]]
    context = _chunk_context(chunks)
    prompt = (
        f"Agent: {agent.name} ({agent.role})\n"
        f"Task: {request.user_request.strip()}\n"
        f"Logs: {request.logs.strip() or 'none'}\n"
        f"Risk flags: {', '.join(summary.risk_flags) or 'none'}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Output 1–3 findings as a JSON array now."
    )
    try:
        raw = generation.generate(prompt, model=model, system_prompt=_FINDINGS_SYSTEM).strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        findings: list[AgentFinding] = []
        for item in parsed[:3]:
            if isinstance(item, dict) and "title" in item:
                findings.append(AgentFinding(
                    severity=str(item.get("severity", "medium")),
                    title=str(item["title"]),
                    detail=str(item.get("detail", "")),
                    files=top_paths,
                ))
        if findings:
            return findings
    except Exception:
        pass
    # Fallback: one generic finding derived from risk flags or request
    severity = "high" if summary.risk_flags else "medium"
    detail = (
        f"Risk patterns detected: {', '.join(summary.risk_flags[:2])}." if summary.risk_flags
        else f"Review the {agent.focus} related to: {request.user_request[:80].strip()}."
    )
    return [AgentFinding(severity=severity, title=f"{agent.name} finding", detail=detail, files=top_paths)]


def _build_prompt(
    agent: AgentDefinition,
    request: AgentRunRequest,
    summary: CodebaseSummary,
    chunks: list[RetrievedChunk],
) -> tuple[str, list[RetrievedChunk]]:
    header = f"""# AGENT: {agent.name}
# ROLE: {agent.role}

## MISSION
{agent.purpose}

## TARGET
Path: {request.target_path}
Task type: {request.task_type or 'Unspecified'}

## USER REQUEST
{request.user_request.strip()}
"""
    if request.logs.strip():
        header += f"\n## LOGS\n{request.logs.strip()}\n"

    attachment_items = [*request.attachments, *request.attachment_names]
    if attachment_items:
        attachment_block = "\n".join(f"- {item}" for item in attachment_items)
        header += f"\n## ATTACHMENTS\n{attachment_block}\n"

    header += "\n## CODEBASE INTELLIGENCE\n"
    header += f"- Languages: {', '.join(summary.languages) or 'Unknown'}\n"
    header += f"- Frameworks: {', '.join(summary.frameworks) or 'Unknown'}\n"
    header += f"- Services: {', '.join(summary.services) or 'Unknown'}\n"
    header += f"- Storage: {', '.join(summary.storage) or 'Unknown'}\n"
    header += f"- Signals: {', '.join(summary.signals) or 'None'}\n"
    header += f"- Risk flags: {', '.join(summary.risk_flags) or 'None'}\n"

    footer = f"""
## OUTPUT FORMAT
{agent.output_format}

## INSTRUCTIONS
1. Keep the response grounded in the target path and the retrieved files.
2. Prefer the smallest safe change or the most direct documentation/audit finding.
3. Use the system analysis to explain the why, not just the what.
4. If the evidence is incomplete, say so clearly and suggest the next verification step.
"""

    budget = settings.token_budget - count_tokens(header) - count_tokens(footer)
    body_parts: list[str] = []
    kept: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        block = f"// {chunk.path} ({chunk.lines}) — relevancy {int(chunk.score * 100)}%\n{chunk.snippet}"
        t = count_tokens(block)
        if kept and used + t > budget:
            break
        body_parts.append(block)
        kept.append(chunk)
        used += t

    prompt = header
    if body_parts:
        prompt += "\n## RELEVANT FILES\n" + "\n\n".join(body_parts) + "\n"
    prompt += footer
    return prompt, kept


def _build_patch_prompt(
    agent: AgentDefinition,
    request: AgentRunRequest,
    summary: CodebaseSummary,
    chunks: list[RetrievedChunk],
    findings: list[AgentFinding],
    plan: list[str],
) -> str:
    file_blocks = "\n\n".join(
        f"### {chunk.path} ({chunk.lines})\n{chunk.snippet}" for chunk in chunks[: min(4, len(chunks))]
    )
    findings_block = "\n".join(
        f"- [{finding.severity}] {finding.title}: {finding.detail}" for finding in findings
    ) or "- None"
    plan_block = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan)) or "1. Review the target files and propose the smallest safe change."
    attachment_items = [*request.attachments, *request.attachment_names]
    attachments_block = "\n".join(f"- {item}" for item in attachment_items) or "- None"

    return f"""You are {agent.name}. Draft a patch diff only.

Goal:
Generate a minimal, reviewable unified diff for the codebase task below.

Rules:
- Output ONLY a unified diff.
- Use only files that appear in the relevant files list.
- If the context is insufficient for a safe diff, output exactly: NO_PATCH
- Do not include commentary, markdown fences, or explanations.

Task:
{request.user_request.strip()}

Target path:
{request.target_path}

Task type:
{request.task_type or 'Unspecified'}

Codebase intelligence:
- Languages: {', '.join(summary.languages) or 'Unknown'}
- Frameworks: {', '.join(summary.frameworks) or 'Unknown'}
- Services: {', '.join(summary.services) or 'Unknown'}
- Storage: {', '.join(summary.storage) or 'Unknown'}
- Signals: {', '.join(summary.signals) or 'None'}
- Risk flags: {', '.join(summary.risk_flags) or 'None'}

Relevant files:
{file_blocks or 'None'}

Findings:
{findings_block}

Plan:
{plan_block}

Attachments:
{attachments_block}
"""


def _clean_patch_output(output: str) -> str:
    text = output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:diff|patch)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
    if text.upper().strip() == "NO_PATCH":
        return ""
    return text


def _generate_patch_draft(
    agent: AgentDefinition,
    request: AgentRunRequest,
    summary: CodebaseSummary,
    chunks: list[RetrievedChunk],
    findings: list[AgentFinding],
    plan: list[str],
    model: str,
) -> tuple[str, list[str]]:
    prompt = _build_patch_prompt(agent, request, summary, chunks, findings, plan)
    draft = _clean_patch_output(generation.generate(prompt, model))
    files = list(dict.fromkeys(chunk.path for chunk in chunks[: min(4, len(chunks))]))
    return draft, files


def _build_retrieval_queries(
    agent: AgentDefinition,
    request: AgentRunRequest,
    summary: CodebaseSummary,
    optimized: str,
    expansions: list[str],
) -> list[str]:
    queries = [optimized]
    if request.logs.strip():
        queries.append(request.logs.strip())
    queries.extend(expansions)
    queries.extend(agent.retrieval_bias)
    if request.task_type:
        queries.append(request.task_type)
    if summary.signals:
        queries.extend(summary.signals)
    return [q for q in queries if q.strip()]


def _generate_agent_answer(prompt: str, model: str, system_prompt: str | None = None) -> str:
    return generation.generate(prompt, model, system_prompt=system_prompt)


def _run_agent_internal(
    request: AgentRunRequest,
    emit: SourceEvent | None = None,
) -> AgentRunResult:
    from . import semantic_router  # local import avoids circular dependency at module load

    # Off-topic guard — applied in all modes before any expensive work.
    if semantic_router._classify(request.user_request) == "off_topic":
        raise ValueError(semantic_router.OFF_TOPIC_REPLY)

    started = time.time()
    resolved_agent_id = _infer_agent_id(request) if request.agent_id in {"", "auto"} else request.agent_id
    agent = get_agent(resolved_agent_id)
    # Agent runs always use the large model for deep reasoning.
    model = request.model or settings.large_model
    logs: list[AgentLogItem] = []

    def stage(name: str, message: str):
        item = AgentLogItem(stage=name, message=message)
        logs.append(item)
        if emit:
            emit("log", item.model_dump(by_alias=True))

    if emit:
        emit("stage", "intake")
    repo_root = _resolve_target_path(request.repo_path or ".")
    if not repo_root.exists() or not repo_root.is_dir():
        raise RuntimeError(f"Connected repo path not found: {request.repo_path}")

    target = _resolve_target_path(request.target_path or ".", repo_root)
    if not target.exists():
        stage(
            "intake",
            f"Target path '{request.target_path}' was not found inside the connected repo, so the agent will analyze the repo root.",
        )
        target = repo_root
    target_scope = _relative_path(target, repo_root)
    stage("intake", f"Using {agent.name} against '{target_scope}' in repo '{repo_root.name}'.")
    request = request.model_copy(update={"target_path": target_scope})

    code_col = collection(settings.code_collection)
    if code_col.count() == 0:
        if emit:
            emit("stage", "indexing")
        stage("indexing", "No indexed repository context was found, so the connected repo is being indexed first.")
        indexed = indexing.index_repo(str(repo_root), repo_root.name)
        stage("indexing", f"Indexed {indexed.files} files into {indexed.chunks} chunks.")

    if not target.exists():
        raise RuntimeError(f"Target path not found: {request.target_path}")

    if emit:
        emit("stage", "analysis")
    summary = _detect_codebase_summary(target, request, agent)
    stage(
        "analysis",
        f"Detected {len(summary.languages) or 0} language families and {summary.file_count} files in the target folder.",
    )
    if emit:
        emit("artifact", {"kind": "architecture", "data": summary.model_dump(by_alias=True)})

    raw_query = request.user_request.strip()
    if request.logs.strip():
        raw_query = f"{raw_query}\n\nRuntime logs:\n{request.logs.strip()}" if raw_query else request.logs.strip()
    if not raw_query:
        raw_query = f"{agent.name} request against {request.target_path}"

    if emit:
        emit("stage", "query-optimization")
    do_expand = request.use_expansion and agent.use_expansion
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_opt = pool.submit(optimizer.optimize_query, raw_query, model)
        f_exp = pool.submit(optimizer.expand_query, raw_query, model) if do_expand else None
        optimized = f_opt.result()
        expansions = f_exp.result() if f_exp else []
    stage("query-optimization", f"Optimized the request into: {optimized}")
    if expansions:
        stage("query-optimization", f"Generated {len(expansions)} expansion query variants.")

    if emit:
        emit("stage", "retrieval")
    retrieval_queries = _build_retrieval_queries(agent, request, summary, optimized, expansions)
    retrieved = retrieval.retrieve(retrieval_queries, max(request.top_k, agent.default_top_k))
    if not retrieved:
        if emit:
            emit("stage", "indexing")
        stage("indexing", "Retrieval returned no chunks, so the connected repo is being re-indexed.")
        indexed = indexing.index_repo(str(repo_root), repo_root.name)
        stage("indexing", f"Indexed {indexed.files} files into {indexed.chunks} chunks.")
        retrieved = retrieval.retrieve(retrieval_queries, max(request.top_k, agent.default_top_k))
    ranked = _boost_chunks(retrieved, target_scope, agent, summary.signals, max(request.top_k, agent.default_top_k))
    stage("retrieval", f"Retrieved {len(ranked)} relevant chunks from the indexed codebase.")
    if emit:
        emit("artifact", {"kind": "retrieval", "data": [chunk.model_dump(by_alias=True) for chunk in ranked]})

    if emit:
        emit("stage", "planning")
    plan = _derive_agent_plan(agent, summary, ranked, request, model)
    findings = _derive_agent_findings(agent, summary, ranked, request, model)
    stage("planning", f"Prepared a {len(plan)} step execution plan and {len(findings)} findings seed(s).")

    prompt, kept = _build_prompt(agent, request, summary, ranked)
    stage("assembly", f"Assembled a prompt with {len(kept)} context chunk(s) within the token budget.")
    if emit:
        emit("artifact", {"kind": "prompt", "data": {"prompt": prompt, "context_chunks": len(kept)}})

    if emit:
        emit("stage", "generation")
    answer = _generate_agent_answer(prompt, model, system_prompt=agent.system_prompt)
    stage("generation", f"Generated {len(answer)} characters of agent guidance.")

    if emit:
        emit("stage", "patching")
    patch_diff, patch_files = _generate_patch_draft(agent, request, summary, kept, findings, plan, model)
    if patch_diff:
        stage("patching", f"Drafted a patch proposal touching {len(patch_files)} file(s).")
        if emit:
            emit("artifact", {"kind": "patch", "data": {"files": patch_files, "diff": patch_diff}})
    else:
        stage("patching", "No safe patch draft could be produced from the available context.")

    latency_ms = int((time.time() - started) * 1000)
    summary_text = f"{agent.name} analyzed {summary.file_count} files under {target_scope} and returned {len(kept)} grounded context chunk(s)."
    result = AgentRunResult(
        agent_id=agent.id,
        agent_name=agent.name,
        model=model,
        target_path=target_scope,
        summary=summary_text,
        architecture=summary,
        relevant_files=kept,
        findings=findings,
        plan=plan,
        prompt=prompt,
        answer=answer,
        patch_diff=patch_diff,
        patch_files=patch_files,
        logs=logs,
        latency_ms=latency_ms,
    )
    return result


def run_agent(request: AgentRunRequest) -> AgentRunResult:
    return _run_agent_internal(request)


def run_agent_stream(request: AgentRunRequest):
    sentinel = object()
    event_queue: queue.Queue = queue.Queue()

    def emit(event: str, data: Any):
        event_queue.put((event, data))

    def worker():
        try:
            result = _run_agent_internal(request, emit=emit)
            event_queue.put(("done", result))
        except Exception as exc:
            event_queue.put(("error", str(exc)))
        finally:
            event_queue.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    try:
        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            event, data = item
            if event == "done":
                yield f"event: done\ndata: {data.model_dump_json(by_alias=True)}\n\n"
            elif event == "error":
                yield f"event: error\ndata: {json.dumps(data)}\n\n"
            else:
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
    except Exception as exc:
        yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
