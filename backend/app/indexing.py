"""Index-time pipeline: load files -> chunk -> embed -> store in Chroma.

This is the offline half of the architecture diagram. Run it once per repo and
again whenever files change.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from . import bm25_index, graph_memory, ollama_client
from .ast_chunker import chunk_file
from .config import settings
from .schemas import IndexResponse
from .store import collection
from .graph_memory import extract_imports, save_graph

_HASH_STORE_NAME = ".chroma/file_hashes.json"


def _hash_store_path(root: str) -> Path:
    return Path(root) / _HASH_STORE_NAME


def _load_hashes(root: str) -> dict[str, str]:
    p = _hash_store_path(root)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_hashes(root: str, hashes: dict[str, str]) -> None:
    p = _hash_store_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hashes))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".h", ".cs", ".php", ".sql", ".md", ".yaml", ".yml",
}
SKIP_DIRS = {"node_modules", ".git", "dist", "build", ".venv", "__pycache__", ".chroma"}


def _iter_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in CODE_EXTENSIONS:
                yield os.path.join(dirpath, fn)



def _path_variants(rel_path: str) -> set[str]:
    stem = os.path.splitext(rel_path)[0].replace("\\", "/").lower()
    base = os.path.splitext(os.path.basename(rel_path))[0].lower()
    return {
        stem,
        stem.replace("/", "."),
        stem.replace("/", "_"),
        base,
    }


def _import_variants(name: str) -> set[str]:
    raw = name.strip().strip("'\"`").replace("\\", "/").replace("::", "/").lower()
    raw = re.sub(r"\.(py|js|ts|tsx|jsx|java|go|rs|rb|php|c|cpp|h|cs|sql|md|yaml|yml)$", "", raw)
    raw = raw.lstrip("./")
    dotted = raw.replace("/", ".")
    leaf = os.path.basename(raw)
    return {raw, dotted, raw.replace("/", "_"), leaf}


def _link_graph(
    imports_by_file: dict[str, list[str]],
    chunk_ids_by_file: dict[str, list[str]],
) -> None:
    graph_memory.reset_graph()
    graph = graph_memory.get_graph()

    file_variants = {rel: _path_variants(rel) for rel in imports_by_file}

    for source_rel, imports in imports_by_file.items():
        source_chunk_ids = chunk_ids_by_file.get(source_rel, [])
        if not source_chunk_ids:
            continue

        for imp in imports:
            imp_variants = _import_variants(imp)
            matched_targets = [
                target_rel
                for target_rel, target_variants in file_variants.items()
                if target_rel != source_rel and any(
                    iv == tv or iv.endswith(tv) or tv.endswith(iv) or iv in tv or tv in iv
                    for iv in imp_variants
                    for tv in target_variants
                )
            ]

            for target_rel in matched_targets:
                target_chunk_ids = chunk_ids_by_file.get(target_rel, [])
                for source_chunk_id in source_chunk_ids:
                    for target_chunk_id in target_chunk_ids:
                        graph.add_edge(source_chunk_id, target_chunk_id, relation="import")


def index_repo(path: str, name: str | None = None, incremental: bool = True) -> IndexResponse:
    name = name or os.path.basename(os.path.abspath(path.rstrip("/")))
    col = collection(settings.code_collection)

    known_hashes = _load_hashes(path) if incremental else {}
    new_hashes: dict[str, str] = {}

    n_files = 0
    n_skipped = 0
    total_bytes = 0
    chunk_ids_by_file: dict[str, list[str]] = defaultdict(list)
    imports_by_file: dict[str, list[str]] = {}

    # Pass 1: collect chunks for changed/new files only.
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict] = []

    for fp in _iter_files(path):
        try:
            text = open(fp, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        rel = os.path.relpath(fp, path)
        digest = _sha256(text)
        new_hashes[rel] = digest
        imports_by_file[rel] = extract_imports(rel, text)
        total_bytes += len(text.encode("utf-8"))
        n_files += 1

        if incremental and known_hashes.get(rel) == digest:
            # File unchanged — reuse existing chunk IDs from Chroma without re-embedding.
            existing = col.get(where={"path": rel}, include=[], limit=10_000)  # type: ignore[arg-type]
            for cid in (existing.get("ids") or []):
                chunk_ids_by_file[rel].append(cid)
            n_skipped += 1
            continue

        ext = os.path.splitext(fp)[1].lower()
        for j, (snippet, lines) in enumerate(chunk_file(ext, text, settings.chunk_size, settings.chunk_overlap)):
            cid = f"{rel}:{j}"
            all_ids.append(cid)
            all_docs.append(snippet)
            all_metas.append({"path": rel, "lines": lines})
            chunk_ids_by_file[rel].append(cid)

    # Pass 2: embed only new/changed chunks concurrently.
    all_embeds = ollama_client.embed_batch(all_docs) if all_ids else []
    n_new_chunks = len(all_ids)

    # Pass 3: upsert changed chunks into Chroma in batches of 128.
    for i in range(0, n_new_chunks, 128):
        col.upsert(
            ids=all_ids[i : i + 128],
            documents=all_docs[i : i + 128],
            embeddings=all_embeds[i : i + 128],
            metadatas=all_metas[i : i + 128],
        )

    # Pass 4: file-level summary index — reuse each file's first chunk embedding.
    # No extra LLM calls: the embedding is already in all_embeds at the same index.
    summary_col = collection(settings.summary_collection)
    cid_to_embed = dict(zip(all_ids, all_embeds))
    summary_ids, summary_docs, summary_embeds, summary_metas = [], [], [], []
    seen_files: set[str] = set()
    for cid, doc, meta in zip(all_ids, all_docs, all_metas):
        rel = meta["path"]
        if rel in seen_files:
            continue
        seen_files.add(rel)
        summary_ids.append(f"summary:{rel}")
        summary_docs.append(doc)
        summary_embeds.append(cid_to_embed[cid])
        summary_metas.append({"path": rel})
    if summary_ids:
        summary_col.upsert(ids=summary_ids, documents=summary_docs,
                           embeddings=summary_embeds, metadatas=summary_metas)

    # Delete chunks + summaries for files that no longer exist in the repo.
    deleted_file_rels = set(known_hashes.keys()) - set(new_hashes.keys())
    for del_rel in deleted_file_rels:
        existing = col.get(where={"path": del_rel}, include=[], limit=10_000)  # type: ignore[arg-type]
        del_ids = existing.get("ids") or []
        if del_ids:
            col.delete(ids=del_ids)
        try:
            collection(settings.summary_collection).delete(ids=[f"summary:{del_rel}"])
        except Exception:
            pass

    _save_hashes(path, new_hashes)

    # BM25 incremental update — only re-tokenize chunks from changed/new/deleted files.
    # Avoids the 500k-document Chroma fetch; unchanged chunks stay in the corpus_map.
    changed_file_rels = {
        rel for rel, digest in new_hashes.items()
        if known_hashes.get(rel) != digest
    }
    bm25_index.update(changed_file_rels | deleted_file_rels, all_ids, all_docs)
    bm25_index.save()

    _link_graph(imports_by_file, chunk_ids_by_file)
    save_graph()

    n_chunks = col.count()
    return IndexResponse(
        name=name,
        files=n_files,
        chunks=n_chunks,
        indexed_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        embed_model=settings.embed_model,
        size_mb=round(total_bytes / 1_048_576, 1),
    )
