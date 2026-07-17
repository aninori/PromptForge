# PromptForge — Complete Documentation
> Written for someone who knows nothing about RAG, LLMs, embeddings, or vector databases.
> Every concept is explained from first principles before showing how it applies to PromptForge.

---

## Table of Contents

1. [What is PromptForge?](#1-what-is-promptforge)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [Core Concepts — Explained from Scratch](#3-core-concepts--explained-from-scratch)
   - 3.1 Large Language Models (LLMs)
   - 3.2 Tokens
   - 3.3 Context Windows
   - 3.4 Embeddings
   - 3.5 Vector Databases
   - 3.6 Chunking
   - 3.7 Retrieval-Augmented Generation (RAG)
   - 3.8 Semantic Search vs Keyword Search
   - 3.9 Semantic Caching
   - 3.10 Server-Sent Events (SSE)
   - 3.11 Agents
   - 3.12 Ollama
4. [System Architecture — Full Walkthrough](#4-system-architecture--full-walkthrough)
   - 4.1 The Two Phases
   - 4.2 Phase 1 — Indexing (Offline)
   - 4.3 Phase 2 — Query (Runtime)
   - 4.4 Agent Mode
5. [Every File Explained](#5-every-file-explained)
6. [Technology Stack — Why Each Tool Was Chosen](#6-technology-stack--why-each-tool-was-chosen)
7. [Configuration Reference](#7-configuration-reference)
8. [API Endpoints Reference](#8-api-endpoints-reference)
9. [Known Flaws — What is Broken and Why](#9-known-flaws--what-is-broken-and-why)
10. [The Fixed Architecture — What It Should Look Like](#10-the-fixed-architecture--what-it-should-look-like)
11. [Setup and Running](#11-setup-and-running)
12. [Glossary](#12-glossary)

---

## 1. What is PromptForge?

PromptForge is a **local, privacy-first AI assistant for your own codebase**. You point it at a folder of code, it reads every file, and then you can ask it questions in plain English:

- "Where is rate limiting implemented?"
- "Add error handling to the payment service"
- "Find all functions that call the database"
- "Why is this test failing?"

It answers by actually reading your code — not by guessing from training data. Everything runs on your own computer. No code ever leaves your machine. No API costs per question.

Think of it as having a senior developer sitting next to you who has read every file in your project and can answer questions about it instantly.

---

## 2. The Problem It Solves

### The core problem with AI coding assistants

When you ask ChatGPT or GitHub Copilot about your codebase, they do not actually read your code. They answer from their training data — patterns they saw across millions of public repositories. This creates two problems:

**Problem 1 — They do not know your specific code.**
Your `PaymentService` class, your database schema, your internal API contracts — the model has never seen any of it. So it guesses, and the guesses look confident but are often wrong.

**Problem 2 — Your codebase does not fit in a single prompt.**
Even if you copy-pasted your entire codebase into the chat, it would not fit. Modern large codebases can be millions of lines. The AI can only see a few thousand lines at a time (this limit is called the context window, explained in Section 3.3).

### How PromptForge solves it

PromptForge uses a technique called **RAG** (Retrieval-Augmented Generation). Instead of putting your entire codebase in every prompt, it:

1. Pre-reads and indexes every file (done once, offline)
2. When you ask a question, finds only the relevant pieces of code
3. Puts just those relevant pieces into the prompt
4. Lets the AI answer using that targeted context

This way, the AI only sees what it needs to see, the answer is grounded in your actual code, and the whole thing runs locally on your machine.

---

## 3. Core Concepts — Explained from Scratch

### 3.1 Large Language Models (LLMs)

An LLM is a program that predicts text. Given some text as input (called a **prompt**), it produces more text as output (called a **completion** or **response**).

Under the hood, an LLM is a massive neural network — billions of mathematical parameters trained on hundreds of billions of words of text and code from the internet. Through training, it learned statistical patterns: what words tend to follow other words, what code patterns solve what problems, how to explain things, and so on.

**Important:** an LLM does not "know" things the way a human does. It does not have access to the internet at query time. It can only work with (a) what it learned during training and (b) what you put in the prompt. PromptForge exploits point (b) — it puts your actual code into the prompt so the model has real, accurate context to work with.

**PromptForge uses two LLMs via Ollama:**
- `deepseek-coder-v2:16b` — a 16 billion parameter model trained specifically on code, used to generate answers
- `nomic-embed-text` — a smaller model used only to create embeddings (see Section 3.4)

### 3.2 Tokens

LLMs do not process text character by character or word by word. They process **tokens** — small chunks of text that are roughly 3–4 characters each on average.

Some examples:
- The word "hello" = 1 token
- The word "unbelievable" = 3–4 tokens
- A line of Python code like `def authenticate(user, password):` = about 10 tokens

Tokens matter because:
- Models have a maximum number of tokens they can process at once (the context window)
- Processing time scales with token count — more tokens = slower response
- PromptForge's default token budget for assembling context is 6,000 tokens

### 3.3 Context Windows

Imagine the LLM has a desk. The context window is the size of that desk. Everything the model can "see" when generating an answer must fit on that desk — the question, the retrieved code, the instructions, everything.

- `deepseek-coder-v2:16b` has a context window of roughly 128,000 tokens
- But PromptForge deliberately limits the assembled prompt to 6,000 tokens (`CF_TOKEN_BUDGET`)
- This is intentional — large prompts are slower, more expensive (if using cloud APIs), and can confuse the model with irrelevant context

The art of a good RAG system is fitting the *right* code into that budget, not the *most* code.

### 3.4 Embeddings

This is the most important concept to understand for RAG.

An embedding is a way of turning text into a list of numbers — a **vector** — such that similar text produces similar numbers.

**Concrete example:**

```
"authenticate user login"   → [0.21, -0.87, 0.44, 0.09, ...]   (768 numbers)
"user login validation"     → [0.19, -0.85, 0.47, 0.11, ...]   (768 numbers, very similar)
"render the homepage"       → [-0.33, 0.12, -0.55, 0.78, ...]  (768 numbers, very different)
```

The embedding model (`nomic-embed-text`) learned to produce these vectors during its own training. The key property: **text that means the same thing produces vectors that are close together in mathematical space**, even if the exact words are different.

This means you can ask "where does user login happen?" and find a function called `check_credentials()` — because their embeddings are mathematically close — even though they share no keywords.

**How similarity is measured:**

The similarity between two vectors is calculated using cosine similarity — a number between -1 and 1 where 1 means identical meaning and 0 means unrelated. PromptForge uses this to find the most relevant code chunks for any query.

### 3.5 Vector Databases

A vector database is a special database optimised for storing and searching embeddings.

**Normal database query:** "Find all rows where `user_id = 42`" — exact match

**Vector database query:** "Find the 5 vectors most similar to this query vector" — approximate semantic match

PromptForge uses **ChromaDB** as its vector database. During indexing, it stores:
- The embedding vector (768 numbers from `nomic-embed-text`)
- The original code text (the actual source code chunk)
- Metadata (file path, start line, language, etc.)

During query time, it finds the chunks whose embedding vectors are closest to the query's embedding vector. Those are the most semantically relevant code chunks.

### 3.6 Chunking

You cannot embed an entire file as one embedding — it would be too long, and the embedding would average out all the meaning until it is too vague to be useful. You need to split files into smaller pieces first. Those pieces are called **chunks**.

**PromptForge's current approach (and why it is problematic):**

PromptForge splits every file into 800-token windows with 120-token overlap. This is called fixed-size chunking.

```
File: auth_service.py (2000 tokens)

Chunk 1: lines 1–80   (800 tokens)  ← may end in the middle of a function
Chunk 2: lines 65–145 (800 tokens)  ← starts in the middle of a function
Chunk 3: lines 130–210 (800 tokens) ← etc.
```

The 120-token overlap means adjacent chunks share some content, reducing the chance of cutting a sentence in half. But it does not prevent cutting a function in half.

**The problem:** A function that is 200 lines long gets split across three chunks. No single chunk contains the complete function. When the retriever finds "chunk 2" it only has the middle of the function — no signature, no return statement. The LLM receives an incomplete picture and produces a wrong or incomplete answer.

**The better approach (not yet implemented):** AST-aware chunking using tree-sitter. This parses the code into its syntax tree and splits at function and class boundaries — so each chunk is always a complete, syntactically valid unit.

### 3.7 Retrieval-Augmented Generation (RAG)

RAG is the core technique that makes PromptForge work. It has two phases:

**Phase 1 — Indexing (done once, offline):**
```
Your codebase
    → Read every file
    → Split into chunks
    → Embed each chunk (nomic-embed-text produces a vector for each)
    → Store vectors + original text in ChromaDB
```

**Phase 2 — Query (done on every question):**
```
Your question
    → Embed the question (same model, same vector space)
    → Search ChromaDB for the most similar chunk vectors
    → Retrieve the top-K chunks (actual code text)
    → Assemble into a prompt: "Here is relevant code: [chunks]. Answer: [question]"
    → Send prompt to deepseek-coder-v2:16b
    → Return the answer
```

The key insight: the question and the code chunks exist in the same vector space because they were embedded by the same model. Similarity in that vector space = semantic relevance. So "find all functions that validate passwords" will retrieve code chunks containing password validation logic even if neither the word "validate" nor "password" appears in the code's comments.

### 3.8 Semantic Search vs Keyword Search

**Keyword search (traditional):** Find documents containing the exact words in the query. Fast, precise for exact terms, fails for synonyms, paraphrases, or when the user does not know the exact function name.

**Semantic search (what PromptForge uses for dense retrieval):** Find documents by meaning. Slow to index (embedding is expensive), but finds relevant content even when exact words differ.

**The gap in PromptForge:** Semantic search is bad at exact identifier matching. If you ask about `RateLimiterMiddleware`, dense search finds "semantically similar" code — which might be a completely different throttling class. It should also match the literal string `RateLimiterMiddleware` with BM25 (a keyword search algorithm), then combine both results. This is called hybrid retrieval and is one of PromptForge's known missing features.

### 3.9 Semantic Caching

Normal caching: "If the exact same question has been asked before, return the stored answer."

**Semantic caching:** "If a sufficiently similar question has been asked before, return the stored answer."

PromptForge's `cache.py` stores every query and its result. When a new query arrives:
1. Embed the new query
2. Compare it to all stored query embeddings using cosine similarity
3. If the closest match is above a threshold (currently 0.92), return the cached result — skip the expensive retrieval + generation entirely

This is valuable for performance but dangerous if the threshold is too low. "Add rate limiting to the login route" and "Add rate limiting to the API gateway" might score above 0.92 similarity but require completely different answers. PromptForge's threshold of 0.92 is calibrated for FAQ-style chatbots, not code queries. For code, 0.97 is the correct setting.

### 3.10 Server-Sent Events (SSE)

When the LLM generates text, it does so token by token — one word at a time. SSE is a technology that streams these tokens to the browser as they are generated, so you see the answer appearing word-by-word rather than waiting for the entire response to be computed first.

SSE works as a one-way channel from server to browser over a normal HTTP connection. The server sends `data: token\n\n` messages as each token arrives, and the browser JavaScript appends them to the UI.

**Why this matters for PromptForge's bugs:** PromptForge implements SSE for both `/query/stream` and `/agents/stream`. The agent streaming path uses a background thread plus a Python Queue to communicate between the worker (running the LLM) and the web server (sending SSE events). If the worker thread crashes without putting a termination signal on the queue, the web server blocks on `queue.get()` forever. This is one of the root causes of the hang symptom.

### 3.11 Agents

An agent is an LLM given a specific persona, a structured task, and a set of tools or steps it follows.

In PromptForge, four agents are defined in `agent_registry.py`:

| Agent | Persona | What it does |
|-------|---------|--------------|
| Debug Agent | Bug triage specialist | Analyzes stack traces, pinpoints bugs, generates surgical patch diffs |
| Refactor Pro | Software architect | Identifies legacy patterns, suggests modular restructuring |
| Doc Bot | Technical writer | Generates docstrings, API references, README sections |
| Security Auditor | Security reviewer | Scans for injection risks, auth flaws, exposed secrets |

Each agent is essentially a different system prompt plus a set of retrieval bias keywords. When you run the Debug Agent, it tells the LLM "you are a debugging specialist, here is the relevant code, produce output in this format: summary, findings, plan, patch diff." The "intelligence" comes entirely from the base LLM — the agent just frames the task.

### 3.12 Ollama

Ollama is the local model server that PromptForge uses to run LLMs on your machine.

Think of it as Docker but for AI models:
- You run `ollama pull deepseek-coder-v2:16b` and it downloads the model (like `docker pull`)
- Ollama starts an HTTP server on `localhost:11434`
- Your application sends requests to that HTTP server (like calling a local API)
- Ollama loads the model into RAM/VRAM and runs inference

Under the hood, Ollama uses `llama.cpp` — a highly optimised C++ inference engine that can run models on CPU, GPU, or both. Models are stored in GGUF format, a compressed binary that includes the model weights, tokenizer, and metadata in one file.

**The two Ollama models PromptForge uses:**

`nomic-embed-text` (274 MB) — a tiny, fast embedding model. Every time text needs to be converted to a vector, this model is called. It is called many times per query (once per sub-query during retrieval). Because it is small, it loads instantly and runs fast.

`deepseek-coder-v2:16b` (8.9 GB) — a large code generation model. Called once per query to generate the final answer. Because it is large (16 billion parameters), it needs significant RAM to run and takes several seconds to produce each response.

**The memory problem:** On a machine with 16 GB RAM, loading deepseek-coder-v2:16b leaves very little room for anything else. If both models need to be in memory simultaneously (embedding the query while the generation model is warming up), Ollama may evict one to load the other. This model-swapping on every query is one of the primary causes of PromptForge's performance problems.

---

## 4. System Architecture — Full Walkthrough

### 4.1 The Two Phases

PromptForge has two completely separate phases that run at different times:

**Indexing phase** — runs once (or whenever your code changes). You trigger it from the Repository page. It reads your codebase and builds the searchable index in ChromaDB. Nothing in this phase involves the user asking a question — it is entirely about building the knowledge base.

**Query phase** — runs every time you ask a question. It uses the index built in the first phase to find relevant code and generate an answer.

### 4.2 Phase 1 — Indexing (Offline)

**File:** `backend/app/indexing.py`

```
Step 1 — Walk the filesystem
    Find every file under the target directory.
    Skip folders: node_modules, .git, dist, __pycache__, .venv, build
    Skip file types that are not source code (images, binaries, etc.)

Step 2 — Read each file
    Load the raw text of each source file.
    Detect the language from the file extension (Python, JS, TS, Go, Java, Rust, etc.)

Step 3 — Chunk each file
    Split the text into overlapping windows of ~800 tokens each.
    Adjacent chunks overlap by 120 tokens.
    This overlap reduces the chance of cutting a sentence completely in half.
    Each chunk gets metadata: file path, start line, end line, language.

Step 4 — Embed each chunk
    Call nomic-embed-text via Ollama for each chunk.
    Each chunk becomes a 768-dimensional vector.
    This is the most time-consuming step during indexing.

Step 5 — Store in ChromaDB
    Store the vector + original text + metadata in the "codebase" collection.
    ChromaDB builds an HNSW index over these vectors for fast similarity search later.

Step 6 — Build the import graph
    Parse import statements from each file.
    Build an adjacency list: file A imports file B → add edge A→B.
    Store in graph_memory.py for later use during retrieval.
```

After indexing, ChromaDB has a searchable map of your entire codebase. Nothing needs to be rerun until your code changes.

### 4.3 Phase 2 — Query (Runtime)

This is what happens every time you type a question in the Workbench page and press Enter.

#### Step 1 — Semantic cache check (`cache.py`)

Before doing any expensive work, PromptForge checks if a similar question has been asked before.

```
1. Embed the incoming query using nomic-embed-text
2. Compare the embedding to all stored query embeddings in prompt_history ChromaDB collection
3. If cosine similarity >= 0.92 → return the cached answer immediately
4. If no match → continue to Step 2
```

A cache hit saves the entire pipeline — no Ollama generation call needed, answer returns in milliseconds.

#### Step 2 — Query optimization (`optimizer.py`)

PromptForge tries to improve your raw question before using it to search the codebase.

```
Input:  "Add rate limiting to the API"

Step 2a — Rewrite:
    Send the question to deepseek-coder-v2:16b with a prompt like:
    "Rewrite this as a detailed, self-contained search query for a codebase"
    Output: "Implement request rate limiting middleware in the FastAPI routes
             to prevent API abuse, limiting requests per IP per minute"

Step 2b — Expand into 3 sub-queries:
    Also ask the model to generate 3 alternative angles on the same question:
    Sub-query 1: "FastAPI middleware rate limiting per IP"
    Sub-query 2: "throttle request frequency token bucket algorithm"
    Sub-query 3: "API abuse prevention request counting"
```

**Why this is dangerous (see Section 9):** The model doing this rewriting does not know your codebase. It may hallucinate file paths, function names, or module names that do not exist in your code, then those hallucinated terms get embedded and searched for — finding nothing or finding the wrong things.

#### Step 3 — Retrieval (`retrieval.py`)

For each sub-query (plus the rewritten main query), PromptForge searches ChromaDB.

```
For each query (up to 4 total after expansion):
    1. Embed the query with nomic-embed-text → get a 768-dim vector
    2. Search the ChromaDB codebase collection for the top-K most similar chunks
       Default K = 4, so up to 16 candidates across 4 queries

Merge results:
    Deduplicate by chunk ID (same chunk found by multiple queries counts once)
    Sort by relevance score

Graph memory expansion:
    For each top chunk, look up which files it came from
    Find neighboring files in the import graph (1 hop away)
    Fetch top chunks from those neighbor files
    Apply a 0.85x score penalty to indirect matches

Re-rank:
    Sort all candidates by final score
    Return the top-K for prompt assembly
```

#### Step 4 — Prompt assembly (`prompt_builder.py`)

PromptForge builds the final prompt that will be sent to the generation model.

```
Token budget: 6000 tokens (configurable via CF_TOKEN_BUDGET)

Structure:
┌─────────────────────────────────────────────────────┐
│ HEADER                                               │
│ "You are a code assistant. Answer based on the      │
│  following code from the repository..."              │
├─────────────────────────────────────────────────────┤
│ CHUNK 1 (highest relevance score)                   │
│ File: src/routes/auth.py, lines 45–92               │
│ [actual code text]                                   │
├─────────────────────────────────────────────────────┤
│ CHUNK 2 (second highest score)                      │
│ File: src/middleware/throttle.py, lines 1–34        │
│ [actual code text]                                   │
├─────────────────────────────────────────────────────┤
│ ... more chunks until token budget is exhausted ... │
├─────────────────────────────────────────────────────┤
│ FOOTER                                               │
│ "User question: Add rate limiting to the API"       │
└─────────────────────────────────────────────────────┘

Chunks are added highest-score-first.
When adding the next chunk would exceed 6000 tokens, stop.
```

#### Step 5 — Generation (`generation.py`)

The assembled prompt is sent to the generation model.

```
Blocking mode (/query):
    POST to http://localhost:11434/api/generate
    Wait for the full response
    Return as JSON

Streaming mode (/query/stream):
    POST to http://localhost:11434/api/generate with stream=true
    Ollama sends back tokens one by one as Server-Sent Events
    FastAPI forwards each token to the browser as SSE
    Browser appends each token to the UI in real time
```

#### Step 6 — Cache write and feedback

```
Save to cache:
    Store the query embedding + answer in prompt_history collection
    Future similar queries will hit the cache

Feedback:
    User can thumbs-up or thumbs-down the answer
    Votes are stored as metadata on the chunk in ChromaDB
    Intended for future re-ranking (not yet fully implemented)
```

### 4.4 Agent Mode

Agent mode runs the same pipeline as above but with two extra steps: an intake analysis step at the beginning and a structured output format enforced at the end.

```
User submits: task description + optional error log/stack trace

Step 1 — Intake & agent selection:
    _infer_agent_id() looks for keywords in the request:
    "error", "bug", "crash", "stack trace" → Debug Agent
    "refactor", "clean", "legacy", "modular" → Refactor Pro
    "document", "docstring", "README" → Doc Bot
    "security", "vulnerability", "auth", "injection" → Security Auditor

Step 2 — Retrieval with bias:
    Each agent has retrieval bias keywords
    Debug Agent biases toward: error handlers, exception classes, test files
    These bias keywords are added to the sub-queries to steer retrieval

Step 3 — Plan generation:
    Each agent has a 3-step plan it follows:
    Debug Agent: [Understand the bug] → [Identify root cause] → [Generate fix]

Step 4 — Structured generation:
    The system prompt enforces a specific output format:
    {
        "summary": "What the issue is",
        "relevant_files": ["file1.py", "file2.py"],
        "findings": ["Finding 1", "Finding 2"],
        "plan": ["Step 1", "Step 2", "Step 3"],
        "answer": "Detailed explanation",
        "patch": "--- a/file.py\n+++ b/file.py\n..."
    }

Step 5 — SSE streaming (agents/stream):
    Uses a background thread + asyncio.Queue
    Worker thread runs Steps 2–4 and puts progress events on the queue
    Main thread reads from queue and sends SSE events to browser
```

---

## 5. Every File Explained

### Backend (`backend/app/`)

**`main.py` — The web server**
The entry point of the backend. Defines all HTTP endpoints using FastAPI. When a request arrives at `/query`, this file receives it, calls the appropriate pipeline functions, and returns the result. Also handles CORS (allowing the frontend to call the backend from a different port).

**`config.py` — Environment configuration**
Reads all `CF_*` environment variables and exposes them as a typed Python object. Every other file imports this instead of reading environment variables directly. This means to change the model or token budget, you change the env var, not the source code.

**`schemas.py` — Data shapes**
Defines the exact shape of every request and response using Pydantic models. For example, a `QueryRequest` has a `query: str` field and optionally `top_k: int`. FastAPI uses these to validate incoming requests and produce typed OpenAPI documentation automatically. Uses `camelCase` field aliases to match TypeScript interfaces in the frontend.

**`ollama_client.py` — LLM communication**
All calls to Ollama go through this file. It provides three functions:
- `embed(text)` — calls nomic-embed-text, returns a 768-dim vector
- `chat(prompt, model)` — calls the generation model, returns the full response
- `chat_stream(prompt, model)` — calls the generation model, yields tokens one by one

**Critical flaw in this file:** No timeouts are set on any HTTP call. If Ollama is slow or unresponsive, these calls will wait forever.

**`optimizer.py` — Query rewriting**
Takes the user's raw question and asks the generation LLM to rewrite it into a better search query, then expand it into 3 sub-queries. Returns a list of 4 queries (1 rewritten + 3 sub-queries).

**`retrieval.py` — Vector search**
Takes the list of queries from the optimizer. For each query, embeds it and searches ChromaDB. Merges results across all queries using deduplication by chunk ID. Calls graph_memory.py to expand results with neighboring files. Returns a ranked list of code chunks.

**`graph_memory.py` — Import graph**
Builds and queries an adjacency list of file-level import relationships. During indexing, parses import statements from each file to build the graph. During retrieval, given a set of retrieved chunks, returns chunks from neighboring files (files that import or are imported by the retrieved files).

**`indexing.py` — Offline indexing**
Triggered by the `/index` endpoint. Walks the filesystem, chunks each file, embeds each chunk, and stores it in ChromaDB. Also builds the import graph via graph_memory.py. This is the most resource-intensive operation — it calls the embedding model once per chunk, which can be thousands of times for a large codebase.

**`prompt_builder.py` — Prompt assembly**
Given a list of ranked chunks and a token budget, builds the final prompt string. Adds chunks from highest to lowest score, stopping when the next chunk would exceed the budget. Counts tokens using a simple character-based approximation (not a real tokenizer, which is a minor inaccuracy).

**`generation.py` — LLM generation**
Takes the assembled prompt and calls `ollama_client.py`. Supports both blocking (returns full string) and streaming (yields tokens). For streaming, uses Python `asyncio` async generators to yield tokens as they arrive from Ollama.

**`cache.py` — Semantic cache**
Maintains the `prompt_history` ChromaDB collection. On each query, embeds the query and checks for similar stored queries. If found above the 0.92 threshold, returns the cached answer. Otherwise, after generation, stores the query + answer for future lookups.

**`agent_runner.py` — Agent orchestration**
Implements the agent workflow on top of the standard RAG pipeline. For streaming agent runs, spawns a background thread that runs the full pipeline and puts progress events onto an `asyncio.Queue`. The FastAPI SSE endpoint reads from this queue and sends events to the browser. Contains the threading/async integration code that is most prone to deadlocks.

**`agent_registry.py` — Agent definitions**
A static registry of the four agents. Each entry defines: system prompt, output format specification, retrieval bias keywords, top_k override, whether to use query expansion, and the 3-step plan. No ML involved — these are hand-written prompt templates.

**`store.py` — ChromaDB helpers**
Wrapper around the ChromaDB Python client. Provides functions to get or create collections, insert embeddings, query for similar vectors, and update metadata (used for feedback votes). Also contains the token counting utility used by prompt_builder.py.

### Frontend (`src/`)

**`pages/Workbench.tsx` — Main query interface**
The primary UI page where users type questions and see answers. Handles both blocking requests (shows a loading spinner then the full answer) and SSE streaming (shows tokens appearing in real time). Contains the input box, model selector, and answer display area.

**`pages/Repository.tsx` — Index management**
Lets users browse the local filesystem tree (via the `/repo/tree` endpoint) and trigger indexing of a selected folder (via the `/index` endpoint). Shows indexing progress.

**`pages/History.tsx` — Query history**
Displays past queries and their answers fetched from the `/history` endpoint. Allows users to save or revisit previous results.

**`lib/api.ts` — Backend client**
All calls to the FastAPI backend go through this file. Handles both regular fetch calls (for blocking endpoints) and EventSource (for SSE streaming endpoints). Manages error handling and response parsing.

**`lib/mockData.ts` — Development fixtures**
Static mock data used during frontend development when the backend is not running. TypeScript interfaces defined here must match the camelCase aliases in `schemas.py`.

---

## 6. Technology Stack — Why Each Tool Was Chosen

### FastAPI (Python web framework)

FastAPI is chosen because it natively supports async Python, which is critical for SSE streaming — the server needs to yield tokens one at a time without blocking other requests. It also auto-generates OpenAPI documentation from the Pydantic schemas, making the API self-documenting.

**Alternative considered:** Flask is simpler but does not support async natively, making SSE streaming awkward. Django is too heavyweight for this use case.

### ChromaDB (vector database)

ChromaDB is chosen because it runs entirely in-process with no separate server needed. You install it as a Python package and it manages a local database on disk. Perfect for a single-developer local tool.

**Alternative for production:** Qdrant or pgvector (PostgreSQL extension) handle concurrent access better and scale further. ChromaDB's SQLite backend has write-lock limitations under concurrent access.

### Ollama (local LLM server)

Ollama is chosen because it is the simplest way to run LLMs locally. One command to download a model, one HTTP call to run it. It handles GPU/CPU allocation automatically, supports dozens of models, and exposes an OpenAI-compatible API so the client code is familiar.

**Alternative:** llama.cpp directly gives more control but requires more setup. vLLM is faster for high-throughput but needs more GPU VRAM.

### nomic-embed-text (embedding model)

Chosen because it is small (274 MB), fast, and available in Ollama with one pull command. Produces 768-dimensional vectors which are a good balance between quality and storage/search speed.

**Better alternative:** BGE-M3, which produces dense embeddings, sparse (BM25-style) embeddings, and ColBERT multi-vector embeddings simultaneously from a single model. This would give PromptForge hybrid retrieval for free at the cost of a slightly larger model (567 MB).

### deepseek-coder-v2:16b (generation model)

Chosen because it is one of the best open-source code generation models available locally. Trained specifically on code, it understands syntax, APIs, and common programming patterns across 30+ languages.

**The problem:** 8.9 GB is very large for a local model. On machines with less than 16 GB of RAM, it cannot run alongside nomic-embed-text without model swapping. A 7B code model (like `qwen2.5-coder:7b` at ~4.7 GB) would be faster and more stable on most hardware.

### React + TypeScript + Vite (frontend)

React is chosen for component-based UI development. TypeScript adds type safety, catching bugs at compile time. Vite is a fast build tool with hot-module replacement (changes appear instantly in the browser without a full reload). Tailwind CSS provides utility-class styling without writing custom CSS files.

---

## 7. Configuration Reference

All configuration is done via environment variables. Set these in a `.env` file in the `backend/` directory or export them before running `run.sh`.

| Variable | Default | What it does | When to change |
|----------|---------|--------------|----------------|
| `CF_PROVIDER` | `ollama` | LLM provider. Set to `groq` to use cloud API instead of local Ollama | When local hardware is too slow |
| `CF_GEN_MODEL` | `deepseek-coder-v2:16b` | Which model to use for generation | Change to `qwen2.5-coder:7b` if RAM is limited |
| `CF_EMBED_MODEL` | `nomic-embed-text` | Which model to use for embeddings | Change to `bge-m3` for hybrid retrieval |
| `CF_TOP_K` | `4` | How many chunks to retrieve per query | Increase for broader context, decrease for speed |
| `CF_TOKEN_BUDGET` | `6000` | Maximum tokens in the assembled prompt | Increase if answers are truncated, decrease for speed |
| `CF_USE_EXPANSION` | `true` | Whether to run the 3-sub-query expansion | Set to `false` to disable the optimizer (faster, may improve quality) |
| `CF_CHUNK_SIZE` | `800` | Tokens per chunk during indexing | Decrease for finer-grained retrieval |
| `CF_CHUNK_OVERLAP` | `120` | Overlap tokens between adjacent chunks | Increase if important content is being split |
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama is running | Change if Ollama is on a different machine or port |
| `GROQ_API_KEY` | (none) | API key for Groq cloud generation | Required only if `CF_PROVIDER=groq` |

---

## 8. API Endpoints Reference

All endpoints are served by the FastAPI backend on `http://localhost:8000`.

### `GET /health`
Check if the backend is running and how many chunks are indexed.

**Response:**
```json
{
  "status": "ok",
  "chunk_count": 4521,
  "model": "deepseek-coder-v2:16b",
  "embed_model": "nomic-embed-text"
}
```

If `chunk_count` is 0, the codebase has not been indexed yet. Run indexing from the Repository page.

### `GET /config`
Returns the current runtime configuration. The frontend calls this on startup to adapt its UI to the backend settings.

### `GET /repo/tree`
Returns the local filesystem directory tree for the Repository page browser.

### `POST /index`
Triggers indexing of a local directory.

**Request:**
```json
{ "path": "C:/Users/nsani/Desktop/my-project" }
```

**Response:** Streams progress events as indexing proceeds.

### `POST /query`
Run the full RAG pipeline in blocking mode. Waits for the complete answer.

**Request:**
```json
{
  "query": "Where is authentication implemented?",
  "topK": 4,
  "useExpansion": true
}
```

**Response:**
```json
{
  "answer": "Authentication is implemented in src/auth/...",
  "sources": [
    { "file": "src/auth/service.py", "startLine": 45, "score": 0.94 }
  ],
  "tokensUsed": 1823,
  "cached": false
}
```

### `POST /query/stream`
Same as `/query` but streams the answer token by token using SSE. Use this for the real-time typing effect in the UI.

### `POST /feedback`
Record user feedback on a result.

**Request:**
```json
{
  "chunkId": "abc123",
  "vote": "up"
}
```

### `GET /history`
Returns all past queries and their answers from the semantic cache.

### `GET /agents`
Returns the list of available agents with their descriptions and capabilities.

### `POST /agents/run`
Run an agent in blocking mode.

**Request:**
```json
{
  "task": "Debug this error: TypeError: 'NoneType' object is not subscriptable",
  "logs": "Traceback (most recent call last):\n  File...",
  "agentId": "debug"
}
```

### `POST /agents/stream`
Same as `/agents/run` but streams progress events via SSE as the agent works through its plan.

---

## 9. Known Flaws — What is Broken and Why

These are the seven architectural flaws identified in the current codebase, ordered from "most urgently breaks things" to "degrades quality over time."

---

### Flaw 1 — No timeouts on Ollama calls (CAUSES THE HANG AND CRASH)

**File:** `ollama_client.py`

**What is happening:** Every call to Ollama (both for embedding and generation) is a raw HTTP request with no timeout set. If Ollama takes 5 minutes to respond (because it is swapping models, running on CPU, or has crashed internally), Python's `requests` or `httpx` library waits forever.

**How this causes your three symptoms:**
- **Hang:** The web server's worker thread is stuck waiting for Ollama. The frontend's browser connection times out after 30–60 seconds, the user sees a spinner forever, then a network error.
- **Crash:** On the second concurrent request, FastAPI creates a second worker thread, also waiting for Ollama. When system memory runs out (from holding two Ollama requests open), the process crashes.
- **Empty:** The frontend's SSE connection disconnects after its own timeout. The backend eventually gets a response from Ollama and tries to send it, but the connection is gone. Returns empty.

**The specific SSE deadlock:** In `agent_runner.py`, the streaming agent path uses a `threading.Thread` and an `asyncio.Queue`. If the thread raises an unhandled exception and exits without putting `None` (the sentinel/termination signal) on the queue, the main coroutine blocks on `queue.get()` forever. There is no finally block ensuring the sentinel is always put.

**Fix:**
```python
# In agent_runner.py
def _worker(queue, ...):
    try:
        result = run_full_pipeline(...)
        queue.put({"type": "result", "data": result})
    except Exception as e:
        queue.put({"type": "error", "message": str(e)})
    finally:
        queue.put(None)  # ALWAYS put sentinel, even on crash

# In ollama_client.py
async def generate(prompt, model, timeout=60):
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json()
```

---

### Flaw 2 — The query optimizer poisons retrieval with hallucinated terms

**File:** `optimizer.py`

**What is happening:** The optimizer asks the 16B generation model to rewrite the user's query and generate 3 sub-queries before any retrieval happens. The generation model does not have access to your codebase — it only knows what it learned during training about code in general.

**The hallucination problem:** When you ask about something specific to your project (a custom class name, an internal module structure, a proprietary pattern), the optimizer invents plausible-sounding but wrong details. It might rewrite "add caching to the data layer" as "add Redis caching to src/data/repository.py using the cache_manager.py utility" — but your project uses memcached, not Redis, and has no `cache_manager.py`. The sub-queries are now searching for files that do not exist. Retrieval finds nothing relevant.

**The cost problem:** This step calls the 16B model before retrieval. On a machine running the model on CPU, this alone takes 30–60 seconds — before a single search has been performed. If retrieval and generation each take another 30–60 seconds, total response time exceeds 2 minutes. Frontends typically timeout at 30–60 seconds.

**Research confirms this:** A 2025 ACM paper found that "LLM-based query expansion can significantly degrade retrieval effectiveness when the LLM's knowledge is insufficient or query ambiguity is high." This is exactly the scenario here.

**Fix — Replace with HyDE (Hypothetical Document Embeddings):** Instead of asking the model to rewrite the query (which requires knowing your codebase), ask it to write a fake code snippet that would answer the question. This stays grounded in what the model knows about code in general, without inventing your specific file structure.

```python
async def hyde_expand(query: str) -> str:
    prompt = f"""Write a short Python code snippet that would answer this question.
Only write code, no explanation, no comments.
Question: {query}
Code snippet:"""
    result = await ollama_generate(prompt, model="mistral:7b")  # small model
    return result["response"]

# Then embed the hypothetical snippet, not the rewritten query
hyde_code = await hyde_expand(user_query)
query_vector = await embed(hyde_code)
chunks = chroma.query(query_vector)
```

---

### Flaw 3 — Fixed-size chunking splits functions in half

**File:** `indexing.py`

**What is happening:** Every file is split into 800-token windows regardless of code structure. A 150-line function gets split into 2 chunks at line 100 — one chunk has the function signature and first half of the body, another has the second half and the return statement. Neither chunk is a complete, syntactically valid unit.

**Why this breaks retrieval:** When the retriever finds "chunk 2" (the second half of the function), it has no function signature, no parameter names, no understanding of what the function does from its name — those are all in chunk 1. The LLM receives this fragment and cannot produce a correct answer.

**The fix — tree-sitter AST chunking:** Parse each file into its Abstract Syntax Tree (syntax tree), then chunk at function and class boundaries. Every chunk is a complete function or class.

```python
from tree_sitter import Language, Parser

def ast_chunk_file(source_code: str, language: str) -> list[str]:
    parser = get_parser(language)  # cached per language
    tree = parser.parse(bytes(source_code, "utf-8"))
    chunks = []
    for node in tree.root_node.children:
        if node.type in (
            "function_definition",    # Python
            "class_definition",       # Python
            "function_declaration",   # JS/TS
            "method_definition",      # JS/TS
            "function_item",          # Rust
            "method_declaration",     # Java/Go
        ):
            chunks.append(source_code[node.start_byte:node.end_byte])
    return chunks if chunks else [source_code]
```

Research shows this improves Recall@5 by 4.3 points on codebase benchmarks (cAST paper, 2025).

---

### Flaw 4 — Pure dense retrieval misses exact identifier matches

**File:** `retrieval.py`

**What is happening:** All retrieval is done via semantic embedding similarity (dense retrieval). There is no keyword/exact-match retrieval component.

**Why this is a problem for code:** Code is full of unique identifiers — class names, function names, variable names, API endpoints. When you ask "fix the `RateLimiterMiddleware` class", you want to find the exact file and class named `RateLimiterMiddleware`. Dense retrieval finds *semantically similar* code — maybe a `ThrottleHandler` class, which does something similar but is not the same thing.

**The fix — hybrid retrieval (BM25 + dense) with RRF fusion:**

BM25 is a keyword search algorithm. It finds chunks that contain the exact words/identifiers in your query. Dense retrieval finds chunks that mean the same thing. Combining both (hybrid retrieval) gets you the best of both worlds.

The simplest implementation: swap `nomic-embed-text` for `bge-m3`. BGE-M3 produces dense embeddings, sparse (BM25-style) embeddings, and reranking scores from a single model. One model pull, three capabilities.

```bash
ollama pull bge-m3
```

Then use Reciprocal Rank Fusion (RRF) to combine dense and sparse scores:
```python
def rrf_merge(dense_results, sparse_results, k=60):
    scores = {}
    for rank, chunk in enumerate(dense_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)
    for rank, chunk in enumerate(sparse_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda id: scores[id], reverse=True)
```

---

### Flaw 5 — ChromaDB is doing three jobs it was not designed for simultaneously

**File:** `store.py`

**What is happening:** A single ChromaDB instance holds three collections:
1. `codebase` — the code chunk vector index (its real job)
2. `prompt_history` — the semantic cache + query history
3. Feedback votes stored as metadata on chunks in `codebase`

**Why this is a problem:**
- ChromaDB's underlying storage engine (SQLite) has a single-writer lock. When indexing writes new chunks at the same time a query is reading, and feedback writes vote metadata at the same time — all three are competing for the same write lock. Under concurrent access, this produces database locked errors, which appear as crashes.
- The entire ChromaDB HNSW index must fit in RAM. With all three collections loaded, memory pressure increases significantly. On your machine with 8.9 GB already consumed by the generation model, this tips you into OOM territory.
- Using the prompt_history collection for semantic cache lookup means every cache check is a vector search through all past queries — which gets slower as history grows.

**The fix — separate concerns:**

| Concern | Correct tool |
|---------|-------------|
| Code chunk vectors | ChromaDB (its actual job) |
| Semantic cache + history | SQLite directly — it is just key-value storage |
| Feedback votes | Same SQLite database, a separate `votes` table |

---

### Flaw 6 — Cache threshold of 0.92 is calibrated for chatbots, not code queries

**File:** `cache.py`

**What is happening:** The semantic cache threshold of 0.92 cosine similarity was chosen for general-purpose FAQ chatbots where "how do I reset my password" and "I forgot my password" should return the same answer. Code queries are far more specific.

**The problem:** "Add rate limiting to the login route" and "Add rate limiting to the API gateway" may score above 0.92 similarity — they use almost the same words and mean similar things in general. But they require completely different answers pointing to different files. Returning the cached answer for one when the other was asked is a wrong answer.

Production guidance for code generation: use 0.95–0.97 as the threshold for code-specific queries.

Additionally, if the embedding model is ever changed (from `nomic-embed-text` to `bge-m3` for example), all stored cache embeddings become incompatible with the new model's vector space. The cache would return completely wrong results silently until manually cleared.

**Fix:**
```python
CACHE_THRESHOLD = 0.97  # for code queries

# Tag cache entries with embedding model version
cache_key_prefix = f"nomic-embed-text-v1"

# Add TTL — do not serve cached answers older than 24 hours
MAX_CACHE_AGE_HOURS = 24
```

---

### Flaw 7 — The graph memory is an import graph pretending to be a call graph

**File:** `graph_memory.py`

**What is happening:** During indexing, PromptForge parses import statements from each file to build an "import graph" — an adjacency list where file A points to file B if A imports B. During retrieval, it expands the top-K chunks by also fetching chunks from neighboring files (files 1 hop away in the import graph).

**Why this introduces noise instead of signal:** An import graph is not the same as a call graph. File A importing File B means A has *access* to B's exports — it does not mean A *calls* B's functions. A utility module like `utils.py` may be imported by 50 files. Retrieving a chunk from `auth.py` and then expanding to all files that import `utils.py` would pull in 49 irrelevant files, filling the token budget with noise and pushing out relevant chunks.

What PromptForge actually needs is a **call graph** — a graph where edges represent "function X in file A calls function Y in file B." This is much harder to build (requires deeper AST analysis) but produces actual signal. The `code-graph-rag` project does this correctly using tree-sitter to extract typed relationships: calls, inherits, implements.

**Interim fix — measure before using:** The graph memory is not guaranteed to help. The correct action is to measure Recall@5 with and without graph expansion enabled (`CF_USE_GRAPH=false`). If it does not improve recall, disable it.

---

## 10. The Fixed Architecture — What It Should Look Like

This is what PromptForge's query pipeline should look like after all seven flaws are addressed:

```
User query
    │
    ▼
[Semantic cache] — SQLite, threshold 0.97, model-versioned keys, 24h TTL
    │ MISS
    ▼
[HyDE expansion] — mistral:7b generates a hypothetical code snippet
    │                uses small model, stays grounded, no file hallucination
    ▼
[AST-chunked index] — tree-sitter splits at function/class boundaries
    │                  re-indexed with bge-m3 embeddings
    ├── Dense search (bge-m3 dense vectors) ──────┐
    ├── Sparse search (bge-m3 sparse vectors) ────┤── RRF merge → top 30 candidates
    └── (no import graph noise) ─────────────────┘
    │
    ▼
[Cross-encoder reranker] — bge-reranker-v2 scores all 30, picks top 6–8
    │
    ▼
[Prompt builder] — assembles context within 6000-token budget
    │
    ▼
[Generation] — deepseek-coder-v2:16b via httpx with 60s timeout
    │           try/finally ensures sentinel always sent on SSE queue
    ▼
[Cache write] — store in SQLite with model version tag + timestamp
    │
    ▼
Answer returned to user
```

**Priority order for implementing fixes:**

1. **Today (stability):** Add 60s timeouts to `ollama_client.py` and add `finally: queue.put(None)` to every worker thread in `agent_runner.py`. This stops the crash/hang/empty.

2. **This week (quality, biggest ROI):** Replace fixed chunking with tree-sitter AST chunking in `indexing.py`. Re-index your test codebase. Measure whether retrieval results improve before doing anything else.

3. **This week (quality):** Swap embedding model to `bge-m3`. Implement RRF fusion in `retrieval.py` to combine dense and sparse results.

4. **Next week (quality):** Add a cross-encoder reranker. Pull `bge-reranker-v2` via Ollama. After step 3 retrieves 30 candidates, rerank them and take the top 8.

5. **Next week (reliability):** Move semantic cache to SQLite. Remove the `prompt_history` ChromaDB collection. Raise threshold to 0.97. Add model version tags.

6. **Later (quality, measure first):** Replace the query optimizer with HyDE. Disable graph memory, measure Recall@5, re-enable only if it helps.

---

## 11. Setup and Running

### Prerequisites

Before running PromptForge, you need:

1. **Python 3.11+** — The backend language
2. **Node.js 18+** — Required for the React frontend
3. **Ollama** — The local model server. Download from [ollama.com](https://ollama.com)
4. **8 GB RAM minimum** — 16 GB recommended for the 16B model

### Step 1 — Install and start Ollama

Download and install Ollama from [ollama.com](https://ollama.com). After installation, pull the required models:

```powershell
# Pull the generation model (~8.9 GB download)
ollama pull deepseek-coder-v2:16b

# Pull the embedding model (~274 MB download)
ollama pull nomic-embed-text

# Verify both are present
ollama list
```

Ollama starts automatically as a background service. Verify it is running:
```powershell
Invoke-RestMethod -Uri "http://localhost:11434" -TimeoutSec 5
# Should return: "Ollama is running"
```

### Step 2 — Start the backend

```powershell
# Navigate to the project directory
cd C:\Users\nsani\Desktop\PromptForge

# Create and activate the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r backend\requirements.txt

# Start the FastAPI backend
cd backend
uvicorn app.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Verify it is working:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

### Step 3 — Start the frontend

Open a second PowerShell window:

```powershell
cd C:\Users\nsani\Desktop\PromptForge

# Install Node dependencies
npm install

# Start the Vite dev server
npm run dev
```

Open your browser to `http://localhost:5173`.

### Step 4 — Index a codebase

1. Click the **Repository** tab in the sidebar
2. Browse to a code folder you want to index
3. Click **Index this folder**
4. Wait for indexing to complete (a large project may take several minutes — the embedding model is called once per chunk)
5. Check `/health` to confirm chunk_count > 0

### Step 5 — Ask questions

Click the **Workbench** tab, type a question about your indexed codebase, and press Enter.

### Troubleshooting

**"OLLAMA NOT RESPONDING"** — Ollama is not running. Open the Ollama app from your taskbar or run `ollama serve` in PowerShell.

**Response takes 2+ minutes or times out** — The 16B model is running on CPU, not GPU. Run `ollama ps` and check the PROCESSOR column. If it says CPU, you need more RAM or a smaller model. Try `ollama pull mistral:7b` and set `CF_GEN_MODEL=mistral:7b`.

**Empty answers** — Check that `chunk_count > 0` in `/health`. If it is 0, indexing failed silently. Check the backend terminal for error messages. Common cause: the path you indexed does not exist or has permission errors.

**"Port 8000 already in use"** — Another process is using that port. Either kill it or run the backend on a different port: `uvicorn app.main:app --port 8001`. Then update the frontend's API base URL.

---

## 12. Glossary

**Agent** — An LLM configured with a specific persona, task framing, and output format. In PromptForge, four agents (Debug, Refactor, DocBot, Security) are prompt templates layered on top of the standard RAG pipeline.

**AST (Abstract Syntax Tree)** — A tree representation of source code's grammatical structure. In an AST, a function definition is a single node with children for its name, parameters, and body. Used in AST-aware chunking to split code at meaningful boundaries.

**BM25** — A keyword-based ranking algorithm used in search engines. Given a query, BM25 scores documents by how often query terms appear in them, adjusted for document length. Complements dense embedding search in hybrid retrieval.

**Chunk** — A fragment of a source file, typically 200–800 tokens, used as the basic unit of indexing and retrieval. The quality of chunking directly determines the quality of retrieval.

**ChromaDB** — An open-source vector database. Stores embedding vectors alongside the original text and metadata. Supports fast approximate nearest-neighbor search to find the most similar vectors to a query.

**Completion** — The text output generated by an LLM in response to a prompt.

**Context window** — The maximum number of tokens an LLM can process in a single call. Everything the model "sees" — the prompt, the retrieved context, and the generated answer — must fit within this limit.

**Cosine similarity** — A measure of how similar two vectors are, regardless of their magnitude. Returns a value from -1 (opposite) to 1 (identical). Used to compare query embeddings with stored chunk embeddings during retrieval.

**Dense retrieval** — Retrieval based on embedding vector similarity. Finds documents by semantic meaning, not exact keywords.

**Embedding** — A numerical vector representation of text, produced by an embedding model. Similar text produces similar embeddings (close together in vector space). The fundamental technology behind semantic search.

**FastAPI** — A Python web framework for building APIs. Used as PromptForge's backend web server. Supports async programming and Server-Sent Events natively.

**GGUF** — A file format for storing quantized LLM weights. Used by Ollama to store and load models efficiently. A 16B parameter model in GGUF format fits in ~8.9 GB on disk.

**Graph memory** — In PromptForge, an adjacency list of file-level import relationships. Used to expand retrieval results with code from neighboring files.

**Hallucination** — When an LLM generates text that sounds plausible but is factually incorrect. In the context of PromptForge's query optimizer, the risk of the optimizer generating file paths or function names that do not exist in your codebase.

**Hybrid retrieval** — Combining dense (embedding) retrieval with sparse (keyword/BM25) retrieval, then merging the results using Reciprocal Rank Fusion. Captures both semantic similarity and exact keyword matches.

**HyDE (Hypothetical Document Embeddings)** — A retrieval improvement technique where the LLM generates a hypothetical answer document, which is then embedded and used as the retrieval query. More stable than query rewriting because it does not require knowledge of the specific codebase.

**HNSW (Hierarchical Navigable Small World)** — The approximate nearest-neighbor index algorithm used by ChromaDB. Organises vectors in a graph structure that allows very fast similarity search even over millions of vectors.

**LLM (Large Language Model)** — A neural network trained on massive text corpora to predict and generate text. Examples: GPT-4, Claude, Llama, Deepseek Coder.

**nomic-embed-text** — A small, fast embedding model produced by Nomic AI. Available in Ollama, produces 768-dimensional embeddings. Used in PromptForge for all text-to-vector conversions.

**Ollama** — An open-source local LLM runner. Downloads and runs LLMs on your own hardware, exposing them via a local HTTP API.

**Pydantic** — A Python library for data validation using type annotations. Used by FastAPI to define and validate request/response schemas.

**Quantisation** — A technique to reduce model file size by storing weights at lower numerical precision (e.g. 4-bit integers instead of 32-bit floats). Reduces quality slightly but makes models much smaller and faster to run on consumer hardware.

**RAG (Retrieval-Augmented Generation)** — A technique that augments LLM generation with information retrieved from an external knowledge base, grounding the model's answers in real, up-to-date source material rather than training data alone.

**Reranker (cross-encoder)** — A model that takes a query and a candidate document together as input and produces a relevance score. More accurate than cosine similarity but too slow to run over all documents — typically used after retrieval to rerank a small set of top-K candidates.

**RRF (Reciprocal Rank Fusion)** — A simple, parameter-free algorithm for combining ranked lists from multiple retrieval sources. Each document's score is the sum of `1/(k + rank)` across all lists. Used to merge dense and sparse retrieval results in hybrid search.

**Semantic cache** — A cache that stores query-response pairs and serves cached responses when a new query is semantically similar (above a cosine similarity threshold) to a stored query.

**SSE (Server-Sent Events)** — A standard for servers to push text events to browsers over a persistent HTTP connection. Used by PromptForge to stream LLM-generated tokens to the frontend in real time.

**Token** — The basic unit of text processing for LLMs. Roughly 3–4 characters each for English text. All LLM limits (context window, generation budget, token cost) are measured in tokens.

**Token budget** — The maximum number of tokens allowed in the assembled prompt. PromptForge's prompt builder uses this to decide how many chunks to include.

**tree-sitter** — A fast, incremental parsing library that supports 100+ programming languages. Produces concrete syntax trees from source code. Used in AST-aware chunking to split files at syntactically meaningful boundaries.

**Vector** — A list of numbers. In the context of embeddings, a vector in high-dimensional space (768 dimensions for nomic-embed-text) where position encodes semantic meaning.

**Vector database** — A database optimised for storing and searching high-dimensional vectors using approximate nearest-neighbor algorithms. Examples: ChromaDB, Qdrant, Pinecone, pgvector.

**VRAM** — Video RAM — the memory on a GPU. Running LLMs on GPU requires the model weights to fit in VRAM. `deepseek-coder-v2:16b` needs ~10 GB of VRAM for GPU inference.

---

*PromptForge Documentation — Generated June 2026*
*Based on codebase review, architectural analysis, and research into RAG best practices as of 2025–2026.*
