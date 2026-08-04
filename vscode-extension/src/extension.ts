import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import * as path from 'node:path';
import * as fs from 'node:fs';

// ---------- wire types (mirror backend/app/schemas.py, camelCase on the wire) ----------

interface RetrievedChunk {
  id: string;
  path: string;
  lines: string;
  score: number;
  snippet: string;
}

interface PipelineResult {
  optimizedQuery: string;
  expansions: string[];
  chunks: RetrievedChunk[];
  answer: string;
  model: string;
  latencyMs: number;
  cached: boolean;
  mode: 'answer' | 'forge';
  turnId?: string;
  tokens: {
    /** Token cost of pasting the retrieved files in full — what this replaces. */
    naiveBaseline: number;
    /** Token cost of what was actually delivered. */
    optimized: number;
    saved: number;
    savedPct: number;
  };
}

interface HistoryItem {
  id: string;
  query: string;
  date: string;
  model: string;
  tokensSaved: number;
  cached: boolean;
}

// ---------- config / state ----------

const cfg = () => vscode.workspace.getConfiguration('promptforge');
const backendUrl = () =>
  cfg().get<string>('backendUrl', 'http://localhost:8000').replace(/\/+$/, '');

let backendProc: ChildProcess | undefined;
// ponytail: one live session id; per-session Map if concurrent chats matter
let sessionId = randomUUID();

// Refine needs free text, and a button click can't collect it (no stream, no
// input box round-trip needed) — arm this flag instead, then treat the user's
// very next message in the same session as the refinement note.
let pendingRefine: { sessionId: string; turnId: string } | undefined;

// Sentinel prompts for the two no-input review-gate escape hatches, driven by
// followupProvider (clicking a followup resubmits its `.prompt` as a brand-new
// request through this same handler with a fresh stream).
const EXPLAIN_INSTEAD_PROMPT = 'Just explain it instead of a brief.';
const MAKE_BRIEF_PROMPT = 'Turn this into a forged brief instead.';

/** Recover the previous turn's PfChatResult.metadata from chat history — VS
 * Code writes ChatResult onto each ChatResponseTurn but this codebase never
 * read it back until now. Duck-typed (`'result' in h`) rather than an
 * instanceof check on vscode.ChatResponseTurn, since history mixes request
 * and response turn types. */
function lastMetadata(chatContext: vscode.ChatContext): PfChatResult['metadata'] | undefined {
  for (let i = chatContext.history.length - 1; i >= 0; i--) {
    const h = chatContext.history[i] as any;
    if ('result' in h) return (h.result as PfChatResult).metadata;
  }
  return undefined;
}

const workspaceRoot = () => vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

// ---------- HTTP helpers ----------

async function getJson<T>(urlPath: string, timeoutMs = 10_000): Promise<T> {
  const res = await fetch(`${backendUrl()}${urlPath}`, {
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`Backend ${res.status} on ${urlPath}`);
  return (await res.json()) as T;
}

/**
 * Node's `fetch` reports transport failures as a bare `TypeError: fetch failed`
 * and hides the real reason (ECONNREFUSED, HeadersTimeoutError, …) on `.cause`.
 * Always unwrap it — otherwise every network problem looks identical.
 */
function errMessage(err: any): string {
  const msg = err?.message ?? String(err);
  const cause = err?.cause;
  if (!cause) return msg;
  const detail = cause?.code ?? cause?.message ?? String(cause);
  return detail && !msg.includes(detail) ? `${msg} (${detail})` : msg;
}

async function postJson<T>(urlPath: string, body: unknown): Promise<T> {
  const res = await fetch(`${backendUrl()}${urlPath}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(detail.detail ?? `Backend ${res.status} on ${urlPath}`);
  }
  return (await res.json()) as T;
}

/** SSE over POST (backend frames as "event: name\ndata: json\n\n"); EventSource can't POST. */
async function* sse(
  urlPath: string,
  body: unknown,
  signal: AbortSignal
): AsyncGenerator<{ event: string; data: any }> {
  const res = await fetch(`${backendUrl()}${urlPath}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(detail.detail ?? `Backend ${res.status} on ${urlPath}`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let ev = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop()!;
    for (const line of lines) {
      if (line.startsWith('event: ')) ev = line.slice(7).trim();
      else if (line.startsWith('data: ') && line.slice(6).trim()) {
        const raw = line.slice(6);
        // /query/stream's "stage" event sends an unquoted string, unlike every
        // other event; fall back to the raw text when it isn't valid JSON.
        let data: any;
        try {
          data = JSON.parse(raw);
        } catch {
          data = raw;
        }
        yield { event: ev, data };
      }
    }
  }
}

// ---------- backend lifecycle ----------

interface Health {
  ok: boolean;
  indexedChunks: number;
  provider: string;
  /** Absolute path of the repo currently in the index (null if never indexed). */
  indexedRepo?: string | null;
}

/** Path comparison for index identity: case-insensitive on Windows, slash-agnostic. */
function samePath(a: string | null | undefined, b: string | null | undefined): boolean {
  if (!a || !b) return false;
  const norm = (p: string) => {
    const resolved = path.resolve(p).replace(/[\\/]+$/, '');
    return process.platform === 'win32' ? resolved.toLowerCase() : resolved;
  };
  return norm(a) === norm(b);
}

function resolveBackendDir(): string | undefined {
  const set = cfg().get<string>('backendDir', '');
  if (set) return set;
  for (const f of vscode.workspace.workspaceFolders ?? []) {
    const dir = path.join(f.uri.fsPath, 'backend');
    if (fs.existsSync(path.join(dir, 'app', 'main.py'))) return dir;
  }
  return undefined;
}

async function health(timeoutMs = 1500): Promise<Health | undefined> {
  try {
    return await getJson<Health>('/health', timeoutMs);
  } catch {
    return undefined;
  }
}

// ---------- auto-index-on-activation state ----------
// `ensureBackend`/`indexWorkspace` run once at activation with nobody watching
// (kicked off before any chat turn exists), and the shared promise below lets a
// request that arrives mid-index reuse that same in-flight work instead of
// double-spawning uvicorn. Since neither function ever has a live stream to
// write into, failures are recorded here instead and surfaced by the chat
// handler once `kickOffBackend()`'s promise resolves.
let backendError: string | undefined;

/** Health-check, spawn uvicorn if down, auto-index the workspace if the index is empty. */
async function ensureBackend(): Promise<Health | undefined> {
  backendError = undefined;
  let h = await health();
  if (!h) {
    const dir = resolveBackendDir();
    if (!dir) {
      backendError =
        `PromptForge backend is not running at \`${backendUrl()}\` and I can't find ` +
        '`backend/app/main.py` in this workspace. Start it manually or set ' +
        '`promptforge.backendDir` in settings.';
      return undefined;
    }
    const python =
      cfg().get<string>('pythonPath', '') ||
      path.join(
        dir,
        '.venv',
        process.platform === 'win32' ? 'Scripts\\python.exe' : 'bin/python'
      );
    if (!fs.existsSync(python)) {
      backendError =
        `Can't start the backend: Python not found at \`${python}\`. ` +
        'Set `promptforge.pythonPath` in settings.';
      return undefined;
    }
    const port = new URL(backendUrl()).port || '8000';
    // no --reload: the reloader parent orphans its worker on Windows kill
    backendProc = spawn(python, ['-m', 'uvicorn', 'app.main:app', '--port', port], {
      cwd: dir,
      stdio: 'ignore',
    });
    // cold start (unwarmed file cache) measured at >70s on Windows; warm ~15s
    for (let i = 0; i < 120 && !h; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      h = await health();
    }
    if (!h) {
      backendError =
        'Backend failed to start within 120s. Check that dependencies are installed ' +
        `(\`${python} -m pip install -r requirements.txt\` in \`${dir}\`).`;
      return undefined;
    }
  }
  // The index is global (one repo at a time), so re-index whenever it's empty
  // OR belongs to a different workspace than the one that's open — otherwise
  // opening another project would silently answer from the previous repo's code.
  const root = workspaceRoot();
  if (root && (h.indexedChunks === 0 || !samePath(h.indexedRepo, root))) {
    await indexWorkspace();
    h = (await health()) ?? h;
  }
  return h;
}

async function indexWorkspace(): Promise<void> {
  const root = workspaceRoot();
  if (!root) return; // no workspace open — nothing to index, nothing to report
  await postJson<{ name: string; files: number; chunks: number }>('/index', { path: root });
}

// Shared in-flight promise so activation-time indexing and a request that
// arrives mid-index converge on the same work instead of double-spawning
// uvicorn. `indexPhase` lets the chat handler show the right progress message
// without needing a stream reference inside the activation-time call itself.
type IndexPhase = 'not_started' | 'indexing' | 'ready' | 'failed';
let indexPhase: IndexPhase = 'not_started';
let backendReadyPromise: Promise<Health | undefined> | null = null;

function kickOffBackend(): Promise<Health | undefined> {
  if (backendReadyPromise) return backendReadyPromise;
  indexPhase = 'indexing';
  backendReadyPromise = ensureBackend()
    // indexWorkspace()'s postJson call can throw (e.g. mid-index network error);
    // fold that into the same "failed" bookkeeping instead of leaving the shared
    // promise permanently rejected.
    .catch((err) => {
      backendError = errMessage(err);
      return undefined;
    })
    .then((h) => {
      indexPhase = h ? 'ready' : 'failed';
      // Don't cache a failure forever — clear the shared promise so the next
      // chat turn (or a retry) gets a fresh attempt instead of being stuck.
      if (!h) backendReadyPromise = null;
      return h;
    });
  return backendReadyPromise;
}

// ---------- rendering helpers ----------

const STAGE_LABELS: Record<string, string> = {
  routing: 'Routing…',
  optimizing: 'Optimizing query…',
  retrieving: 'Retrieving code…',
  assembling: 'Assembling prompt…',
  generating: 'Generating…',
};

/** One-line footer describing what actually happened on this turn.
 *
 * Deliberately NOT a savings claim. The previous version reported "~26,400
 * saved vs. pasting these 4 files", but Copilot runs inside VS Code and reads
 * files itself — nobody pastes them, so the avoided cost was imaginary. (Its
 * predecessor was worse: it compared against pasting all 622 chunks of the
 * repo.) Every number below is an event that occurred; real avoided-call
 * totals live behind /savings and are surfaced by `/history`. */
function renderStats(stream: vscode.ChatResponseStream, result: PipelineResult) {
  if (result.cached) {
    stream.markdown('\n\n_Served from cache — no retrieval or generation run._');
    return;
  }
  const files = new Set(result.chunks.map((c) => c.path)).size;
  if (!files) return; // off-topic guardrail — nothing was searched
  const brief = result.tokens?.optimized;
  stream.markdown(
    `\n\n_Narrowed to ${files === 1 ? '1 file' : `${files} files`}` +
      (brief ? ` · ${brief.toLocaleString()}-token ${result.mode === 'forge' ? 'brief' : 'answer'}_` : '_')
  );
}

function addReferences(stream: vscode.ChatResponseStream, chunks: RetrievedChunk[]) {
  const root = workspaceRoot();
  for (const c of chunks) {
    try {
      const p = path.isAbsolute(c.path) ? c.path : path.join(root ?? '', c.path);
      stream.reference(vscode.Uri.file(p));
    } catch {
      // unresolvable path — skip the reference, the answer text still names it
    }
  }
}

// ---------- mode handlers ----------

interface PfChatResult extends vscode.ChatResult {
  metadata?: {
    sessionId: string;
    chunkIds?: string[];
    turnId?: string;
    mode?: 'answer' | 'forge';
  };
}

/** Review-gate buttons — nothing reaches Copilot until Approve is clicked.
 * Approve renders for any forge-mode result (even a cache hit — it needs no
 * chunks, just the text). Refine needs the turn's stored chunks to regenerate
 * without re-retrieving, which cache hits never have (chunks=[]), so it's
 * omitted whenever there's no turnId. */
function attachReviewGate(stream: vscode.ChatResponseStream, result: PipelineResult) {
  if (result.mode !== 'forge') return;
  stream.button({
    command: 'promptforge.approve',
    arguments: [result.answer],
    title: 'Approve → send to Copilot',
  });
  if (result.turnId) {
    stream.button({
      command: 'promptforge.refine',
      arguments: [sessionId, result.turnId],
      title: 'Refine',
    });
  }
}

async function handleQuery(
  request: vscode.ChatRequest,
  stream: vscode.ChatResponseStream,
  signal: AbortSignal,
  mode: 'answer' | 'forge' | 'auto'
): Promise<PfChatResult> {
  // topK/useExpansion are deliberately omitted so the backend's configured
  // values win. Hardcoding them here silently pinned retrieval to 4 chunks and
  // made CF_TOP_K a no-op — /config reported 8 while every query still used 4.
  const body = {
    query: request.prompt,
    sessionId,
    mode,
    model: cfg().get<string>('model', '') || undefined,
  };
  let result: PipelineResult | undefined;
  for await (const { event, data } of sse('/query/stream', body, signal)) {
    if (event === 'stage') stream.progress(STAGE_LABELS[data] ?? String(data));
    else if (event === 'token') stream.markdown(String(data));
    else if (event === 'done') result = data as PipelineResult;
    else if (event === 'error') stream.markdown(`\n\n**Error:** ${data}`);
  }
  if (result) {
    addReferences(stream, result.chunks);
    renderStats(stream, result);
    attachReviewGate(stream, result);
  }
  return {
    metadata: {
      sessionId,
      chunkIds: result?.chunks.map((c) => c.id),
      turnId: result?.turnId,
      mode: result?.mode,
    },
  };
}

/** Refine / "just explain it" / "make a brief" — regenerates a stored turn's
 * content from its already-retrieved chunks via /query/review. Mirrors
 * handleQuery's SSE consumption and button rendering. */
async function handleReviewAction(
  turnId: string,
  action: 'refine' | 'to_answer' | 'to_brief',
  note: string | undefined,
  stream: vscode.ChatResponseStream,
  signal: AbortSignal
): Promise<PfChatResult> {
  const body = {
    turnId,
    action,
    note,
    model: cfg().get<string>('model', '') || undefined,
  };
  let result: PipelineResult | undefined;
  for await (const { event, data } of sse('/query/review', body, signal)) {
    if (event === 'stage') stream.progress(STAGE_LABELS[data] ?? String(data));
    else if (event === 'token') stream.markdown(String(data));
    else if (event === 'done') result = data as PipelineResult;
    else if (event === 'error') stream.markdown(`\n\n**Error:** ${data}`);
  }
  if (result) {
    addReferences(stream, result.chunks);
    renderStats(stream, result);
    attachReviewGate(stream, result);
  }
  return {
    metadata: {
      sessionId,
      chunkIds: result?.chunks.map((c) => c.id),
      turnId: result?.turnId,
      mode: result?.mode,
    },
  };
}

interface SavingsTotals {
  answeredLocally: number;
  refinedLocally: number;
  cacheHits: number;
  contextTokens: number;
  copilotCallsAvoided: number;
  since: string;
}

async function handleHistory(stream: vscode.ChatResponseStream): Promise<void> {
  // Avoided-call totals first — this is the honest version of "tokens saved".
  try {
    const s = await getJson<SavingsTotals>('/savings', 5000);
    if (s.copilotCallsAvoided > 0) {
      stream.markdown(
        `**${s.copilotCallsAvoided} Copilot call${s.copilotCallsAvoided === 1 ? '' : 's'} avoided**` +
          `${s.since ? ` since ${s.since}` : ''} — ` +
          `${s.answeredLocally} answered locally, ${s.refinedLocally} refined before sending, ` +
          `${s.cacheHits} served from cache.\n\n` +
          `Roughly **${s.contextTokens.toLocaleString()} tokens** of code PromptForge read ` +
          `instead of Copilot. Rough estimate — the real saving is probably higher.\n\n`
      );
    }
  } catch {
    // older backend without /savings — the history table below still works
  }

  const items = await getJson<HistoryItem[]>('/history');
  if (!items.length) {
    stream.markdown('No query history yet.');
    return;
  }
  stream.markdown('| Query | Date | Model |\n|---|---|---|\n');
  for (const it of items.slice(0, 20)) {
    const q = it.query.replace(/\|/g, '\\|').replace(/\n/g, ' ').slice(0, 80);
    stream.markdown(`| ${q} | ${it.date} | ${it.model}${it.cached ? ' (cached)' : ''} |\n`);
  }
}

async function handleModelPick(
  request: vscode.ChatRequest,
  stream: vscode.ChatResponseStream
): Promise<void> {
  const ollama = cfg()
    .get<string>('ollamaUrl', 'http://localhost:11434')
    .replace(/\/+$/, '');
  let models: string[];
  try {
    const res = await fetch(`${ollama}/api/tags`, { signal: AbortSignal.timeout(5000) });
    const tags = (await res.json()) as { models: { name: string }[] };
    models = tags.models.map((m) => m.name);
  } catch {
    stream.markdown(`Couldn't reach Ollama at \`${ollama}\` to list models.`);
    return;
  }
  // ask the backend which model it actually generates with; fall back to the
  // documented small-tier default if the backend isn't up yet
  let defaultModel = 'deepseek-coder-v2:16b';
  try {
    const rc = await getJson<{ generationModel: string }>('/config', 2000);
    defaultModel = rc.generationModel;
  } catch {
    // backend not running — keep the fallback
  }

  // "/model <name>" sets it directly; "/model" alone shows the list
  const requested = request.prompt.trim();
  if (requested) {
    const match = models.find((m) => m.toLowerCase() === requested.toLowerCase());
    const value = /^default$/i.test(requested) ? '' : match ?? requested;
    await cfg().update('model', value, vscode.ConfigurationTarget.Global);
    stream.markdown(
      value
        ? `Generation model set to **${value}**.`
        : `Generation model reset to backend default (**${defaultModel}**).`
    );
    return;
  }

  const current = cfg().get<string>('model', '');
  stream.markdown('**Ollama models** — click one, or run `/model <name>`:\n\n');
  stream.button({
    command: 'promptforge.setModel',
    arguments: [''],
    title: current ? `Use backend default (${defaultModel})` : `✓ Backend default (${defaultModel})`,
  });
  for (const name of models) {
    const isDefault = name === defaultModel;
    const isCurrent = name === current;
    const tag = [isCurrent ? '✓' : '', isDefault ? '(default)' : ''].filter(Boolean).join(' ');
    stream.button({
      command: 'promptforge.setModel',
      arguments: [name],
      title: tag ? `${name} ${tag}` : name,
    });
  }
}

// ---------- activation ----------

export function activate(context: vscode.ExtensionContext) {
  // Kick off backend startup + indexing now, not on the user's first message —
  // cold indexing can take minutes, so get ahead of it. Fire-and-forget: nobody's
  // watching yet, and the chat handler awaits this same shared promise later.
  void kickOffBackend().catch(() => {});

  const handler: vscode.ChatRequestHandler = async (request, chatContext, stream, token) => {
    if (chatContext.history.length === 0) sessionId = randomUUID(); // new chat session
    const controller = new AbortController();
    const sub = token.onCancellationRequested(() => controller.abort());
    try {
      // /model only needs Ollama, not the backend
      if (request.command === 'model') {
        await handleModelPick(request, stream);
        return {};
      }
      const wasWaiting = indexPhase === 'not_started' || indexPhase === 'indexing';
      if (indexPhase === 'not_started')
        stream.progress('Starting PromptForge and indexing this workspace — first response may take a bit…');
      else if (indexPhase === 'indexing')
        stream.progress('Still indexing this workspace, hang tight…');
      const h = await kickOffBackend();
      if (!h) {
        stream.markdown(backendError ?? 'PromptForge backend failed to start.');
        return {};
      }
      if (wasWaiting) stream.markdown(`Ready — ${h.indexedChunks} chunks indexed.\n\n`);

      // Refine armed by a button click: this message IS the note (collected in
      // the dialog and prefilled here, or typed directly).
      if (pendingRefine) {
        const { sessionId: armedFor, turnId } = pendingRefine;
        pendingRefine = undefined;
        if (armedFor === sessionId)
          return await handleReviewAction(turnId, 'refine', request.prompt, stream, controller.signal);
        // `sessionId` is regenerated on an empty history, so starting a new chat
        // between the click and the note orphans the armed turn. Say so rather
        // than silently answering the note as a brand-new question.
        stream.markdown(
          "That Refine was for an earlier chat, so it's no longer attached — " +
            'answering this as a new request instead.\n\n'
        );
      }
      // "Just explain it instead" / "Turn this into a brief" followups.
      const priorMeta = lastMetadata(chatContext);
      if (priorMeta?.turnId && request.prompt === EXPLAIN_INSTEAD_PROMPT)
        return await handleReviewAction(priorMeta.turnId, 'to_answer', undefined, stream, controller.signal);
      if (priorMeta?.turnId && request.prompt === MAKE_BRIEF_PROMPT)
        return await handleReviewAction(priorMeta.turnId, 'to_brief', undefined, stream, controller.signal);

      switch (request.command) {
        case 'history':
          await handleHistory(stream);
          return {};
        case 'forge':
          return await handleQuery(request, stream, controller.signal, 'forge');
        case 'query':
          return await handleQuery(request, stream, controller.signal, 'answer');
        default:
          // No explicit command — let the backend router decide question vs. task.
          return await handleQuery(request, stream, controller.signal, 'auto');
      }
    } catch (err: any) {
      if (!token.isCancellationRequested)
        stream.markdown(`\n\n**Error:** ${errMessage(err)}`);
      return {};
    } finally {
      sub.dispose();
    }
  };

  const participant = vscode.chat.createChatParticipant('promptforge.chat', handler);
  participant.onDidReceiveFeedback((e) => {
    const meta = (e.result as PfChatResult).metadata;
    if (!meta?.sessionId) return;
    postJson('/feedback', {
      queryId: meta.sessionId,
      feedback: e.kind === vscode.ChatResultFeedbackKind.Helpful ? 'up' : 'down',
      chunkIds: meta.chunkIds ?? [],
    }).catch(() => {});
  });
  // Escape hatches for the review gate — mirrored on the forge/answer paths.
  // Clicking a followup resubmits its prompt as a brand-new request through
  // `handler` above, which matches it against the two sentinel strings.
  participant.followupProvider = {
    provideFollowups(result: vscode.ChatResult) {
      const meta = (result as PfChatResult).metadata;
      if (!meta?.turnId) return [];
      if (meta.mode === 'forge')
        return [{ prompt: EXPLAIN_INSTEAD_PROMPT, label: 'Just explain it instead' }];
      if (meta.mode === 'answer')
        return [{ prompt: MAKE_BRIEF_PROMPT, label: 'Turn this into a brief' }];
      return [];
    },
  };

  context.subscriptions.push(
    participant,
    vscode.commands.registerCommand('promptforge.approve', async (text: string) => {
      // Clipboard is the guaranteed path — always do it first. `chat.open`'s
      // `isPartialQuery: true` is the only DOCUMENTED usage (prefill without
      // auto-submit; it's not part of the stable public API, and whether
      // omitting isPartialQuery reliably auto-submits is unconfirmed and
      // version-dependent — never rely on that). If chat.open isn't available
      // in this VS Code version, the user still has the brief on their clipboard.
      await vscode.env.clipboard.writeText(text);
      try {
        await vscode.commands.executeCommand('workbench.action.chat.open', {
          query: text,
          isPartialQuery: true,
        });
      } catch {
        vscode.window.setStatusBarMessage('PromptForge: brief copied — paste into Copilot Chat.', 4000);
      }
    }),
    // Collect the refinement note up front in a dialog. A command handler has no
    // ChatResponseStream to render into, so the note is prefilled back into the
    // chat box: submitting it re-enters `handler` as an ordinary turn, where the
    // armed `pendingRefine` reroutes it to /query/review with a live stream.
    vscode.commands.registerCommand('promptforge.refine', async (forSessionId: string, turnId: string) => {
      const note = await vscode.window.showInputBox({
        title: 'Refine this brief',
        prompt: 'What is wrong with this brief? What should change?',
        placeHolder: 'e.g. too vague about error handling; focus on the auth module',
        ignoreFocusOut: true, // survives clicking away mid-typing
        validateInput: (v) => (v.trim() ? undefined : 'Describe what should change'),
      });
      if (!note?.trim()) return; // cancelled — arm nothing, leave no stale state
      pendingRefine = { sessionId: forSessionId, turnId };
      try {
        await vscode.commands.executeCommand('workbench.action.chat.open', {
          query: `@promptforge ${note.trim()}`,
          isPartialQuery: true,
        });
      } catch {
        vscode.window.setStatusBarMessage(
          'PromptForge: send your note as the next @promptforge message.',
          5000
        );
      }
    }),
    vscode.commands.registerCommand('promptforge.setModel', async (name: string) => {
      await cfg().update('model', name, vscode.ConfigurationTarget.Global);
      vscode.window.setStatusBarMessage(
        `PromptForge model set to ${name || '(backend default)'}`,
        3000
      );
    })
  );
}

export function deactivate() {
  // ponytail: kill() only; tree-kill if uvicorn ever spawns workers
  backendProc?.kill();
}
