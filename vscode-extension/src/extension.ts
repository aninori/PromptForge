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
}

interface AgentFinding {
  severity: string;
  title: string;
  detail: string;
  files: string[];
}

interface AgentRunResult {
  agentId: string;
  agentName: string;
  model: string;
  summary: string;
  relevantFiles: RetrievedChunk[];
  findings: AgentFinding[];
  plan: string[];
  answer: string;
  patchDiff: string;
  patchFiles: string[];
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

const workspaceRoot = () => vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

// ---------- HTTP helpers ----------

async function getJson<T>(urlPath: string, timeoutMs = 10_000): Promise<T> {
  const res = await fetch(`${backendUrl()}${urlPath}`, {
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`Backend ${res.status} on ${urlPath}`);
  return (await res.json()) as T;
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

/** Health-check, spawn uvicorn if down, auto-index the workspace if the index is empty. */
async function ensureBackend(
  stream: vscode.ChatResponseStream
): Promise<Health | undefined> {
  let h = await health();
  if (!h) {
    const dir = resolveBackendDir();
    if (!dir) {
      stream.markdown(
        `PromptForge backend is not running at \`${backendUrl()}\` and I can't find ` +
          '`backend/app/main.py` in this workspace. Start it manually or set ' +
          '`promptforge.backendDir` in settings.'
      );
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
      stream.markdown(
        `Can't start the backend: Python not found at \`${python}\`. ` +
          'Set `promptforge.pythonPath` in settings.'
      );
      return undefined;
    }
    stream.progress('Starting PromptForge backend…');
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
      stream.markdown(
        'Backend failed to start within 120s. Check that dependencies are installed ' +
          `(\`${python} -m pip install -r requirements.txt\` in \`${dir}\`).`
      );
      return undefined;
    }
  }
  // ponytail: no repo-identity check on the global index; /index to re-point it
  if (h.indexedChunks === 0 && workspaceRoot()) {
    await indexWorkspace(stream);
    h = (await health()) ?? h;
  }
  return h;
}

async function indexWorkspace(stream: vscode.ChatResponseStream): Promise<void> {
  const root = workspaceRoot();
  if (!root) {
    stream.markdown('No workspace folder open — nothing to index.');
    return;
  }
  stream.progress(`Indexing ${path.basename(root)}…`);
  const res = await postJson<{ name: string; files: number; chunks: number }>('/index', {
    path: root,
  });
  stream.markdown(
    `Indexed **${res.name}**: ${res.files} files → ${res.chunks} chunks.\n\n`
  );
}

// ---------- rendering helpers ----------

const STAGE_LABELS: Record<string, string> = {
  routing: 'Routing…',
  optimizing: 'Optimizing query…',
  retrieving: 'Retrieving code…',
  assembling: 'Assembling prompt…',
  generating: 'Generating…',
};

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
  metadata?: { sessionId: string; chunkIds?: string[] };
}

async function handleQuery(
  request: vscode.ChatRequest,
  stream: vscode.ChatResponseStream,
  signal: AbortSignal,
  mode: 'answer' | 'forge'
): Promise<PfChatResult> {
  const body = {
    query: request.prompt,
    topK: 4,
    useExpansion: true,
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
    if (mode === 'forge')
      stream.button({
        command: 'promptforge.copy',
        arguments: [result.answer],
        title: 'Copy forged prompt',
      });
  }
  return { metadata: { sessionId, chunkIds: result?.chunks.map((c) => c.id) } };
}

async function handleAgent(
  request: vscode.ChatRequest,
  stream: vscode.ChatResponseStream,
  signal: AbortSignal
): Promise<PfChatResult> {
  const root = workspaceRoot() ?? '.';
  const body = {
    agentId: 'auto',
    repoPath: root,
    targetPath: root,
    userRequest: request.prompt,
    logs: '',
    attachments: [],
    topK: 6,
    useExpansion: false,
    model: cfg().get<string>('model', '') || undefined,
  };
  let result: AgentRunResult | undefined;
  for await (const { event, data } of sse('/agents/stream', body, signal)) {
    if (event === 'stage') stream.progress(String(data));
    else if (event === 'log') stream.progress(`${data.stage}: ${data.message}`);
    else if (event === 'done') result = data as AgentRunResult;
    else if (event === 'error') stream.markdown(`\n\n**Error:** ${data}`);
    // artifact events skipped: the done payload is a superset
  }
  if (result) {
    stream.markdown(`**${result.agentName}** (${result.model})\n\n${result.summary}\n\n`);
    if (result.findings.length) {
      stream.markdown('### Findings\n');
      for (const f of result.findings)
        stream.markdown(`- **[${f.severity}]** ${f.title} — ${f.detail}\n`);
      stream.markdown('\n');
    }
    if (result.plan.length) {
      stream.markdown('### Plan\n');
      result.plan.forEach((step, i) => stream.markdown(`${i + 1}. ${step}\n`));
      stream.markdown('\n');
    }
    if (result.answer) stream.markdown(`${result.answer}\n\n`);
    if (result.patchDiff) {
      stream.markdown('### Patch\n```diff\n' + result.patchDiff + '\n```\n');
      stream.button({
        command: 'promptforge.copy',
        arguments: [result.patchDiff],
        title: 'Copy patch',
      });
    }
    addReferences(stream, result.relevantFiles);
  }
  return { metadata: { sessionId, chunkIds: result?.relevantFiles.map((c) => c.id) } };
}

async function handleHistory(stream: vscode.ChatResponseStream): Promise<void> {
  const items = await getJson<HistoryItem[]>('/history');
  if (!items.length) {
    stream.markdown('No query history yet.');
    return;
  }
  stream.markdown('| Query | Date | Model | Tokens saved |\n|---|---|---|---|\n');
  for (const it of items.slice(0, 20)) {
    const q = it.query.replace(/\|/g, '\\|').replace(/\n/g, ' ').slice(0, 80);
    stream.markdown(
      `| ${q} | ${it.date} | ${it.model} | ${it.tokensSaved}${it.cached ? ' (cached)' : ''} |\n`
    );
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
  let defaultModel = 'mistral:7b';
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
      const h = await ensureBackend(stream);
      if (!h) return {};
      switch (request.command) {
        case 'index':
          await indexWorkspace(stream);
          return {};
        case 'history':
          await handleHistory(stream);
          return {};
        case 'agent':
          return await handleAgent(request, stream, controller.signal);
        case 'forge':
          return await handleQuery(request, stream, controller.signal, 'forge');
        case 'query':
        default:
          return await handleQuery(request, stream, controller.signal, 'answer');
      }
    } catch (err: any) {
      if (!token.isCancellationRequested)
        stream.markdown(`\n\n**Error:** ${err?.message ?? String(err)}`);
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

  context.subscriptions.push(
    participant,
    vscode.commands.registerCommand('promptforge.copy', (text: string) => {
      vscode.env.clipboard.writeText(text);
      vscode.window.setStatusBarMessage('PromptForge: copied to clipboard', 2000);
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
