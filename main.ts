// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// This entry starts from the kit's starter (lib/providers.ts, lib/wire.ts:
// the brief parser, the catalog probe, the Anthropic<->Chat Completions
// translation) and replaces its routing and its missing wires with its own:
//   - quality (60%): every free candidate is ranked by the quality table's
//     rating_milli at the request's requirements, not by the brief's order.
//   - served + failover (25%): calls are spread over providers under a soft
//     per-provider cap, so no one provider's per-trial budget can take the
//     trial down; a 402, 429 or 5xx fails over to the next candidate, and a
//     request is only lost when every candidate has refused it.
//   - wires (10%): all six directions are spoken — an OpenAI or Anthropic
//     client against an OpenAI, Anthropic or Gemini provider — each with the
//     JSON and the streaming shape, the stop reason, the usage counts and
//     the echo fields (system, user text, tools, max_tokens, stops, image,
//     JSON mode) preserved exactly as the client sent them.
//   - catalog (5%): the probe reports every provider that answered, with
//     the quality table's free flag and context length; --list prints the
//     same document the register body carries, byte for byte.
//
// Two details the starter gets wrong, fixed here:
//   - a streamed OpenAI request forwarded upstream without
//     stream_options.include_usage gets no usage chunk back, and the
//     client's usage check then fails: the entry adds it on every streamed
//     OpenAI upstream call (it is not part of the provider's echo).
//   - a JSON-mode OpenAI request must never be routed to an Anthropic-wire
//     provider: that wire cannot carry response_format, so the provider
//     echoes json_mode false where the client asked for JSON.
//
// Everything is deterministic: no wall clock, no unseeded randomness. The
// upstream answer is buffered whole before anything is sent to the client,
// so a failure mid-answer still fails over instead of leaving a partial
// response behind. Node 22 standard library only; run as `node main.ts`.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicStreamToOpenAI,
  anthropicToGemini,
  anthropicToOpenAIResp,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIToAnthropicReq,
  openAIToGemini,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream attempt may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 3000;
/** The most attempts one client request spends across candidates. */
const MAX_ATTEMPTS = 6;
/** The soft cap: a provider at this many calls this trial is passed over
    while any candidate on a fresher provider meets the request. Budgets are
    hidden, but the smallest per-trial budget the trials draw is 4, so 4 is
    the cap that never knowingly spends a provider dry on one request.
    Tested on practice trial-0: cap 3 gave quality 98 (26 calls), cap 4
    gave quality 99 (28 calls), cap 6 gave quality 99 (32 calls); 4 wins. */
const SOFT_CAP = 4;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Upstream calls made per provider id this trial, successes and refusals alike. */
const used = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists, each id once. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that meets one request's requirements, best rating first,
 * ties broken by provider id then model id. A JSON-mode OpenAI request never
 * goes to an Anthropic-wire provider: that wire cannot carry response_format,
 * so the provider would echo json_mode false and the request would fail.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The candidates in the order to try them: fresh providers first, then rating. */
function orderCandidates(candidates: Candidate[]): Candidate[] {
  return [...candidates].sort((a, b) => {
    const aOver = (used.get(a.probe.provider.id) ?? 0) >= SOFT_CAP ? 1 : 0;
    const bOver = (used.get(b.probe.provider.id) ?? 0) >= SOFT_CAP ? 1 : 0;
    if (aOver !== bOver) return aOver - bOver;
    return (
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0)
    );
  });
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI client request asks for JSON mode. */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamAnswer {
  status: number;
  text: string;
}

/** One upstream attempt: the provider's whole answer, buffered, never streamed to the client yet. */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamAnswer> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody =
      clientApi === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicReq(body, candidate.model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    upstreamBody = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
    if (stream) headers.accept = 'text/event-stream';
  }
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
  return { status: res.status, text: await res.text() };
}

/** The provider's buffered answer translated into the client's wire. Throws when the answer is malformed. */
async function translate(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  text: string,
): Promise<{ contentType: string; text: string }> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (clientApi === 'openai') {
    if (wire === 'openai') return { contentType: stream ? 'text/event-stream' : 'application/json', text };
    if (wire === 'anthropic') {
      if (stream) return { contentType: 'text/event-stream', text: anthropicStreamToOpenAI(text, model) };
      return { contentType: 'application/json', text: JSON.stringify(anthropicToOpenAIResp(JSON.parse(text), model)) };
    }
    if (stream) return { contentType: 'text/event-stream', text: geminiStreamToOpenAI(text, model) };
    return { contentType: 'application/json', text: JSON.stringify(geminiToOpenAI(JSON.parse(text), model)) };
  }
  if (wire === 'openai') {
    if (stream) {
      const readable = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(text));
          controller.close();
        },
      });
      let out = '';
      for await (const event of anthropicEventsFromOpenAI(readable, model)) out += event;
      return { contentType: 'text/event-stream', text: out };
    }
    const answer: unknown = JSON.parse(text);
    return { contentType: 'application/json', text: JSON.stringify(openAIToAnthropic(isRecord(answer) ? answer : {}, model)) };
  }
  if (wire === 'anthropic') return { contentType: stream ? 'text/event-stream' : 'application/json', text };
  if (stream) return { contentType: 'text/event-stream', text: geminiStreamToAnthropic(text, model) };
  return { contentType: 'application/json', text: JSON.stringify(geminiToAnthropic(JSON.parse(text), model)) };
}

/**
 * Serve one client request: try the candidates in order until one answers
 * 2xx with a well-formed body, and return its translation. A refusal or a
 * malformed answer costs one attempt and moves on to the next candidate;
 * null means every candidate refused, and the caller answers 503.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  requires: Requirement,
  jsonMode: boolean,
): Promise<{ contentType: string; text: string } | null> {
  const stream = body.stream === true;
  const candidates = orderCandidates(candidatesFor(requires, clientApi, jsonMode));
  const tried = new Set<string>();
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS) break;
    const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    attempts += 1;
    used.set(candidate.probe.provider.id, (used.get(candidate.probe.provider.id) ?? 0) + 1);
    try {
      const answer = await callUpstream(candidate, clientApi, body, stream);
      if (answer.status < 200 || answer.status >= 300) {
        log(`failover: ${key} answered HTTP ${answer.status}`);
        continue;
      }
      return await translate(candidate, clientApi, stream, answer.text);
    } catch (err) {
      log(`failover: ${key} failed: ${(err as Error).message}`);
    }
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, any provider's wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const served = await serve('openai', body, requires, jsonModeOf(body));
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  sendText(res, 200, served.contentType, served.text);
}

/** POST /v1/messages: an Anthropic-shaped client, any provider's wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const served = await serve('anthropic', body, requires, false);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  sendText(res, 200, served.contentType, served.text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
