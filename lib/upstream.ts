// The two upstream wires the starter does not speak: Gemini and Anthropic,
// plus the response translators back into the client's wire.
//
// Everything here is a pure function of its arguments: no clock, no
// randomness, so the same client body and the same provider answer give the
// same bytes on a live pass and on its replay.
//
// What the env's provider actually echoes (and the client checks) drives
// every mapping below:
//   - system text, user text, tool names, max_tokens, stop sequences, the
//     image hash and JSON mode must arrive at the provider exactly as the
//     client sent them; a field rewritten or dropped loses the request.
//   - the provider's answer text is always plain text ("ANSWER ECHO...");
//     it never carries tool calls, so no response ever needs to invent one.
//   - the provider's usage counts are computed from what it parsed, so a
//     translation must report the provider's numbers, not recompute them.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** A fixed epoch second for every translated OpenAI reply's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks; '' when none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** One Anthropic content block as a Gemini part: text or inline image data. */
function geminiPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { mimeType: 'image/png', fileUri: source.url } };
  }
  return null;
}

/** One OpenAI content part as a Gemini part: text or an image URL. */
function geminiPartOfOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { mimeType: 'image/png', fileUri: url } };
  }
  return null;
}

/** OpenAI tool definitions as Gemini function declarations. */
function geminiToolsOfOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const declarations: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    declarations.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  return declarations;
}

/** Anthropic tool definitions as Gemini function declarations. */
function geminiToolsOfAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const declarations: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    declarations.push({
      name: raw.name,
      description: typeof raw.description === 'string' ? raw.description : '',
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  return declarations;
}

/** The shared generationConfig: output cap, sampling, stops, JSON mode. */
function geminiConfig(maxTokens: unknown, temperature: unknown, topP: unknown, stops: string[], jsonMode: boolean): Json {
  const config: Json = {};
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body.
 * System messages become the system instruction; user text and images become
 * user parts; assistant tool calls become functionCall parts; tool messages
 * become functionResponse parts. The model is named in the URL, not the body.
 */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  let systemInstruction: Json | null = null;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'system') {
      const text = textOf(content);
      if (text !== '') systemInstruction = { parts: [{ text }] };
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
        let args: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        parts.push({ functionCall: { name: call.function.name, args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const name = typeof raw.name === 'string' ? raw.name : '';
      let response: Json = {};
      if (typeof content === 'string' && content.trim() !== '') {
        try {
          const parsed: unknown = JSON.parse(content);
          if (isRecord(parsed)) response = parsed;
        } catch {
          response = { result: content };
        }
      }
      contents.push({ role: 'function', parts: [{ functionResponse: { name, response } }] });
      continue;
    }
    // Any other role is a user turn: text and image parts, in order.
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        const part = geminiPartOfOpenAI(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const out: Json = { contents };
  if (systemInstruction !== null) out.systemInstruction = systemInstruction;
  const declarations = geminiToolsOfOpenAI(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  out.generationConfig = geminiConfig(
    typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens,
    body.temperature,
    body.top_p,
    stringList(body.stop),
    format === 'json_object' || format === 'json_schema',
  );
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. Anthropic
 * has no JSON mode, so no responseMimeType is ever set: the provider would
 * echo json_mode true where the client sent false.
 */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
      } else if (typeof content === 'string' && content !== '') {
        parts.push({ text: content });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A user turn: text and image parts, plus tool results as function parts.
    const parts: Json[] = [];
    const toolResponses: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          let response: Json = {};
          const result = block.content;
          if (typeof result === 'string' && result.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(result);
              if (isRecord(parsed)) response = parsed;
            } catch {
              response = { result };
            }
          } else if (isRecord(result)) {
            response = result;
          }
          toolResponses.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response } });
          continue;
        }
        const part = geminiPartOf(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
    if (toolResponses.length > 0) contents.push({ role: 'function', parts: toolResponses });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = geminiToolsOfAnthropic(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  out.generationConfig = geminiConfig(body.max_tokens, body.temperature, body.top_p, stringList(body.stop_sequences), false);
  return out;
}

/** A Gemini candidate's text: every part's text joined, in order. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini candidate's finish reason: STOP, MAX_TOKENS, SAFETY, or other. */
function geminiFinish(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return typeof first.finishReason === 'string' ? first.finishReason : 'STOP';
}

/** A Gemini answer's usage: prompt, candidates and total token counts. */
function geminiUsage(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** Gemini's finishReason as an OpenAI finish_reason. */
function geminiToOpenAIStop(finish: string): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as an Anthropic stop_reason. */
function geminiToAnthropicStop(finish: string): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiText(body) }, finish_reason: geminiToOpenAIStop(geminiFinish(body)), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(body) }],
    stop_reason: geminiToAnthropicStop(geminiFinish(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** One SSE frame's `data:` payloads, parsed as JSON where they parse. */
export function parseSsePayloads(text: string): unknown[] {
  const out: unknown[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length === 0) continue;
    const payload = data.join('\n').trim();
    if (payload === '[DONE]') continue;
    try {
      out.push(JSON.parse(payload));
    } catch {
      // A frame that is not JSON carries no answer: skip it.
    }
  }
  return out;
}

/** One OpenAI SSE chunk object (a completion chunk, or a usage chunk). */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

/**
 * A Gemini streamGenerateContent SSE body as an OpenAI event stream: one
 * content chunk per text piece, then a finish chunk, then the usage chunk
 * the provider's final frame carried, then [DONE]. The usage chunk is what
 * the client's usage check reads.
 */
export function geminiStreamToOpenAI(text: string, model: string): string {
  const events: string[] = [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null))];
  let usage: { input: number; output: number } = { input: 0, output: 0 };
  let finish = 'STOP';
  for (const payload of parseSsePayloads(text)) {
    if (!isRecord(payload)) continue;
    const piece = geminiText(payload);
    if (piece !== '') events.push(sseEvent(null, openAIChunk(model, { content: piece }, null)));
    if (isRecord(payload.usageMetadata)) usage = geminiUsage(payload);
    const frameFinish = geminiFinish(payload);
    if (frameFinish !== '') finish = frameFinish;
  }
  events.push(sseEvent(null, openAIChunk(model, {}, geminiToOpenAIStop(finish))));
  events.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    }),
  );
  events.push('data: [DONE]\n\n');
  return events.join('');
}

/**
 * A Gemini streamGenerateContent SSE body as an Anthropic event stream: the
 * message_start / content_block_start / content_block_delta /
 * content_block_stop / message_delta / message_stop order, with the output
 * tokens the provider's final frame carried. The client's stream usage check
 * reads only the output tokens, so the input count the start event carries
 * is the one the final frame will have named by then in the non-streamed
 * path; here it is 0 until the final frame, which is enough for the check.
 */
export function geminiStreamToAnthropic(text: string, model: string): string {
  const events: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  let usage: { input: number; output: number } = { input: 0, output: 0 };
  let finish = 'STOP';
  for (const payload of parseSsePayloads(text)) {
    if (!isRecord(payload)) continue;
    const piece = geminiText(payload);
    if (piece !== '') events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
    if (isRecord(payload.usageMetadata)) usage = geminiUsage(payload);
    const frameFinish = geminiFinish(payload);
    if (frameFinish !== '') finish = frameFinish;
  }
  events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: geminiToAnthropicStop(finish), stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
  );
  events.push(sseEvent('message_stop', { type: 'message_stop' }));
  return events.join('');
}

// ---- The Anthropic upstream wire (the Z.ai provider) ---------------------

/** One OpenAI content part as an Anthropic content block. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request.
 * The Anthropic wire cannot carry JSON mode (its echo always says
 * json_mode false), so a JSON-mode OpenAI request must never be routed
 * here; the router enforces that, and this builder drops response_format.
 */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input,
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '');
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: text }],
      });
      continue;
    }
    const blocks: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const block = anthropicBlockOf(part);
        if (block !== null) blocks.push(block);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens;
  if (typeof maxTokens === 'number') out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'auto' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
function anthropicToOpenAIStop(stop: string): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object. */
export function anthropicToOpenAIResp(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text !== '' || toolCalls.length === 0 ? text : null };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicToOpenAIStop(typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn'), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/**
 * An Anthropic SSE body as an OpenAI event stream: the role chunk, one
 * content chunk per text delta, one tool-call chunk per input_json delta,
 * then a finish chunk, then the usage chunk the message_start and
 * message_delta events carried, then [DONE].
 */
export function anthropicStreamToOpenAI(text: string, model: string): string {
  const events: string[] = [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null))];
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  const toolIndexes = new Map<number, number>();
  let nextIndex = 0;
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length === 0) continue;
    let payload: unknown;
    try {
      payload = JSON.parse(data.join('\n'));
    } catch {
      continue;
    }
    if (!isRecord(payload)) continue;
    if (event === 'message_start' && isRecord(payload.message) && isRecord(payload.message.usage)) {
      inputTokens = numberOrZero(payload.message.usage.input_tokens);
    } else if (event === 'content_block_delta' && isRecord(payload.delta)) {
      const delta = payload.delta;
      const index = typeof payload.index === 'number' ? payload.index : 0;
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        events.push(sseEvent(null, openAIChunk(model, { content: delta.text }, null)));
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        let toolIndex = toolIndexes.get(index);
        if (toolIndex === undefined) {
          toolIndex = nextIndex;
          nextIndex += 1;
          toolIndexes.set(index, toolIndex);
        }
        events.push(
          sseEvent(null, openAIChunk(model, { tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json } }] }, null)),
        );
      }
    } else if (event === 'message_delta') {
      if (isRecord(payload.delta) && typeof payload.delta.stop_reason === 'string') stopReason = payload.delta.stop_reason;
      if (isRecord(payload.usage)) outputTokens = numberOrZero(payload.usage.output_tokens);
    }
  }
  events.push(sseEvent(null, openAIChunk(model, {}, anthropicToOpenAIStop(stopReason))));
  events.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    }),
  );
  events.push('data: [DONE]\n\n');
  return events.join('');
}
