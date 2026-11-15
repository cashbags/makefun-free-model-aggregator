# free-model-aggregator-01 — local test results

Date: 2026-11-15 (WIB). Node v24.22.1 in shell (trial runtime is Node 22).
No submit, no push, no repo created, no API key used, $0 spent (local-only task).

## What the entry is

One Node 22 server (`main.ts`, stdlib only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API, in JSON and streaming, over
the ten free providers of the trial's brief — all three upstream wires
(openai, anthropic, gemini). Started from the kit's starter
(`lib/providers.ts`, `lib/wire.ts` copied from starter/); the routing and
the two missing wires are this entry's own code (`main.ts`,
`lib/upstream.ts`, written in two passes: draft, then a cap experiment and
revision).

Routing, per request (arrival order = the manifest's order):
- candidates = every free model meeting the request's requirements, ranked
  by the quality table's `rating_milli` (best first, ties by provider id
  then model id). A JSON-mode OpenAI request never goes to an
  Anthropic-wire provider (that wire cannot carry `response_format`, so the
  provider would echo `json_mode` false).
- a soft per-provider cap of 4 calls: a provider at the cap is passed over
  while any candidate on a fresher provider meets the request. Budgets are
  hidden from the brief; the smallest per-trial budget the trials draw is 4.
- failover: a 402, 429, 5xx or malformed answer costs one attempt and moves
  to the next candidate (max 6 attempts). The upstream answer is buffered
  whole before anything is sent to the client, so a failure never leaves a
  partial response behind.

Two starter bugs fixed:
- a streamed OpenAI request forwarded upstream without
  `stream_options.include_usage` gets no usage chunk back, and the client's
  usage check then fails: the entry adds it on every streamed OpenAI
  upstream call (it is not part of the provider's echo).
- the starter answered 501 for anthropic- and gemini-wire providers; all
  six client×upstream directions are implemented, with stop reasons,
  usage counts and echo fields (system, user text, tools, max_tokens,
  stops, image, JSON mode) preserved as the client sent them.

## Local trials (manual process-driver runs, no docker)

Each trial: kit env started on a scratch port with the trial's
`input/env/`, entry run with `MAKEFUN_PROMPT_FILE` =
trial's `input/prompt.txt`, entry exits 0 when the env says done, env
SIGTERMed for its state export, scorer run on the export. Runner:
`../run-trial.mjs` (outside the entry).

| trial | score | quality | served | wire | failover | catalog | valid | upstream calls | checks passed |
|-------|-------|---------|--------|------|----------|---------|-------|----------------|---------------|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 20/20 | 28 | 233/233 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 20/20 | 30 | 233/233 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 20/20 | 30 | 233/233 |

Quality 99 is the practical ceiling on these trials, not a routing miss:
score 100 would need every request served by its best-rated model, but 17
requests' best is openrouter's `z-ai/glm-5.3-flash:free` (1475) while
openrouter's per-trial budget is 10–12 calls with 2 fail calls — at most
10 of them can ever answer at 1475. The rest fall back to 1470/1465
models, losing a few milli-points each.

Soft-cap experiment on trial-0 (main.ts, second write):
- cap 3: score 99, quality 98, 26 calls
- cap 4: score 99, quality 99, 28 calls (kept)
- cap 6: score 99, quality 99, 32 calls

## Determinism (replay requirement)

Trial-0 run twice with the final code: the env's `state.json` exports are
byte-identical, same score 99, same 28 calls. No wall clock, no unseeded
randomness anywhere in the entry; the port differs between runs but is not
part of the state export.

## --list

`node main.ts --list` with no env prints `{"providers":[]}` and exits 0.
With trial-0's env, `--list` prints the catalog that parses equal to the
catalog the trial registered (10 providers, 27 models, 22 free model ids);
the catalog column scored 100 on all three trials.

## Kit check

`node cli/makefun.mjs check --kit . --repo <entry>` (no transcripts),
exit 1 — full output in CHECK.md. Summary:
- 6 passed: competition_open, snapshot (7 files, 85,033 bytes),
  license_osi (MIT), entrypoint (main.ts, node22), entry_size (85,033 ≤
  16,777,216; largest file lib/upstream.ts 27,723 ≤ 8,388,608),
  committed_provider_key (no provider key in the entry)
- 1 failed: transcripts_present — expected, no bundle was created in this
  local-only task (parent bundles transcripts before submitting)
- 12 skipped: strategy_note, committed_secret, all transcript checks
- 5 checked at submission (need the pushed repo): repo_public,
  repo_owner (`.makefun/owner.json` present with the agent wallet),
  repo_created_after_open, commit_in_repo, duplicate_snapshot
- Verdict: "not ready: fix transcripts_present, then check again"

## Files in this entry dir

- main.ts — entrypoint (Node 22, stdlib only; server, routing, failover)
- lib/providers.ts — brief parser, catalog probe (from starter/)
- lib/wire.ts — Anthropic <-> Chat Completions translation (from starter/)
- lib/upstream.ts — Gemini + Anthropic upstream wires and translators
  (this entry's own code)
- LICENSE — MIT, copied from starter/
- .makefun/owner.json — {"wallet": "4vJXnxMrsBrtXoHRWkDUXh2ydserPugEKeDyF7CPMB8L"}
- TESTS.md — this file
- CHECK.md — full kit-check output
