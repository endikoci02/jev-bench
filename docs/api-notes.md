# API notes (verified against docs, 2026-10-07)

Source of truth for adapter behaviour. Where this differs from the original project brief, the docs win.

## TypeSafe Jev (docs.typesafe.ai)
- `POST /v1/systemone`, `model: "jev-latest"`; response has `model` (resolved version), `answers`, `usage.{input_tokens,output_tokens}`.
- Choice: `criteria` maps option -> description, max 255 options. Answer has `choice` (argmax), `probabilities` (sums to 1), `confidence`.
- **`confidence` is NOT the max probability.** Docs: "1 when all the probability is on one outcome and 0 when the probability is spread evenly" - a concentration statistic over the whole distribution. We therefore report both `max_prob` (calibration target) and the API `confidence`. Docs make no calibration claims.
- `AsyncTypeSafeClient` exists (`async with`), `RetryPolicy(max_retries, backoff_max, timeout)`, errors raise `TypeSafeAPIError(status, request_id)`. 429/529 -> exponential backoff.
- No documented batch API, temperature, or determinism guarantee. No pricing page; price ($0.042/M input, output free) comes from the console and lives in config.

## Anthropic
| Item | Haiku 4.5 (`claude-haiku-4-5-20251001`) | Sonnet 5.5 (`claude-sonnet-5-5`) |
|---|---|---|
| Price in/out per M | $1 / $5 | $2 / $10 |
| Batch API | 50% off | 50% off |
| `temperature` | allowed (we use 0) | **non-default values return 400** -> default sampling |
| Forced `tool_choice` (`any`/`tool`) | allowed | **400** |
| Structured outputs (`output_config.format`) | supported | supported |
| Thinking | off unless enabled | adaptive by default; turn off with `thinking: {type: "between_tools"}` (effort <= high) |
| Min cacheable prefix | 4096 tokens | 512 tokens |

Consequences:
1. Both models use **structured outputs** (`output_config.format`, JSON schema `{label: enum, confidence: number}`) instead of a forced tool call - one mechanism for both, fair.
2. **Sonnet 5.5 cannot run at temperature 0.** It runs at default sampling with thinking off (`between_tools`). Haiku stays at temperature 0. Documented as a limitation; Sonnet's results are single samples from a stochastic decoder.
3. Prompt caching: Haiku's prompts (<= ~3.5K tokens even for CLINC150) are below the 4096 minimum -> no caching. Sonnet could cache Banking77/CLINC150 label lists (>512). Caching works inside batches but hits are best-effort there; we enable it only for Sonnet batch runs and record `cache_read_input_tokens` / `cache_creation_input_tokens` from real usage. Latency runs are run with caching disabled.
4. JSON-schema `minimum`/`maximum` are not supported; the SDK strips them, we validate `0 <= confidence <= 1` client-side and count violations as invalid output.
