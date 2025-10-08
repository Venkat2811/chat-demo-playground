# chat-demo-playground

Local UI to run A/B side-by-side against two OpenAI-compatible providers with a single mode (25 concurrent). Streams responses and shows simple per-request metrics.

## Quick start

1) Activate the env (you created earlier):

   - `conda activate chat-demo`

2) Install dependencies (only once):

   - `pip install python-fasthtml httpx pyyaml`

3) Configure via YAML (preferred):

   - Copy `config.example.yaml` to `config.yaml`.
   - Option A: set a single `provider:` block (used for both A and B).
   - Option B: set `providers:` with `A:` and `B:` blocks to compare two distinct providers.
   - Providers must be OpenAI-compatible (`/v1/chat/completions` or `/v1/completions`; `openai-responses` also supported).
   - Datasets expect JSONL with a `prompt_field` (and optionally `token_len_field`).

4) Run the app:

   - `python app.py` (opens on `http://localhost:5001`)

## UI & behavior

- Controls: dataset picker (+ sample), prompt preview, output tokens, and a single run mode: "A/B Side-by-Side (25 concurrent)".
- A single Run button launches 25 concurrent requests; each request fans out to provider A and B with the same prompt.
- Each request card shows two response panes (A and B) streaming independently with live TTFT/TPOT/TPS/token counts.

Notes:

- Token counts for outputs are approximated from streamed chunks. Input token total uses either the dataset-provided token length (`token_len_field`) or a simple whitespace-based estimate.
- If `config.yaml` is missing, the app falls back to environment variables for provider setup.
