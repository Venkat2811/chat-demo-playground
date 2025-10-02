# chat-demo-playground

Local UI to run one provider, stream responses, and show simple benchmark-style metrics.

## Quick start

1) Activate the env (you created earlier):

   - `conda activate chat-demo`

2) Install dependencies (only once):

   - `pip install python-fasthtml httpx pyyaml`

3) Configure via YAML (preferred):

   - Copy `config.example.yaml` to `config.yaml` and edit the single `provider` and any `datasets`.
   - The provider must be OpenAI-compatible (`/v1/chat/completions` or `/v1/completions`).
   - Datasets expect JSONL with a `prompt_field` (and optionally `token_len_field`).

4) Run the app:

   - `python app.py` (opens on `http://localhost:5001`)

## UI & behavior

- Shared controls: dataset picker (+ Random sample), prompt (editable), input token target (auto-filled from sample if available), max output tokens, and run mode selector.
- Single provider. One Run button. Output area is scrollable.
- Modes: Single, 10 req (concurrency 1), 50 req (concurrency 10).
- Layout:
  - Single request: shows one Prompt section and one streaming Response under it.
  - Multi-request: stacks multiple Prompt/Response pairs, each streams independently.
- Per-request headers show TTFT, E2EL, mean ITL, and output token count (approx). A summary block mimics vLLM benchmark formatting.

Notes:

- Token counts for outputs are approximated from streamed chunks. Input token total uses either the dataset-provided token length (`token_len_field`) or a simple whitespace-based estimate.
- If `config.yaml` is missing, the app falls back to environment variables for provider setup.
