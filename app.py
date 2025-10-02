from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional
import random

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional import
    httpx = None  # Will raise at runtime with a clear message

from fasthtml.common import *  # FastHTML/Starlette/HTMX utilities


# ---------- Configuration ----------

@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: Optional[str]
    model: str
    kind: str = "openai-chat"  # one of: openai-chat, openai-completions
    force_output_tokens: bool = False  # if True, try to enforce exact output tokens with provider-specific params


@dataclass
class DatasetConfig:
    name: str
    path: str
    fmt: str = "jsonl"  # currently supports jsonl
    prompt_field: str = "prompt"
    response_field: Optional[str] = "response"
    prompt_tokens_field: Optional[str] = "prompt_tokens"
    response_tokens_field: Optional[str] = "response_tokens"


@dataclass
class AppConfig:
    provider: Optional[dict]
    providers: dict
    datasets: list


def _read_yaml_config() -> Optional[AppConfig]:
    cfg_path = "config.yaml"
    if not os.path.exists(cfg_path):
        cfg_path = "config.example.yaml"
        if not os.path.exists(cfg_path):
            return None
    if yaml is None:
        raise RuntimeError("pyyaml is required for YAML config. Install with: pip install pyyaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    provider = data.get("provider")
    providers = data.get("providers", {})
    datasets = data.get("datasets", [])
    return AppConfig(provider=provider, providers=providers, datasets=datasets)


def _from_yaml_provider(app_cfg: Optional[AppConfig]) -> ProviderConfig:
    def mk(name: str, node: dict) -> ProviderConfig:
        return ProviderConfig(
            name=name,
            base_url=str(node.get("base_url", "http://86.38.238.33:8000/v1")),
            api_key=node.get("api_key"),
            model=str(node.get("model", "openai/gpt-oss-120b")),
            kind=str(node.get("kind", "openai-chat")),
            force_output_tokens=bool(node.get("force_output_tokens", False)),
        )

    if app_cfg:
        if app_cfg.provider:
            return mk("MAIN", app_cfg.provider)
        if app_cfg.providers:
            # If legacy A/B provided, prefer A; else first entry
            if "A" in app_cfg.providers:
                return mk("MAIN", app_cfg.providers.get("A", {}))
            # Pick first
            key = next(iter(app_cfg.providers.keys()))
            return mk("MAIN", app_cfg.providers[key])

    # Default configuration
    return ProviderConfig(
        "MAIN",
        "http://86.38.238.33:8000/v1",
        None,
        "openai/gpt-oss-120b",
        "openai-chat",
    )


APP_CFG = _read_yaml_config()
PROV = _from_yaml_provider(APP_CFG)

# Cache dataset entries
_dataset_cache: dict[str, list[dict]] = {}

def _datasets_list() -> list[DatasetConfig]:
    dss: list[DatasetConfig] = []

    # Always add ShareGPT dataset
    dss.append(DatasetConfig(
        name="sharegpt_min1k_50",
        path="/Users/venkat/Downloads/sharegpt_min1k_50.jsonl",
        fmt="jsonl",
        prompt_field="prompt",
        response_field="response",
        prompt_tokens_field="prompt_tokens",
        response_tokens_field="response_tokens"
    ))

    if APP_CFG and APP_CFG.datasets:
        for d in APP_CFG.datasets:
            try:
                dss.append(DatasetConfig(
                    name=str(d.get("name")),
                    path=str(d.get("path")),
                    fmt=str(d.get("fmt", "jsonl")),
                    prompt_field=str(d.get("prompt_field", "prompt")),
                    response_field=d.get("response_field"),
                    prompt_tokens_field=d.get("prompt_tokens_field"),
                    response_tokens_field=d.get("response_tokens_field"),
                ))
            except Exception:
                continue
    return dss

def _dataset_by_name(name: str) -> Optional[DatasetConfig]:
    for d in _datasets_list():
        if d.name == name:
            return d
    return None

def _load_sharegpt_format(path: str, max_entries: int = 50) -> list[dict]:
    """Load ShareGPT format dataset and extract prompts/responses."""
    items: list[dict] = []

    # Try to download ShareGPT sample if not exists
    if not os.path.exists(path):
        try:
            import urllib.request
            url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
            print(f"Downloading ShareGPT sample dataset...")
            urllib.request.urlretrieve(url, path)
        except:
            # Create a sample dataset
            sample_data = []
            for i in range(50):
                sample_data.append({
                    "conversations": [
                        {"from": "human", "value": f"Please explain the concept of {['machine learning', 'quantum computing', 'blockchain', 'artificial intelligence', 'data science'][i % 5]} in simple terms."},
                        {"from": "gpt", "value": f"I'll explain this concept in simple terms. {'This is a sample response. ' * 20}"}
                    ]
                })
            with open(path, 'w') as f:
                json.dump(sample_data, f)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f) if path.endswith('.json') else [json.loads(line) for line in f]

    # Extract prompts and responses
    count = 0
    for conv in data:
        if count >= max_entries:
            break

        if 'conversations' in conv:
            messages = conv['conversations']
            prompt = None
            response = None

            for i, msg in enumerate(messages):
                if msg.get('from') == 'human':
                    prompt = msg.get('value', '')
                    # Look for the next assistant response
                    if i + 1 < len(messages) and messages[i + 1].get('from') == 'gpt':
                        response = messages[i + 1].get('value', '')
                        break

            if prompt and response:
                items.append({
                    'prompt': prompt,
                    'response': response,
                    'prompt_tokens': len(prompt.split()),
                    'response_tokens': len(response.split())
                })
                count += 1

    return items

def _load_jsonl(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def _load_dataset_entries(ds: DatasetConfig, max_entries: int = 50) -> list[dict]:
    if ds.name in _dataset_cache:
        return _dataset_cache[ds.name][:max_entries]

    if ds.fmt == "sharegpt":
        entries = _load_sharegpt_format(ds.path, max_entries)
    elif ds.fmt == "jsonl":
        entries = _load_jsonl(ds.path)[:max_entries]
    else:
        raise RuntimeError(f"Unsupported dataset fmt: {ds.fmt}")

    _dataset_cache[ds.name] = entries
    return entries[:max_entries]


# ---------- Utility functions ----------

def now() -> float:  # monotonic timestamp
    return time.perf_counter()


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1


def safe_len_tokens(text: str) -> int:
    # Cheap approximation: whitespace tokenization
    return len([t for t in text.split() if t])


def _extract_prompt_and_response(entry: dict, ds: DatasetConfig) -> tuple[str, str, int, int]:
    """Extract prompt, response, and token counts from dataset entry."""
    prompt = entry.get(ds.prompt_field, "")
    response = entry.get(ds.response_field, "")

    if isinstance(prompt, list):
        prompt = "\n".join(str(x) for x in prompt)
    else:
        prompt = str(prompt)

    if isinstance(response, list):
        response = "\n".join(str(x) for x in response)
    else:
        response = str(response)

    # Token counts
    prompt_tokens = entry.get(ds.prompt_tokens_field, 0) if ds.prompt_tokens_field else safe_len_tokens(prompt)
    response_tokens = entry.get(ds.response_tokens_field, 0) if ds.response_tokens_field else safe_len_tokens(response)

    return prompt, response, int(prompt_tokens), int(response_tokens)


# ---------- FastHTML app ----------

hdrs = (
    # Enable HTMX SSE extension
    Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),
    Style(
        """
        :root {
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #16a34a;
            --danger-color: #dc2626;
            --bg-color: #ffffff;
            --card-bg: #f8fafc;
            --border-color: #e2e8f0;
            --text-muted: #64748b;
            --radius: 8px;
            --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
            --baseline-color: #2563eb;
            --brrrllm-color: #fb923c;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: #ffffff;
            color: #1e293b;
            line-height: 1.6;
        }

        html {
            background: #ffffff;
        }

        .container {
            max-width: 2200px;
            width: min(96vw, 2200px);
            margin: 0 auto;
            padding: 16px;
        }

        /* Page Title */
        h1 {
            color: #1e293b !important;
            font-weight: 700;
        }

        /* Comparison Banner */
        .comparison-banner {
            text-align: center;
            padding: 15px;
            background: linear-gradient(90deg, #f0f9ff 0%, #fff7ed 100%);
            border-radius: var(--radius);
            margin-bottom: 10px;
            border: 1px solid var(--border-color);
        }

        .comparison-text {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.5px;
        }

        .baseline-text {
            color: var(--baseline-color);
        }

        .vs-text {
            color: var(--text-muted);
            margin: 0 15px;
        }

        .brrrllm-text {
            color: var(--brrrllm-color);
        }

        /* Header Styles */
        .header {
            background: var(--card-bg);
            padding: 14px;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            margin-bottom: 12px;
            border: 1px solid var(--border-color);
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
        }

        .header h1 {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--primary-color);
        }

        .model-info {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 6px 10px;
            background: white;
            border-radius: var(--radius);
            font-size: 0.95rem;
            border: 1px solid var(--border-color);
        }

        .model-info code {
            background: #e2e8f0;
            color: #1e293b;
            padding: 4px 10px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-weight: 600;
        }

        /* Control Panel */
        .control-panel {
            background: var(--card-bg);
            padding: 14px;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            margin-bottom: 16px;
            border: 1px solid var(--border-color);
        }

        .control-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }

        @media (min-width: 768px) {
            .control-grid {
                grid-template-columns: 2fr 1fr;
            }
        }

        .control-section { display: flex; flex-direction: column; gap: 10px; }

        .form-group { display: flex; flex-direction: column; gap: 6px; }

        .form-group label {
            font-weight: 600;
            font-size: 0.9rem;
            color: #475569;
        }

        .form-group select,
        .form-group input,
        .form-group textarea {
            padding: 8px 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 0.95rem;
            background: white;
            color: #1e293b !important;
            font-weight: 500;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .form-group select:focus,
        .form-group input:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgb(37 99 235 / 0.1);
        }

        .form-inline {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .radio-group {
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding: 12px;
            background: white;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }

        .radio-label {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            font-size: 0.95rem;
        }

        input[type="radio"] {
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
            width: 16px !important;
            height: 16px !important;
            min-width: 16px !important;
            min-height: 16px !important;
            max-width: 16px !important;
            max-height: 16px !important;
            border: 2px solid #cbd5e1;
            border-radius: 50%;
            margin-right: 8px;
            cursor: pointer;
            background: white;
            flex-shrink: 0;
            display: inline-block;
            vertical-align: middle;
        }

        input[type="radio"]:checked {
            border: 2px solid #1e293b;
            background: #1e293b;
            box-shadow: inset 0 0 0 3px white;
        }

        input[type="radio"]:hover:not(:checked) {
            border-color: #94a3b8;
        }

        /* Buttons */
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: var(--primary-color);
            color: white;
        }

        .btn-primary:hover {
            background: #1d4ed8;
        }

        .btn-secondary {
            background: var(--secondary-color);
            color: white;
        }

        .btn-secondary:hover {
            background: #475569;
        }

        .btn-success {
            background: var(--success-color);
            color: white;
        }

        .btn-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        /* Results Section */
        .results-container {
            margin-top: 20px;
        }

        .run-container {
            background: var(--card-bg);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            margin-bottom: 20px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }

        .run-header {
            padding: 15px 20px;
            background: linear-gradient(135deg, var(--primary-color), #3b82f6);
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .run-header h3 {
            font-size: 1.1rem;
            font-weight: 600;
        }

        .run-meta {
            display: flex;
            gap: 15px;
            font-size: 0.9rem;
            opacity: 0.95;
        }

        .run-body {
            padding: 20px;
            background: white;
        }

        /* Request/Response Cards */
        .request-grid {
            display: grid;
            gap: 15px;
            margin-bottom: 20px;
        }

        .request-card {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            overflow: hidden;
        }

        .request-header {
            padding: 12px 15px;
            background: var(--card-bg);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .request-number {
            font-weight: 600;
            color: var(--primary-color);
        }

        .request-metrics {
            display: flex;
            gap: 15px;
            font-size: 0.85rem;
            font-family: 'Monaco', 'Menlo', monospace;
        }

        .metric {
            display: flex;
            gap: 5px;
            align-items: center;
        }

        .metric-label {
            color: var(--text-muted);
        }

        .metric-value {
            font-weight: 600;
            color: #1e293b;
        }

        .request-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            height: 250px;
        }

        .prompt-section,
        .response-section {
            padding: 15px;
            overflow-y: auto;
            position: relative;
        }

        .prompt-section {
            background: #f8fafc;
            border-right: 1px solid var(--border-color);
        }

        .response-section {
            background: white;
        }

        .section-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 10px;
            position: sticky;
            top: 0;
            background: inherit;
            padding-bottom: 5px;
        }

        .content-text {
            font-size: 0.9rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        /* Summary Stats */
        .summary-card {
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border-radius: var(--radius);
            padding: 20px;
            margin-top: 20px;
            border: 1px solid var(--border-color);
        }

        .summary-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .stat-item {
            background: white;
            padding: 12px 15px;
            border-radius: 6px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            border: 1px solid var(--border-color);
        }

        .stat-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stat-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-top: 5px;
        }

        /* Loading and Status */
        .loading-indicator {
            display: inline-block;
            width: 15px;
            height: 15px;
            border: 2px solid #e2e8f0;
            border-top-color: var(--primary-color);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .status-badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .status-running {
            background: #fef3c7;
            color: #92400e;
        }

        .status-complete {
            background: #d1fae5;
            color: #065f46;
        }

        .token-counter {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            background: white;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            font-size: 0.9rem;
        }

        .token-counter .count {
            font-weight: 600;
            color: var(--primary-color);
        }

        /* Prompts Section */
        .prompts-section {
            margin: 20px 0;
        }

        .prompt-boxes-grid {
            display: grid;
            gap: 15px;
            margin-top: 20px;
        }

        .prompt-box {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 15px;
            box-shadow: var(--shadow);
        }

        .prompt-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
        }

        .prompt-number {
            font-weight: 600;
            color: var(--primary-color);
            font-size: 0.95rem;
        }

        .token-info {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-family: monospace;
        }

        .prompt-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        .prompt-input-section,
        .prompt-output-section {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .section-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .prompt-textarea {
            width: 100%;
            min-height: 200px;
            max-height: 400px;
            padding: 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.85rem;
            resize: vertical;
            background: #f8fafc;
            color: #1e293b !important;
            font-weight: 500;
        }

        .output-area {
            width: 100%;
            min-height: 200px;
            max-height: 400px;
            padding: 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.85rem;
            background: white;
            overflow-y: auto;
        }

        /* Performance Summary */
        .perf-summary {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: var(--shadow);
        }

        .perf-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }

        .perf-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #1e293b;
        }

        .expand-btn {
            background: none;
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            color: var(--primary-color);
            font-weight: 600;
            transition: all 0.2s;
        }

        .expand-btn:hover {
            background: var(--bg-color);
        }

        .perf-metrics {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            white-space: pre;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        @media (min-width: 1200px) {
            .perf-metrics { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
        }

        .perf-metrics-collapsed {
            max-height: none;
            overflow: hidden;
        }

        .perf-metrics-expanded {
            max-height: none;
        }

        .metric-section {
            margin: 0;
            padding: 10px;
            background: #f8fafc;
            border-radius: 6px;
        }

        .metric-section-title {
            color: var(--primary-color);
            font-weight: 700;
            margin: 0 0 8px 0;
            font-size: 0.95rem;
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        """
    ),
)

app, rt = fast_app(hdrs=hdrs)


# ---------- Provider streaming ----------

class StreamError(Exception):
    pass


@dataclass
class RequestMetrics:
    request_id: int
    start_time: float
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    token_times: list[float] = field(default_factory=list)
    output_tokens: int = 0
    input_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0  # Time per output token
    tps: float = 0.0  # Tokens per second
    e2e_ms: float = 0.0
    itl_list: list[float] = field(default_factory=list)  # Inter-token latencies


async def stream_openai_chat(
    cfg: ProviderConfig,
    prompt: str,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    """Yield content tokens via OpenAI-compatible chat streaming."""
    if httpx is None:
        raise StreamError("httpx is required. Install with: pip install httpx")

    headers = {"Content-Type": "application/json"}
    # Only add Authorization when a non-empty API key is provided
    if cfg.api_key and str(cfg.api_key).strip().lower() not in ("", "none", "null"):
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    # Base payload. If force_output_tokens is set, enforce strict settings.
    body = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    if getattr(cfg, "force_output_tokens", False):
        body["min_tokens"] = max_tokens

    url = cfg.base_url.rstrip("/") + "/chat/completions"

    async def do_stream(stream_body):
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json=stream_body) as resp:
                if resp.status_code >= 400:
                    raise StreamError(f"Provider {cfg.name} HTTP {resp.status_code}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        js = json.loads(data)
                    except Exception:
                        continue
                    for ch in js.get("choices", []):
                        delta = ch.get("delta", {}).get("content")
                        if delta:
                            yield delta

    # No fallback: if strict settings fail, propagate the error
    async for tok in do_stream(body):
        yield tok


async def stream_openai_completions(
    cfg: ProviderConfig,
    prompt: str,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    """Yield content tokens via OpenAI-compatible completions streaming."""
    if httpx is None:
        raise StreamError("httpx is required. Install with: pip install httpx")

    headers = {"Content-Type": "application/json"}
    if cfg.api_key and str(cfg.api_key).strip().lower() not in ("", "none", "null"):
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    body = {
        "model": cfg.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }

    url = cfg.base_url.rstrip("/") + "/completions"

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code >= 400:
                    error_text = await resp.aread()
                    print(f"Error response: {error_text}")
                    raise StreamError(f"Provider {cfg.name} HTTP {resp.status_code}: {error_text}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        js = json.loads(data)
                    except Exception:
                        continue
                    for ch in js.get("choices", []):
                        text = ch.get("text")
                        if text:
                            yield text
        except httpx.HTTPError as e:
            raise StreamError(str(e)) from e


async def stream_openai_responses(
    cfg: ProviderConfig,
    prompt: str,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    """Yield content tokens via vLLM OpenAI-compatible Responses streaming.

    Uses the /v1/responses endpoint with output token controls and parses
    response.output_text.delta events. When force_output_tokens is enabled,
    sends min_output_tokens=max_tokens to better enforce exact output length
    on vLLM.
    """
    if httpx is None:
        raise StreamError("httpx is required. Install with: pip install httpx")

    headers = {"Content-Type": "application/json"}
    if cfg.api_key and str(cfg.api_key).strip().lower() not in ("", "none", "null"):
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    body = {
        "model": cfg.model,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    if getattr(cfg, "force_output_tokens", False):
        # Stronger enforcement on vLLM
        body["min_output_tokens"] = max_tokens

    url = cfg.base_url.rstrip("/") + "/responses"

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code >= 400:
                    err = await resp.aread()
                    raise StreamError(f"Provider {cfg.name} HTTP {resp.status_code}: {err}")
                async for raw in resp.aiter_lines():
                    if not raw:
                        continue
                    if raw.startswith(":"):
                        continue
                    if not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        js = json.loads(data)
                    except Exception:
                        continue
                    et = js.get("type")
                    if et == "response.output_text.delta":
                        delta = js.get("delta")
                        if delta:
                            yield str(delta)
                    # Periodic/final usage
                    usage = None
                    if isinstance(js.get("usage"), dict):
                        usage = js.get("usage")
                    elif isinstance(js.get("response"), dict) and isinstance(js["response"].get("usage"), dict):
                        usage = js["response"]["usage"]
                    if usage and isinstance(usage.get("output_tokens"), int):
                        out_tok = usage.get("output_tokens")
                        in_tok = usage.get("input_tokens") if isinstance(usage.get("input_tokens"), int) else None
                        if in_tok is not None:
                            yield "\x00USAGE:" + f"{out_tok},{in_tok}"
                        else:
                            yield "\x00USAGE:" + str(out_tok)
        except httpx.HTTPError as e:
            raise StreamError(str(e)) from e


async def stream_provider(cfg: ProviderConfig, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
    kind = (cfg.kind or "openai-chat").lower()
    if "responses" in kind:
        async for t in stream_openai_responses(cfg, prompt, max_tokens):
            yield t
    elif "chat" in kind:
        async for t in stream_openai_chat(cfg, prompt, max_tokens):
            yield t
    else:
        async for t in stream_openai_completions(cfg, prompt, max_tokens):
            yield t


# ---------- Batch runner with true concurrency ----------

async def run_one_request(
    cfg: ProviderConfig,
    prompt: str,
    expected_response: str,
    in_tokens: int,
    max_tokens: int,
    enqueue,
    tag: str,
    req_id: int,
) -> RequestMetrics:
    metrics = RequestMetrics(
        request_id=req_id,
        start_time=now()
    )

    out_text_parts: list[str] = []
    last_token_time: Optional[float] = None

    # Inform client to create the req box
    await enqueue("create_box", tag, req_id, {
        "prompt": prompt,
        "expected_response": expected_response,
        "in_tokens": in_tokens
    })

    try:
        token_count = 0
        metrics.input_tokens = in_tokens
        async for tok in stream_provider(cfg, prompt, max_tokens):
            # Special sentinel from Responses API to inject server-reported usage
            if tok.startswith("\x00USAGE:"):
                try:
                    payload = tok.split(":", 1)[1]
                    if "," in payload:
                        out_str, in_str = payload.split(",", 1)
                        metrics.output_tokens = int(out_str)
                        metrics.input_tokens = int(in_str)
                    else:
                        metrics.output_tokens = int(payload)
                except Exception:
                    pass
                # Send a metrics-only update
                await enqueue("token", tag, req_id, {
                    "token": "",
                    "metrics": {
                        "ttft_ms": metrics.ttft_ms,
                        "tpot_ms": metrics.tpot_ms,
                        "tps": metrics.tps,
                        "tokens": metrics.output_tokens,
                        "in_tokens": metrics.input_tokens
                    }
                })
                continue
            t = now()
            if metrics.first_token_time is None:
                metrics.first_token_time = t
                metrics.ttft_ms = (t - metrics.start_time) * 1000.0
            else:
                # Calculate inter-token latency
                if last_token_time is not None:
                    itl = (t - last_token_time) * 1000.0
                    metrics.itl_list.append(itl)

            out_text_parts.append(tok)
            token_count += 1
            metrics.token_times.append(t)
            # Approximate token count based on whitespace tokens of full text so far
            try:
                metrics.output_tokens = safe_len_tokens("".join(out_text_parts))
            except Exception:
                metrics.output_tokens = token_count
            last_token_time = t

            # Calculate live metrics
            if len(metrics.token_times) > 1:
                time_since_first = t - metrics.first_token_time
                metrics.tpot_ms = (time_since_first / (token_count - 1)) * 1000.0 if token_count > 1 else 0
                total_time = t - metrics.start_time
                # Approximate tokens per second using the estimated output token count
                metrics.tps = (metrics.output_tokens / total_time) if total_time > 0 else 0

            # Send token and live metrics
            await enqueue("token", tag, req_id, {
                "token": tok,
                "metrics": {
                    "ttft_ms": metrics.ttft_ms,
                    "tpot_ms": metrics.tpot_ms,
                    "tps": metrics.tps,
                    "tokens": metrics.output_tokens
                }
            })

    except Exception as e:
        await enqueue("error", tag, req_id, str(e))
        if metrics.first_token_time is None:
            metrics.first_token_time = now()
            metrics.ttft_ms = (metrics.first_token_time - metrics.start_time) * 1000.0

    metrics.end_time = now()
    metrics.e2e_ms = (metrics.end_time - metrics.start_time) * 1000.0
    # Strict check: fail if output tokens do not match requested max_tokens
    if metrics.output_tokens != max_tokens:
        raise StreamError(f"Output tokens {metrics.output_tokens} != expected {max_tokens}")

    await enqueue("end", tag, req_id, metrics)
    return metrics


async def run_batch(
    cfg: ProviderConfig,
    dataset_entries: list[dict],
    ds_config: DatasetConfig,
    out_tokens: int,
    total_reqs: int,
    concurrency: int,
    tag: str,
    use_loaded: bool = False,
) -> AsyncGenerator[bytes, None]:
    """Main SSE generator for batch processing with true concurrency."""
    queue: asyncio.Queue[tuple[str, str, int, Any]] = asyncio.Queue()
    all_metrics: list[RequestMetrics] = []
    metrics_lock = asyncio.Lock()

    async def enqueue(kind: str, tag: str, req_id: int, payload: Any):
        await queue.put((kind, tag, req_id, payload))

    # True concurrent execution
    batch_start = now()

    async def task_fn(i: int):
        entry = dataset_entries[min(i - 1, len(dataset_entries) - 1)]
        prompt, response, p_tokens, r_tokens = _extract_prompt_and_response(entry, ds_config)
        metrics = await run_one_request(cfg, prompt, response, p_tokens, out_tokens, enqueue, tag, i)
        async with metrics_lock:
            all_metrics.append(metrics)

    # Launch tasks with concurrency limiting
    print(f"run_batch: total_reqs={total_reqs}, concurrency={concurrency}, dataset_entries={len(dataset_entries)}")
    max_conc = max(1, int(concurrency or 1))
    sem = asyncio.Semaphore(max_conc)

    async def limited_task(i: int):
        async with sem:
            await task_fn(i)

    tasks: list[asyncio.Task] = []
    for i in range(1, min(total_reqs, len(dataset_entries)) + 1):
        tasks.append(asyncio.create_task(limited_task(i)))

    completed = 0
    completed_lock = asyncio.Lock()

    async def gen_html(kind: str, tag: str, req_id: int, payload: Any) -> str:
        nonlocal completed

        if kind == "create_box":
            prm = str(payload.get("prompt", ""))[:500]  # Truncate for display
            in_tok = int(payload.get("in_tokens", 0))

            if use_loaded:
                # Reuse the pre-rendered prompt card response area. Place metrics above the
                # scrollable output box by wrapping and putting the scroll class on the child.
                outer_id = f"output-{req_id-1}"
                html = (
                    f'<div id="{outer_id}" hx-swap-oob="outerHTML">'
                    f'  <div class="request-metrics" style="margin-bottom:6px">'
                    f'    <span class="metric-label">TTFT: </span><span class="metric-value" id="ttft-{tag}-{req_id}">--</span>'
                    f'    <span class="metric-label" style="margin-left:10px">TPOT: </span><span class="metric-value" id="tpot-{tag}-{req_id}">--</span>'
                    f'    <span class="metric-label" style="margin-left:10px">TPS: </span><span class="metric-value" id="tps-{tag}-{req_id}">--</span>'
                    f'    <span class="metric-label" style="margin-left:10px">Tokens: </span><span class="metric-value" id="tokens-{tag}-{req_id}">0</span>'
                    f'  </div>'
                    f'  <div class="output-area">'
                    f'    <div id="out-{tag}-{req_id}" class="content-text"></div>'
                    f'  </div>'
                    f'</div>'
                )
                return html
            else:
                box = Div(
                    Div(
                        Div(f"Request #{req_id}", cls="request-number"),
                        Div(
                            Div(Span("TTFT: ", cls="metric-label"), Span("--", id=f"ttft-{tag}-{req_id}", cls="metric-value"), cls="metric"),
                            Div(Span("TPOT: ", cls="metric-label"), Span("--", id=f"tpot-{tag}-{req_id}", cls="metric-value"), cls="metric"),
                            Div(Span("TPS: ", cls="metric-label"), Span("--", id=f"tps-{tag}-{req_id}", cls="metric-value"), cls="metric"),
                            Div(Span("Tokens: ", cls="metric-label"), Span("0", id=f"tokens-{tag}-{req_id}", cls="metric-value"), cls="metric"),
                            cls="request-metrics"
                        ),
                        cls="request-header"
                    ),
                    Div(
                        Div(
                            Div("PROMPT", cls="section-label"),
                            Div(prm, cls="content-text"),
                            cls="prompt-section"
                        ),
                        Div(
                            Div("RESPONSE", cls="section-label"),
                            Div(id=f"out-{tag}-{req_id}", cls="content-text"),
                            cls="response-section"
                        ),
                        cls="request-content"
                    ),
                    id=f"req-{tag}-{req_id}",
                    cls="request-card",
                    hx_swap_oob=f"beforeend:#reqs-{tag}",
                )
                return to_xml(box)

        elif kind == "token":
            data = payload
            token = data.get("token", "")
            metrics = data.get("metrics", {})

            print(f"Token update: req_id={req_id}, token_len={len(token)}, tokens={metrics.get('tokens', 0)}")

            # Build OOB updates with proper HTML escaping
            import html as html_lib

            html_parts = []

            # Append token to response area (escape HTML entities)
            if token:
                escaped_token = html_lib.escape(token)
                html_parts.append(f'<span id="out-{tag}-{req_id}" hx-swap-oob="beforeend">{escaped_token}</span>')

            # Update metrics with replacement
            html_parts.append(f'<span id="ttft-{tag}-{req_id}" hx-swap-oob="true">{metrics.get("ttft_ms", 0):.0f}ms</span>')
            html_parts.append(f'<span id="tpot-{tag}-{req_id}" hx-swap-oob="true">{metrics.get("tpot_ms", 0):.1f}ms</span>')
            html_parts.append(f'<span id="tps-{tag}-{req_id}" hx-swap-oob="true">{metrics.get("tps", 0):.1f}</span>')
            html_parts.append(f'<span id="tokens-{tag}-{req_id}" hx-swap-oob="true">{metrics.get("tokens", 0)}</span>')

            # If available, update the input token counter in the loaded prompt header
            if "in_tokens" in metrics:
                idx = req_id - 1
                html_parts.append(f'<span id="in-tokens-{idx}" hx-swap-oob="true"> - Input: {int(metrics.get("in_tokens", 0))} tokens</span>')

            return "".join(html_parts)

        elif kind == "end":
            async with completed_lock:
                completed += 1
            return ""

        elif kind == "summary":
            # Generate comprehensive summary statistics
            async with metrics_lock:
                ttfts = [m.ttft_ms for m in all_metrics if m.ttft_ms > 0]
                tpots = [m.tpot_ms for m in all_metrics if m.tpot_ms > 0]
                e2es = [m.e2e_ms for m in all_metrics if m.e2e_ms > 0]
                all_itls = []
                for m in all_metrics:
                    all_itls.extend(m.itl_list)
                # Prefer provider-reported input tokens; fall back to prompt whitespace approx
                total_input_tokens = 0
                for m in all_metrics:
                    if m.input_tokens and m.input_tokens > 0:
                        total_input_tokens += m.input_tokens
                    else:
                        try:
                            entry = dataset_entries[min(m.request_id-1, len(dataset_entries)-1)]
                            total_input_tokens += safe_len_tokens(str(entry.get('prompt','')))
                        except Exception:
                            pass
                total_output_tokens = sum(m.output_tokens for m in all_metrics)

            duration = now() - batch_start
            throughput = completed / duration if duration > 0 else 0
            output_token_throughput = total_output_tokens / duration if duration > 0 else 0
            total_token_throughput = (total_input_tokens + total_output_tokens) / duration if duration > 0 else 0

            summary = Div(
                Div("Summary Statistics", cls="summary-title"),
                Div(
                    Div(
                        Div("Successful Requests", cls="stat-label"),
                        Div(str(completed), cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("Duration", cls="stat-label"),
                        Div(f"{duration:.2f}s", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("Request Throughput", cls="stat-label"),
                        Div(f"{throughput:.2f} req/s", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("Output Token Throughput", cls="stat-label"),
                        Div(f"{output_token_throughput:.1f} tok/s", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("Total Token Throughput", cls="stat-label"),
                        Div(f"{total_token_throughput:.1f} tok/s", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("Total Output Tokens", cls="stat-label"),
                        Div(str(total_output_tokens), cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("Mean TTFT", cls="stat-label"),
                        Div(f"{(sum(ttfts)/len(ttfts) if ttfts else 0):.0f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("P50 TTFT", cls="stat-label"),
                        Div(f"{pct(ttfts, 50):.0f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("P90 TTFT", cls="stat-label"),
                        Div(f"{pct(ttfts, 90):.0f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("P95 TTFT", cls="stat-label"),
                        Div(f"{pct(ttfts, 95):.0f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("P99 TTFT", cls="stat-label"),
                        Div(f"{pct(ttfts, 99):.0f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("Mean TPOT", cls="stat-label"),
                        Div(f"{(sum(tpots)/len(tpots) if tpots else 0):.1f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("P50 TPOT", cls="stat-label"),
                        Div(f"{pct(tpots, 50):.1f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("P90 TPOT", cls="stat-label"),
                        Div(f"{pct(tpots, 90):.1f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("Mean ITL", cls="stat-label"),
                        Div(f"{(sum(all_itls)/len(all_itls) if all_itls else 0):.1f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("P50 ITL", cls="stat-label"),
                        Div(f"{pct(all_itls, 50):.1f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("P90 ITL", cls="stat-label"),
                        Div(f"{pct(all_itls, 90):.1f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("Mean E2E", cls="stat-label"),
                        Div(f"{(sum(e2es)/len(e2es) if e2es else 0):.0f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("P50 E2E", cls="stat-label"),
                        Div(f"{pct(e2es, 50):.0f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    Div(
                        Div("P99 E2E", cls="stat-label"),
                        Div(f"{pct(e2es, 99):.0f}ms", cls="stat-value"),
                        cls="stat-item"
                    ),
                    cls="stats-grid"
                ),
                id=f"summary-{tag}",
                cls="summary-card",
                hx_swap_oob="outerHTML"
            )
            # Update the top-level perf summary placeholder using the collapsible layout
            # defined on page load (with Expand All / Collapse behavior).
            lines = (
                f"Successful requests:                {completed}\n"
                f"Benchmark duration (s):             {duration:.2f}\n"
                f"Total input tokens:                 {total_input_tokens}\n"
                f"Total generated tokens:             {total_output_tokens}\n"
                f"Request throughput (req/s):         {throughput:.2f}\n"
                f"Output token throughput (tok/s):    {output_token_throughput:.1f}"
            )
            ttft_lines = (
                f"Mean TTFT (ms):      {(sum(ttfts)/len(ttfts) if ttfts else 0):.0f}\n"
                f"Median TTFT (ms):    {pct(ttfts, 50):.0f}\n"
                f"P90 TTFT (ms):       {pct(ttfts, 90):.0f}\n"
                f"P99 TTFT (ms):       {pct(ttfts, 99):.0f}"
            )
            tpot_lines = (
                f"Mean TPOT (ms):      {(sum(tpots)/len(tpots) if tpots else 0):.1f}\n"
                f"Median TPOT (ms):    {pct(tpots, 50):.1f}\n"
                f"P90 TPOT (ms):       {pct(tpots, 90):.1f}"
            )
            itl_lines = (
                f"Mean ITL (ms):       {(sum(all_itls)/len(all_itls) if all_itls else 0):.1f}\n"
                f"Median ITL (ms):     {pct(all_itls, 50):.1f}\n"
                f"P90 ITL (ms):        {pct(all_itls, 90):.1f}"
            )
            e2e_lines = (
                f"Mean E2EL (ms):      {(sum(e2es)/len(e2es) if e2es else 0):.0f}\n"
                f"Median E2EL (ms):    {pct(e2es, 50):.0f}\n"
                f"P90 E2EL (ms):       {pct(e2es, 90):.0f}\n"
                f"P99 E2EL (ms):       {pct(e2es, 99):.0f}"
            )

            top_summary = Div(
                Div(
                    Div("Performance Summary", cls="perf-title"),
                    Button(
                        "Expand All",
                        cls="expand-btn",
                        onclick="togglePerfMetrics()",
                        id="expand-toggle"
                    ),
                    cls="perf-header"
                ),
                Div(
                    Div(lines, cls="metric-section"),
                    Div(Div("Time to First Token", cls="metric-section-title"), ttft_lines, cls="metric-section", id="ttft-expanded", style="display: none;"),
                    Div(Div("Time per Output Token", cls="metric-section-title"), tpot_lines, cls="metric-section", id="tpot-expanded", style="display: none;"),
                    Div(Div("Inter-token Latency", cls="metric-section-title"), itl_lines, cls="metric-section", id="itl-expanded", style="display: none;"),
                    Div(Div("End-to-end Latency", cls="metric-section-title"), e2e_lines, cls="metric-section", id="e2e-expanded", style="display: none;"),
                    id="perf-metrics-content",
                    cls="perf-metrics perf-metrics-collapsed"
                ),
                id="perf-summary",
                hx_swap_oob="outerHTML",
                cls="perf-summary"
            )
            # Only return the top-level summary to avoid duplicate cards
            return to_xml(top_summary)

        return ""

    # Consumer: read queue and yield sse_messages
    producer_done = False

    async def sse_yield(html: str):
        yield sse_message(NotStr(html))

    async def finalize_when_done():
        nonlocal producer_done
        # Wait for all tasks to complete regardless of the concurrency setting
        if tasks:
            await asyncio.gather(*tasks)
        await queue.put(("summary", tag, 0, None))
        producer_done = True

    asyncio.create_task(finalize_when_done())

    # Real-time consume queue
    while not (producer_done and queue.empty()):
        try:
            kind, tag_, req_id, payload = await asyncio.wait_for(queue.get(), timeout=0.1)
            html = await gen_html(kind, tag_, req_id, payload)
            if html:
                async for msg in sse_yield(html):
                    yield msg
        except asyncio.TimeoutError:
            continue


# ---------- Routes ----------

RUNS: dict[str, Any] = {}

@rt
def index(req):
    dsets = _datasets_list()
    ds_opts = [Option(d.name, value=d.name, selected=True if i == 0 else False)
               for i, d in enumerate(dsets)] or [Option("none", value="", selected=True)]

    main = Div(
        # Comparison Banner
        Div(
            Div(
                Span("Baseline", cls="baseline-text"),
                Span("vs", cls="vs-text"),
                Span("BrrrLLM", cls="brrrllm-text"),
                cls="comparison-text"
            ),
            cls="comparison-banner"
        ),

        # Header
        Div(
            Div(
                Div(
                    Div(
                        Span("Model:"),
                        Code("openai/gpt-oss-120b"),
                        cls="model-info"
                    ),
                    cls="header-left"
                ),
                Button(
                    "Run Chat",
                    cls="btn btn-success",
                    hx_post="/start_run",
                    hx_target="#results",
                    hx_swap="innerHTML",  # Replace old results instead of appending
                    # Include both controls and prompts so we get loaded_entries
                    **{"hx-include": "#controls, #prompts-container"},
                    style="height: fit-content;"
                ),
                cls="header-content"
            ),
            cls="header"
        ),

        # Control Panel
        Div(
            Form(id="controls")(
                Div(
                    # Left column - Dataset and settings
                    Div(
                        Div(
                            Label("Dataset"),
                            Select(*ds_opts, name="dataset", id="dataset-select",
                                   hx_post="/load_prompts",
                                   hx_target="#prompts-container",
                                   hx_trigger="change",
                                   hx_include="#controls"),
                            cls="form-group"
                        ),
                        Div(
                            Label("Max Output Tokens"),
                            Input(type="number", name="out_tokens", value=500, min=1, max=2000),
                            cls="form-group"
                        ),
                        cls="control-section"
                    ),

                    # Right column - Run mode
                    Div(
                        Div(
                            Label("Run Mode"),
                            Div(
                                Label(
                                    Input(type="radio", name="mode", value="single", checked=True,
                                          hx_post="/load_prompts",
                                          hx_target="#prompts-container",
                                          hx_trigger="change",
                                          hx_include="#controls"),
                                    "Single Request",
                                    cls="radio-label"
                                ),
                                Label(
                                    Input(type="radio", name="mode", value="10x1",
                                          hx_post="/load_prompts",
                                          hx_target="#prompts-container",
                                          hx_trigger="change",
                                          hx_include="#controls"),
                                    "10 Requests (2 Concurrent)",
                                    cls="radio-label"
                                ),
                                Label(
                                    Input(type="radio", name="mode", value="50x10",
                                          hx_post="/load_prompts",
                                          hx_target="#prompts-container",
                                          hx_trigger="change",
                                          hx_include="#controls"),
                                    "50 Requests (10 Concurrent)",
                                    cls="radio-label"
                                ),
                                cls="radio-group"
                            ),
                            cls="form-group"
                        ),
                        cls="control-section"
                    ),
                    cls="control-grid"
                )
            ),
            cls="control-panel"
        ),

        # Results (place above prompts so the run header appears first)
        Div(id="results", cls="results-container"),

        # Prompts Container (auto-load on page load)
        Div(
            id="prompts-container",
            cls="prompts-section",
            hx_post="/load_prompts",
            hx_trigger="load",
            hx_include="#controls"
        ),

        cls="container"
    )

    return Titled("Chat Playground", main)


@rt
def load_prompts(dataset: str = None, mode: str = None):
    """Load prompts and create dynamic UI based on mode selection."""
    # Default values if not provided
    if not dataset:
        dataset = "sharegpt_min1k_50"  # Default dataset
    if not mode:
        mode = "single"  # Default mode

    ds = _dataset_by_name(dataset)
    if not ds:
        return Div(f"Dataset not found: {dataset}", style="color: var(--danger-color);")

    # Parse mode to get number of requests
    total, _ = _parse_mode(mode)

    try:
        entries = _load_dataset_entries(ds, max(total, 50))  # Load up to 50 entries
        if not entries:
            return Div("No entries found in dataset", style="color: var(--danger-color);")

        # Create prompt boxes
        prompt_boxes = []
        for i, entry in enumerate(entries[:total]):
            prompt, response, p_tokens, r_tokens = _extract_prompt_and_response(entry, ds)

            box = Div(
                Div(
                    Div(f"Request #{i+1}", cls="prompt-number"),
                    cls="prompt-header"
                ),
                Div(
                    Div(
                        Div(
                            Span("PROMPT", cls="section-title"),
                            Span(f" - Input: {p_tokens} tokens", id=f"in-tokens-{i}", cls="token-info", style="margin-left: 10px;"),
                            style="display: flex; align-items: center;"
                        ),
                        Textarea(
                            prompt,
                            cls="prompt-textarea",
                            readonly=True,
                            id=f"prompt-{i}",
                            name=f"prompt-{i}"
                        ),
                        cls="prompt-input-section"
                    ),
                    Div(
                        Div("RESPONSE", cls="section-title"),
                        Div(
                            id=f"output-{i}",
                            cls="output-area"
                        ),
                        cls="prompt-output-section"
                    ),
                    cls="prompt-content"
                ),
                cls="prompt-box"
            )
            prompt_boxes.append(box)

        # Performance summary placeholder
        perf_summary = Div(
            Div(
                Div("Performance Summary", cls="perf-title"),
                Button(
                    "Expand All",
                    cls="expand-btn",
                    onclick="togglePerfMetrics()",
                    id="expand-toggle"
                ),
                cls="perf-header"
            ),
            Div(
                Div(
                    "Successful requests:                --\n"
                    "Benchmark duration (s):             --\n"
                    "Total input tokens:                 --\n"
                    "Total generated tokens:             --\n"
                    "Request throughput (req/s):         --\n"
                    "Output token throughput (tok/s):    --",
                    cls="metric-section"
                ),
                Div(
                    Div("Time to First Token", cls="metric-section-title"),
                    "Mean TTFT (ms):      --\n"
                    "Median TTFT (ms):    --\n"
                    "P90 TTFT (ms):       --\n"
                    "P99 TTFT (ms):       --",
                    cls="metric-section",
                    id="ttft-expanded",
                    style="display: none;"
                ),
                Div(
                    Div("Time per Output Token", cls="metric-section-title"),
                    "Mean TPOT (ms):      --\n"
                    "Median TPOT (ms):    --\n"
                    "P90 TPOT (ms):       --\n"
                    "P99 TPOT (ms):       --",
                    cls="metric-section",
                    id="tpot-expanded",
                    style="display: none;"
                ),
                Div(
                    Div("Inter-token Latency", cls="metric-section-title"),
                    "Mean ITL (ms):       --\n"
                    "Median ITL (ms):     --\n"
                    "P90 ITL (ms):        --\n"
                    "P99 ITL (ms):        --",
                    cls="metric-section",
                    id="itl-expanded",
                    style="display: none;"
                ),
                Div(
                    Div("End-to-end Latency", cls="metric-section-title"),
                    "Mean E2EL (ms):      --\n"
                    "Median E2EL (ms):    --\n"
                    "P90 E2EL (ms):       --\n"
                    "P99 E2EL (ms):       --",
                    cls="metric-section",
                    id="e2e-expanded",
                    style="display: none;"
                ),
                id="perf-metrics-content",
                cls="perf-metrics perf-metrics-collapsed"
            ),
            Script("""
                function togglePerfMetrics() {
                    const content = document.getElementById('perf-metrics-content');
                    const button = document.getElementById('expand-toggle');
                    const ttft = document.getElementById('ttft-expanded');
                    const tpot = document.getElementById('tpot-expanded');
                    const itl = document.getElementById('itl-expanded');
                    const e2e = document.getElementById('e2e-expanded');

                    if (content.classList.contains('perf-metrics-collapsed')) {
                        content.classList.remove('perf-metrics-collapsed');
                        content.classList.add('perf-metrics-expanded');
                        button.textContent = 'Collapse';
                        ttft.style.display = 'block';
                        tpot.style.display = 'block';
                        itl.style.display = 'block';
                        e2e.style.display = 'block';
                    } else {
                        content.classList.remove('perf-metrics-expanded');
                        content.classList.add('perf-metrics-collapsed');
                        button.textContent = 'Expand All';
                        ttft.style.display = 'none';
                        tpot.style.display = 'none';
                        itl.style.display = 'none';
                        e2e.style.display = 'none';
                    }
                }
            """),
            cls="perf-summary",
            id="perf-summary"
        )

        return Div(
            perf_summary,
            H3("ShareGPT Dataset", style="color: #1e293b; margin-bottom: 10px; margin-top: 14px;"),
            Div(*prompt_boxes, cls="prompt-boxes-grid"),
            # Store metadata for the run
            Input(type="hidden", name="loaded_entries", value=json.dumps([
                {
                    "prompt": e.get("prompt", ""),
                    "response": e.get("response", ""),
                    "prompt_tokens": e.get("prompt_tokens", 0),
                    "response_tokens": e.get("response_tokens", 0)
                } for e in entries[:total]
            ]))
        )

    except Exception as e:
        return Div(f"Error loading dataset: {str(e)}", style="color: var(--danger-color);")


@rt
def sample(dataset: str):
    """Load dataset info and show sample."""
    if not dataset:
        return Div("Please select a dataset", style="color: var(--danger-color);")

    ds = _dataset_by_name(dataset)
    if not ds:
        return Div("Dataset not found", style="color: var(--danger-color);")

    try:
        entries = _load_dataset_entries(ds, 50)
        if not entries:
            return Div("No entries found in dataset", style="color: var(--danger-color);")

        # Show dataset info
        return Div(
            Div(
                Strong(f"Dataset loaded: "),
                Span(f"{len(entries)} entries"),
                style="color: var(--success-color); margin: 10px 0;"
            ),
            Div(
                f"Sample prompt: \"{entries[0]['prompt'][:100]}...\"" if entries else "",
                style="font-size: 0.85rem; color: var(--text-muted); font-style: italic;"
            ),
            cls="token-counter"
        )
    except Exception as e:
        return Div(f"Error loading dataset: {str(e)}", style="color: var(--danger-color);")


def _parse_mode(mode: str) -> tuple[int, int]:
    m = (mode or "single").lower()
    if m == "single":
        return 1, 1
    if m == "10x1":
        # Run 10 requests at concurrency 2 as requested
        return 10, 2
    if m == "50x10":
        return 50, 10
    return 1, 1


@rt("/start_run")
async def post(dataset: str, out_tokens: int, mode: str, loaded_entries: str = None):
    print(f"start_run called with dataset={dataset}, out_tokens={out_tokens}, mode={mode}")
    rid = uuid.uuid4().hex[:8]
    ttag = f"RUN-{rid}"
    total, conc = _parse_mode(mode)

    # Load dataset
    if not dataset:
        return Div(
            "Please select a dataset first",
            style="padding: 20px; color: var(--danger-color); background: white; border-radius: 8px; margin: 10px 0;"
        )

    ds = _dataset_by_name(dataset)
    if not ds:
        return Div(
            "Dataset not found",
            style="padding: 20px; color: var(--danger-color); background: white; border-radius: 8px; margin: 10px 0;"
        )

    # Prefer the prompts that were already loaded into the page if provided
    try:
        entries: list[dict]
        if loaded_entries:
            try:
                parsed = json.loads(loaded_entries)
                if isinstance(parsed, list) and parsed:
                    entries = parsed[:min(total, 50)]
                else:
                    entries = _load_dataset_entries(ds, min(total, 50))
            except Exception:
                entries = _load_dataset_entries(ds, min(total, 50))
        else:
            entries = _load_dataset_entries(ds, min(total, 50))

        if not entries:
            return Div(
                "No entries found in dataset",
                style="padding: 20px; color: var(--danger-color); background: white; border-radius: 8px; margin: 10px 0;"
            )
    except Exception as e:
        return Div(
            f"Error loading dataset: {str(e)}",
            style="padding: 20px; color: var(--danger-color); background: white; border-radius: 8px; margin: 10px 0;"
        )

    # Clear all previous runs to prevent stale SSE streams
    old_runs = list(RUNS.keys())
    RUNS.clear()

    RUNS[rid] = {
        "dataset_entries": entries,
        "ds_config": ds,
        "out_tokens": int(out_tokens),
        "total": min(total, len(entries)),
        "conc": conc,
        "use_loaded": bool(loaded_entries)
    }
    print(f"Cleared old runs: {old_runs}")
    print(f"Created run {rid} in RUNS. Current runs: {list(RUNS.keys())}")

    # Create run container with a script to clear previous results
    run = Div(
        # Clear old results before adding new one
        Script("""
            // Close any existing SSE connections
            document.querySelectorAll('[sse-connect]').forEach(el => {
                if (el.htmx && el.htmx.sseSource) {
                    el.htmx.sseSource.close();
                }
            });
        """),
        Div(
            Div(
                H3(f"Benchmark Run #{rid}"),
                Span("RUNNING", cls="status-badge status-running", id=f"status-{ttag}"),
                cls="run-left"
            ),
            Div(
                Span(f"📊 {min(total, len(entries))} requests"),
                Span(f"⚡ {conc} concurrent"),
                Span(f"📝 {out_tokens} max tokens"),
                cls="run-meta"
            ),
            cls="run-header"
        ),
        Div(
            Div(id=f"summary-{ttag}"),  # Summary placeholder
            Div(id=f"reqs-{ttag}", cls="request-grid"),  # Requests container
            cls="run-body"
        ),
        # SSE connector
        Div(
            hx_ext="sse",
            sse_connect=f"/stream/{rid}",
            sse_swap="message",
            sse_close="close",  # ensure the SSE connection closes when the server signals 'close'
            hx_swap="none",
        ),
        cls="run-container",
        id=f"run-{rid}"  # Add unique ID for this run container
    )
    return run


shutdown_event = signal_shutdown()


@rt("/stream/{run_id}")
async def get(run_id: str):
    async def empty_gen():
        if False:
            yield

    if shutdown_event.is_set():
        return EventStream(empty_gen())

    ri = RUNS.get(run_id)
    if not ri:
        print(f"Run ID {run_id} not found in RUNS: {list(RUNS.keys())}")
        return EventStream(empty_gen())

    tag = f"RUN-{run_id}"

    async def gen():
        async for msg in run_batch(
            PROV,
            ri["dataset_entries"],
            ri["ds_config"],
            ri["out_tokens"],
            ri["total"],
            ri["conc"],
            tag,
            ri.get("use_loaded", False),
        ):
            yield msg

        # Update status to complete
        yield sse_message(NotStr(to_xml(
            Span("COMPLETE", cls="status-badge status-complete",
                 id=f"status-{tag}", hx_swap_oob="outerHTML")
        )))
        # Tell the htmx SSE extension to close the EventSource and NOT reconnect
        yield NotStr("event: close\ndata: \n\n")

    return EventStream(gen())


# Run the server when executed directly
serve()
