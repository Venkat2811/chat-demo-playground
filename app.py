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
    hardware: Optional[dict] = None


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
    hardware = data.get("hardware")
    return AppConfig(provider=provider, providers=providers, datasets=datasets, hardware=hardware)


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
# Hardware settings
HW = (APP_CFG.hardware if APP_CFG and APP_CFG.hardware else {})
GPU_COUNT = 1
try:
    GPU_COUNT = int(HW.get("gpu_count", 1))  # type: ignore[arg-type]
except Exception:
    GPU_COUNT = 1

# Cache dataset entries
_dataset_cache: dict[str, list[dict]] = {}

def _datasets_list() -> list[DatasetConfig]:
    dss: list[DatasetConfig] = []

    # Always add ShareGPT datasets
    dss.append(DatasetConfig(
        name="sharegpt_min1k_50",
        path="/Users/venkat/Downloads/sharegpt_min1k_50.jsonl",
        fmt="jsonl",
        prompt_field="prompt",
        response_field="response",
        prompt_tokens_field="prompt_tokens",
        response_tokens_field="response_tokens"
    ))

    dss.append(DatasetConfig(
        name="sharegpt_1k_to_1025_all",
        path="/Users/venkat/Downloads/sharegpt_1k_to_1025_all.jsonl",
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
    # Global script: toggle expand/collapse of performance metrics
    Script(
        """
        window.togglePerfMetrics = function(btn) {
            try {
                var root = null;
                if (btn && btn.closest) root = btn.closest('#perf-summary, .perf-summary');
                if (!root) root = document.getElementById('perf-summary');
                if (!root) return;
                var button = btn || root.querySelector('#expand-toggle');
                var content = root.querySelector('#perf-metrics-content');
                // Find all metric section cards within the metrics rows
                var rows = root.querySelectorAll('.metrics-row');
                var sections = [];
                for (var r=0; r<rows.length; r++) {
                    var cards = rows[r].querySelectorAll('.metric-section-card');
                    for (var c=0; c<cards.length; c++) {
                        sections.push(cards[c]);
                    }
                }
                // Fallback to ID-based selection if no cards found
                if (sections.length === 0) {
                    var ids = ['hw-expanded','req-expanded','user-expanded','pergpu-expanded','totals-expanded','ttft-expanded','tpot-expanded','itl-expanded','e2e-expanded'];
                    for (var i=0; i<ids.length; i++) {
                        var el = document.getElementById(ids[i]);
                        if (el) sections.push(el);
                    }
                }
                var expanded = (root.getAttribute('data-expanded') === '1');

                if (!expanded) {
                    if (content) { content.classList.add('perf-metrics-expanded'); content.classList.remove('perf-metrics-collapsed'); }
                    for (var j=0; j<sections.length; j++) { if (sections[j]) sections[j].style.display = 'block'; }
                    if (button) button.textContent = 'Collapse All';
                    root.setAttribute('data-expanded', '1');
                } else {
                    if (content) { content.classList.remove('perf-metrics-expanded'); content.classList.add('perf-metrics-collapsed'); }
                    for (var k=0; k<sections.length; k++) { if (sections[k]) sections[k].style.display = 'none'; }
                    if (button) button.textContent = 'Expand All';
                    root.setAttribute('data-expanded', '0');
                }
            } catch (e) {
                try { console.error('togglePerfMetrics error:', e); } catch(_){ }
            }
        }

        window.openGroupPopup = function(groupId, title) {
            try {
                var overlay = document.getElementById('popup-overlay');
                var box = document.getElementById('popup-box');
                var titleEl = document.getElementById('popup-title');
                var contentEl = document.getElementById('popup-content');
                var group = document.getElementById(groupId);
                if (!overlay || !box || !contentEl || !group) return;
                var titleNode = group.querySelector('.metric-section-title');
                titleEl.textContent = title || (titleNode ? titleNode.textContent : 'Details');
                contentEl.innerHTML = '';
                var pre = document.createElement('pre');
                var raw = group.textContent || '';
                // Drop the first line if it's the section title
                var lines = raw.split('\\n');
                if (lines.length > 1 && lines[0].trim().length > 0) lines.shift();
                pre.textContent = lines.join('\\n').trim();
                contentEl.appendChild(pre);
                overlay.style.display = 'flex';
                document.body.style.overflow = 'hidden';
            } catch (e) {
                try { console.error('openGroupPopup error:', e); } catch(_){ }
            }
        }

        window.closeGroupPopup = function() {
            var overlay = document.getElementById('popup-overlay');
            if (overlay) overlay.style.display = 'none';
            document.body.style.overflow = '';
        }
        """
    ),
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
            --panel-h: 300px; /* unified prompt/response panel height */
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

        .request-content { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; align-items: start; }

        .prompt-section, .response-section { padding: 15px; overflow: hidden; position: relative; display: flex; flex-direction: column; }

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

        .prompt-textarea { width: 100%; height: var(--panel-h); max-height: var(--panel-h); padding: 10px; border: 1px solid var(--border-color); border-radius: 6px; font-family: monospace; font-size: 0.85rem; resize: vertical; background: #f8fafc; color: #1e293b !important; font-weight: 500; overflow: auto; }

        .output-area { width: 100%; height: var(--panel-h); max-height: var(--panel-h); padding: 10px; border: 1px solid var(--border-color); border-radius: 6px; font-family: monospace; font-size: 0.85rem; background: white; overflow: auto; }
        .response-section .content-text { height: var(--panel-h); max-height: var(--panel-h); overflow: auto; }

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
            white-space: normal;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        /* Row layout for metric sections */
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            width: 100%;
        }

        @media (min-width: 768px) {
            .metrics-row:first-of-type {
                grid-template-columns: repeat(4, 1fr);
            }
            .metrics-row:nth-of-type(2) {
                grid-template-columns: repeat(5, 1fr);
            }
        }

        @media (min-width: 1400px) {
            .metrics-row {
                gap: 16px;
            }
        }

        /* Override for tile grids inside perf metrics */
        .perf-metrics .stats-grid {
            white-space: normal;
            font-family: inherit;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
        }

        .metric-section {
            white-space: pre;
            margin: 0;
            padding: 10px;
            background: #f8fafc;
            border-radius: 6px;
        }

        .metric-section-card {
            white-space: pre;
            padding: 12px;
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease;
            font-size: 0.85rem;
            min-height: 100px;
        }

        .metric-section-card:hover {
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }

        .perf-metrics-collapsed {
            max-height: none;
            overflow: hidden;
        }

        .perf-metrics-expanded {
            max-height: none;
        }

        .stat-item { cursor: pointer; }

        /* Popup styles */
        .popup-overlay {
            position: fixed;
            inset: 0;
            display: none;
            align-items: center;
            justify-content: center;
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(4px);
            z-index: 1000;
            padding: 20px;
            animation: fadeIn 0.2s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .popup-box {
            width: min(720px, 95vw);
            max-height: 85vh;
            overflow: auto;
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 12px;
            border: 1px solid rgba(226, 232, 240, 0.8);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            animation: slideUp 0.3s ease;
        }

        .popup-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
            border-bottom: 1px solid rgba(226, 232, 240, 0.8);
            background: linear-gradient(180deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.9) 100%);
            border-radius: 12px 12px 0 0;
        }

        .popup-title {
            font-weight: 700;
            color: var(--primary-color);
            font-size: 1.1rem;
        }

        .popup-body {
            padding: 20px;
            font-family: 'Monaco', 'Courier New', monospace;
            background: white;
            border-radius: 0 0 12px 12px;
        }

        .popup-body pre {
            background: #f8fafc;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .popup-close {
            background: transparent;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #64748b;
            transition: all 0.2s ease;
            padding: 0 4px;
        }

        .popup-close:hover {
            color: var(--danger-color);
            transform: rotate(90deg);
        }
        
        .metric-section-title {
            color: var(--primary-color);
            font-weight: 700;
            margin: 0 0 8px 0;
            font-size: 0.95rem;
        }

        .metric-section-title.clickable {
            cursor: pointer;
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            transition: all 0.2s ease;
        }

        .metric-section-title.clickable:hover {
            background: var(--primary-color);
            color: white;
            transform: translateX(2px);
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
    # Force chat completions path even if 'responses' is configured, to avoid
    # server-side tool pipelines on some vLLM builds.
    if "chat" in kind or "responses" in kind:
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
    usage_seen: bool = False

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
                usage_seen = True
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
            if metrics.output_tokens > 1 and metrics.first_token_time is not None:
                # Live TPOT calculation matching vLLM approach
                time_since_first = (t - metrics.first_token_time) * 1000.0  # Convert to ms
                metrics.tpot_ms = time_since_first / (metrics.output_tokens - 1)
            else:
                metrics.tpot_ms = 0.0

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

    # Calculate final TPOT following vLLM benchmark approach
    if metrics.output_tokens > 1 and metrics.ttft_ms > 0:
        # TPOT = (total_latency - ttft) / (output_tokens - 1)
        latency_minus_ttft_ms = metrics.e2e_ms - metrics.ttft_ms
        metrics.tpot_ms = latency_minus_ttft_ms / (metrics.output_tokens - 1)
    else:
        metrics.tpot_ms = 0.0

    # Strict check only if provider usage was observed
    if usage_seen and metrics.output_tokens != max_tokens:
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

    # For large batches (203+ requests), update UI less frequently
    # Also check if this is the sharegpt_1k_to_1025_all dataset
    is_large_batch = total_reqs >= 203 or ds_config.name == "sharegpt_1k_to_1025_all"

    async def enqueue(kind: str, tag: str, req_id: int, payload: Any):
        await queue.put((kind, tag, req_id, payload))

    # True concurrent execution
    batch_start = now()

    async def task_fn(i: int):
        # For 1421 requests mode, wrap around the 203 entries
        if total_reqs == 1421:
            # Use modulo to cycle through the 203 entries
            entry_idx = (i - 1) % len(dataset_entries)
        else:
            entry_idx = min(i - 1, len(dataset_entries) - 1)

        entry = dataset_entries[entry_idx]
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
    # Handle special cases for dataset
    if total_reqs == 203:
        num_requests = len(dataset_entries)
    elif total_reqs == 1421:
        # For 1421 requests, we'll cycle through the 203 entries 7 times
        num_requests = 1421
    else:
        num_requests = min(total_reqs, len(dataset_entries))

    print(f"Creating {num_requests} tasks (total_reqs={total_reqs}, dataset_entries={len(dataset_entries)})")
    for i in range(1, num_requests + 1):
        tasks.append(asyncio.create_task(limited_task(i)))

    completed = 0
    completed_lock = asyncio.Lock()

    async def gen_html(kind: str, tag: str, req_id: int, payload: Any) -> str:
        nonlocal completed

        if kind == "create_box":
            # Skip creating individual boxes for large batches to avoid UI overwhelm
            if is_large_batch:
                return ""

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
            # Skip token updates for large batches to avoid UI overwhelm
            if is_large_batch:
                return ""

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
                # For large batches, provide progress updates every 10 completions
                if is_large_batch and completed % 10 == 0:
                    progress_html = f'<span id="status-{tag}" hx-swap-oob="outerHTML">'
                    progress_html += f'<span class="status-badge status-running">RUNNING ({completed}/{total_reqs})</span>'
                    progress_html += '</span>'
                    return progress_html
            return ""

        elif kind == "summary":
            # Generate comprehensive summary statistics
            async with metrics_lock:
                ttfts = [m.ttft_ms for m in all_metrics if m.ttft_ms > 0]
                # Only include TPOT for requests with >1 output tokens (per vLLM)
                tpots = [m.tpot_ms for m in all_metrics if m.output_tokens > 1 and m.tpot_ms > 0]
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
            per_user_output_throughput = (output_token_throughput / max(concurrency, 1)) if duration > 0 else 0
            per_gpu_output_throughput = (output_token_throughput / max(GPU_COUNT, 1)) if duration > 0 else 0
            sum_e2e_ms = sum(e2es) if e2es else 0.0
            avg_req_latency_ms = (sum_e2e_ms / len(e2es)) if e2es else 0.0
            mean_tpot_ms = (sum(tpots)/len(tpots)) if tpots else 0.0
            per_user_output_speed = (1000.0 / mean_tpot_ms) if mean_tpot_ms > 0 else 0.0

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
                f"Output token throughput (tok/s):    {output_token_throughput:.1f}\n"
                f"Per-user output throughput (tok/s/user): {per_user_output_throughput:.2f}\n"
                f"Per-GPU output throughput (tok/s/gpu):   {per_gpu_output_throughput:.2f}\n"
                f"Total token throughput (tok/s):     {total_token_throughput:.1f}\n"
                f"Total latency (ms):                 {sum_e2e_ms:.1f}\n"
                f"Average request latency (ms):       {avg_req_latency_ms:.1f}\n"
                f"Per-user output speed [1/TPOT] (tok/s/user): {per_user_output_speed:.4f}"
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

            # Build a compact stats grid for the collapsed view
            collapsed_stats = Div(
                # Put Hardware first
                Div(
                    Div("Hardware", cls="stat-label"),
                    Div(f"{HW.get('gpu_model','GPU')} × {GPU_COUNT}", cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('hw-expanded','Hardware')"
                ),
                Div(
                    Div("Successful Requests", cls="stat-label"),
                    Div(str(completed), cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('req-expanded','Requests')"
                ),
                Div(
                    Div("Duration", cls="stat-label"),
                    Div(f"{duration:.2f}s", cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('req-expanded','Requests')"
                ),
                Div(
                    Div("Total Input Tokens", cls="stat-label"),
                    Div(str(total_input_tokens), cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('req-expanded','Requests')"
                ),
                Div(
                    Div("Total Generated Tokens", cls="stat-label"),
                    Div(str(total_output_tokens), cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('req-expanded','Requests')"
                ),
                Div(
                    Div("Concurrency", cls="stat-label"),
                    Div(str(concurrency), cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('req-expanded','Requests')"
                ),
                Div(
                    Div("Req Throughput", cls="stat-label"),
                    Div(f"{throughput:.2f} req/s", cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('req-expanded','Requests')"
                ),
                Div(
                    Div("Output Tok/s", cls="stat-label"),
                    Div(f"{output_token_throughput:.1f}", cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('totals-expanded','Totals')"
                ),
                Div(
                    Div("Tok/s per User", cls="stat-label"),
                    Div(f"{per_user_output_throughput:.2f}", cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('user-expanded','Per User')"
                ),
                Div(
                    Div("Tok/s per GPU", cls="stat-label"),
                    Div(f"{per_gpu_output_throughput:.2f}", cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('pergpu-expanded','Per GPU')"
                ),
                Div(
                    Div("Total Tok/s", cls="stat-label"),
                    Div(f"{total_token_throughput:.1f}", cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('totals-expanded','Totals')"
                ),
                Div(
                    Div("Avg Req Latency", cls="stat-label"),
                    Div(f"{avg_req_latency_ms:.1f}ms", cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('totals-expanded','Totals')"
                ),
                Div(
                    Div("1/TPOT per User", cls="stat-label"),
                    Div(f"{per_user_output_speed:.2f}", cls="stat-value"),
                    cls="stat-item",
                    onclick="openGroupPopup('user-expanded','Per User')"
                ),
                cls="stats-grid"
            )

            # Build expanded group cards
            hw_lines = (
                f"Model:  {HW.get('gpu_model','GPU')}\n"
                f"Count:  {GPU_COUNT}"
            )
            req_lines = (
                f"Total requests:       {total_reqs}\n"
                f"Successful:           {completed}\n"
                f"Total input tokens:    {total_input_tokens}\n"
                f"Total output tokens:   {total_output_tokens}\n"
                f"Concurrency:          {concurrency}\n"
                f"Req throughput (req/s): {throughput:.2f}"
            )
            # New Throughput section with all token metrics
            duration_s = duration if duration > 0 else 1  # duration is already in seconds
            throughput_lines = (
                f"Input tok/s:          {(total_input_tokens / duration_s):.1f}\n"
                f"Output tok/s:         {output_token_throughput:.1f}\n"
                f"Total tok/s:          {total_token_throughput:.1f}\n"
                f"Output tok/s/user:   {per_user_output_throughput:.2f}\n"
                f"Output tok/s/gpu:    {per_gpu_output_throughput:.2f}\n"
                f"Req throughput:       {throughput:.2f} req/s"
            )

            # Per User section
            per_user_lines = (
                f"Mean TTFT (ms):       {(sum(ttfts)/len(ttfts) if ttfts else 0):.0f}\n"
                f"Mean TPOT (ms):       {(sum(tpots)/len(tpots) if tpots else 0):.1f}\n"
                f"Mean ITL (ms):        {(sum(all_itls)/len(all_itls) if all_itls else 0):.1f}\n"
                f"Mean E2EL (ms):       {(sum(e2es)/len(e2es) if e2es else 0):.0f}"
            )

            totals_lines = (
                f"Total latency (ms):   {sum_e2e_ms:.1f}\n"
                f"Avg req latency (ms): {avg_req_latency_ms:.1f}"
            )

            top_summary = Div(
                Div(
                    Div("Performance Summary", cls="perf-title"),
                    Button(
                        "Expand All",
                        cls="expand-btn",
                        onclick="togglePerfMetrics(this)",
                        id="expand-toggle",
                        type="button"
                    ),
                    cls="perf-header"
                ),
                Div(
                    # Main collapsed summary at the top (full width)
                    Div(collapsed_stats, cls="metric-section full-span"),

                    # First row: Hardware, Throughput, Requests, Per User/GPU
                    Div(
                        Div(
                            Div("Hardware", cls="metric-section-title clickable", onclick="openGroupPopup('hw-expanded','Hardware')"),
                            hw_lines, cls="metric-section-card", id="hw-expanded", style="display: none;"
                        ),
                        Div(
                            Div("Throughput", cls="metric-section-title clickable", onclick="openGroupPopup('throughput-expanded','Throughput')"),
                            throughput_lines, cls="metric-section-card", id="throughput-expanded", style="display: none;"
                        ),
                        Div(
                            Div("Requests", cls="metric-section-title clickable", onclick="openGroupPopup('req-expanded','Requests')"),
                            req_lines, cls="metric-section-card", id="req-expanded", style="display: none;"
                        ),
                        Div(
                            Div("Per User", cls="metric-section-title clickable", onclick="openGroupPopup('user-expanded','Per User')"),
                            per_user_lines, cls="metric-section-card", id="user-expanded", style="display: none;"
                        ),
                        cls="metrics-row"
                    ),

                    # Second row: Totals and timing metrics
                    Div(
                        Div(
                            Div("Totals", cls="metric-section-title clickable", onclick="openGroupPopup('totals-expanded','Totals')"),
                            totals_lines, cls="metric-section-card", id="totals-expanded", style="display: none;"
                        ),
                        Div(
                            Div("Time to First Token", cls="metric-section-title clickable", onclick="openGroupPopup('ttft-expanded','Time to First Token')"),
                            ttft_lines, cls="metric-section-card", id="ttft-expanded", style="display: none;"
                        ),
                        Div(
                            Div("Time per Output Token", cls="metric-section-title clickable", onclick="openGroupPopup('tpot-expanded','Time per Output Token')"),
                            tpot_lines, cls="metric-section-card", id="tpot-expanded", style="display: none;"
                        ),
                        Div(
                            Div("Inter-token Latency", cls="metric-section-title clickable", onclick="openGroupPopup('itl-expanded','Inter-token Latency')"),
                            itl_lines, cls="metric-section-card", id="itl-expanded", style="display: none;"
                        ),
                        Div(
                            Div("End-to-end Latency", cls="metric-section-title clickable", onclick="openGroupPopup('e2e-expanded','End-to-end Latency')"),
                            e2e_lines, cls="metric-section-card", id="e2e-expanded", style="display: none;"
                        ),
                        cls="metrics-row"
                    ),
                    id="perf-metrics-content",
                    cls="perf-metrics perf-metrics-collapsed"
                ),
                id="perf-summary",
                **{"data-expanded": "0"},
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
                                   hx_include="#controls",
                                   **{"hx-on::after-request": "htmx.trigger(htmx.find('#run-mode-options'), 'reload-modes')"}),
                            cls="form-group"
                        ),
                        # Hidden input for out_tokens with default value
                        Input(type="hidden", name="out_tokens", value=500),
                        cls="control-section"
                    ),

                    # Right column - Run mode (will be dynamically loaded)
                    Div(
                        Div(
                            Label("Run Mode"),
                            Div(
                                id="run-mode-options",
                                cls="radio-group",
                                hx_post="/load_run_modes",
                                hx_trigger="load, reload-modes",
                                hx_include="#dataset-select"
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

        # Popup overlay for group details
        Div(
            Div(
                Div(
                    Span("Details", id="popup-title", cls="popup-title"),
                    Button("×", cls="popup-close", onclick="closeGroupPopup()")
                , cls="popup-header"),
                Div(id="popup-content", cls="popup-body"),
                cls="popup-box", id="popup-box"
            ),
            id="popup-overlay", cls="popup-overlay"
        ),

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

    # Parse mode to get number of requests and concurrency
    total, conc = _parse_mode(mode)

    try:
        # For the new dataset with 203 entries, always load all of them
        # This ensures all prompts are available regardless of the selected mode
        if dataset == "sharegpt_1k_to_1025_all":
            max_load = 203  # Always load all 203 entries for this dataset
        else:
            max_load = max(total, 50)
        entries = _load_dataset_entries(ds, max_load)
        if not entries:
            return Div("No entries found in dataset", style="color: var(--danger-color);")

        # Create prompt boxes
        prompt_boxes = []
        # For the sharegpt_1k_to_1025_all dataset, never show individual request/response boxes
        if dataset == "sharegpt_1k_to_1025_all":
            display_count = 0
            # Add a notice instead
            if total == 1421:
                message = f"⚡ {total} requests mode - Running 203 dataset entries × 7 iterations"
            else:
                message = f"⚡ {total} requests mode - Running all 203 dataset entries"

            prompt_boxes.append(
                Div(
                    message,
                    Div(
                        f"Concurrency: {conc if conc else 'N/A'}",
                        style="margin-top: 10px; font-weight: bold;"
                    ),
                    cls="info-message",
                    style="padding: 20px; background: var(--info-bg, #e8f4f8); color: var(--info-color, #0066cc); border-radius: 8px; margin: 20px 0;"
                )
            )
        else:
            # For other datasets, show prompt boxes as usual
            display_count = total

        for i, entry in enumerate(entries[:display_count]):
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
                    onclick="togglePerfMetrics(this)",
                    id="expand-toggle",
                    type="button",
                ),
                cls="perf-header",
            ),
            Div(
                Div(
                    # Hardware first in placeholder too (clickable)
                    Div(Div("HARDWARE", cls="stat-label"), Div(f"{HW.get('gpu_model','GPU')} × {GPU_COUNT}", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('hw-expanded','Hardware')"),
                    Div(Div("SUCCESSFUL REQUESTS", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('req-expanded','Requests')"),
                    Div(Div("DURATION", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('req-expanded','Requests')"),
                    Div(Div("TOTAL INPUT TOKENS", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('req-expanded','Requests')"),
                    Div(Div("TOTAL GENERATED TOKENS", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('req-expanded','Requests')"),
                    Div(Div("CONCURRENCY", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('req-expanded','Requests')"),
                    Div(Div("REQ THROUGHPUT", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('req-expanded','Requests')"),
                    Div(Div("OUTPUT TOK/S", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('totals-expanded','Totals')"),
                    Div(Div("TOK/S PER USER", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('user-expanded','Per User')"),
                    Div(Div("TOK/S PER GPU", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('pergpu-expanded','Per GPU')"),
                    Div(Div("TOTAL TOK/S", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('totals-expanded','Totals')"),
                    Div(Div("AVG REQ LATENCY", cls="stat-label"), Div("--", cls="stat-value"), cls="stat-item", onclick="openGroupPopup('totals-expanded','Totals')"),
                    cls="stats-grid",
                ),
                cls="metric-section full-span",
            ),
            Div(
                Div(
                    Div("Time to First Token", cls="metric-section-title clickable", onclick="openGroupPopup('ttft-expanded','Time to First Token')"),
                    "Mean TTFT (ms):      --\n"
                    "Median TTFT (ms):    --\n"
                    "P90 TTFT (ms):       --\n"
                    "P99 TTFT (ms):       --",
                    cls="metric-section",
                    id="ttft-expanded",
                    style="display: none;",
                ),
                Div(
                    Div("Time per Output Token", cls="metric-section-title clickable", onclick="openGroupPopup('tpot-expanded','Time per Output Token')"),
                    "Mean TPOT (ms):      --\n"
                    "Median TPOT (ms):    --\n"
                    "P90 TPOT (ms):       --\n"
                    "P99 TPOT (ms):       --",
                    cls="metric-section",
                    id="tpot-expanded",
                    style="display: none;",
                ),
                Div(
                    Div("Inter-token Latency", cls="metric-section-title clickable", onclick="openGroupPopup('itl-expanded','Inter-token Latency')"),
                    "Mean ITL (ms):       --\n"
                    "Median ITL (ms):     --\n"
                    "P90 ITL (ms):        --\n"
                    "P99 ITL (ms):        --",
                    cls="metric-section",
                    id="itl-expanded",
                    style="display: none;",
                ),
                Div(
                    Div("End-to-end Latency", cls="metric-section-title clickable", onclick="openGroupPopup('e2e-expanded','End-to-end Latency')"),
                    "Mean E2EL (ms):      --\n"
                    "Median E2EL (ms):    --\n"
                    "P90 E2EL (ms):       --\n"
                    "P99 E2EL (ms):       --",
                    cls="metric-section",
                    id="e2e-expanded",
                    style="display: none;",
                ),
                id="perf-metrics-content",
                cls="perf-metrics perf-metrics-collapsed",
            ),
            cls="perf-summary",
            id="perf-summary",
            **{"data-expanded": "0"},
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
def load_run_modes(dataset: str = None):
    """Load run mode options based on the selected dataset."""
    if not dataset:
        dataset = "sharegpt_min1k_50"  # Default dataset

    # Special handling for the new dataset
    if dataset == "sharegpt_1k_to_1025_all":
        return Div(
            Label(
                Input(type="radio", name="mode", value="203x40", checked=True,
                      hx_post="/load_prompts",
                      hx_target="#prompts-container",
                      hx_trigger="change",
                      hx_include="#controls"),
                "All 203 Requests (40 Concurrent)",
                cls="radio-label"
            ),
            Label(
                Input(type="radio", name="mode", value="203x80",
                      hx_post="/load_prompts",
                      hx_target="#prompts-container",
                      hx_trigger="change",
                      hx_include="#controls"),
                "All 203 Requests (80 Concurrent)",
                cls="radio-label"
            ),
            Label(
                Input(type="radio", name="mode", value="203x120",
                      hx_post="/load_prompts",
                      hx_target="#prompts-container",
                      hx_trigger="change",
                      hx_include="#controls"),
                "All 203 Requests (120 Concurrent)",
                cls="radio-label"
            ),
            Label(
                Input(type="radio", name="mode", value="1421x40",
                      hx_post="/load_prompts",
                      hx_target="#prompts-container",
                      hx_trigger="change",
                      hx_include="#controls"),
                "1421 Requests - 203×7 (40 Concurrent)",
                cls="radio-label"
            ),
            Label(
                Input(type="radio", name="mode", value="1421x80",
                      hx_post="/load_prompts",
                      hx_target="#prompts-container",
                      hx_trigger="change",
                      hx_include="#controls"),
                "1421 Requests - 203×7 (80 Concurrent)",
                cls="radio-label"
            ),
            Label(
                Input(type="radio", name="mode", value="1421x120",
                      hx_post="/load_prompts",
                      hx_target="#prompts-container",
                      hx_trigger="change",
                      hx_include="#controls"),
                "1421 Requests - 203×7 (120 Concurrent)",
                cls="radio-label"
            ),
            cls="radio-group"
        )
    else:
        # Default options for other datasets
        return Div(
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
        )

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
    # New modes for the sharegpt_1k_to_1025_all dataset
    if m == "203x40":
        return 203, 40
    if m == "203x80":
        return 203, 80
    if m == "203x120":
        return 203, 120
    # 1421 requests (203 * 7) modes
    if m == "1421x40":
        return 1421, 40
    if m == "1421x80":
        return 1421, 80
    if m == "1421x120":
        return 1421, 120
    return 1, 1


@rt("/start_run")
async def post(dataset: str, out_tokens: int, mode: str, loaded_entries: str = None):
    print(f"start_run called with dataset={dataset}, out_tokens={out_tokens}, mode={mode}")
    rid = uuid.uuid4().hex[:8]
    ttag = f"RUN-{rid}"
    total, conc = _parse_mode(mode)
    print(f"Parsed mode '{mode}' to total={total}, conc={conc}")

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
                    # For sharegpt_1k_to_1025_all, use all loaded entries
                    if dataset == "sharegpt_1k_to_1025_all":
                        entries = parsed  # Use all 203 entries
                    else:
                        entries = parsed[:total]  # Use requested entries for other datasets
                else:
                    # Load all 203 for the special dataset, otherwise load requested amount
                    max_load = 203 if dataset == "sharegpt_1k_to_1025_all" else total
                    entries = _load_dataset_entries(ds, max_load)
            except Exception:
                # Load all 203 for the special dataset, otherwise load requested amount
                max_load = 203 if dataset == "sharegpt_1k_to_1025_all" else total
                entries = _load_dataset_entries(ds, max_load)
        else:
            # Load all 203 for the special dataset, otherwise load requested amount
            max_load = 203 if dataset == "sharegpt_1k_to_1025_all" else total
            entries = _load_dataset_entries(ds, max_load)

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
        "total": total,
        "conc": conc,
        "use_loaded": bool(loaded_entries)
    }
    print(f"Cleared old runs: {old_runs}")
    print(f"Created run {rid} in RUNS. Current runs: {list(RUNS.keys())}")
    print(f"Run {rid}: total={total}, conc={conc}, entries={len(entries)}")

    # Create run container with a script to clear previous results
    run = Div(
        # Clear old results before adding new one
        Script("""
            // Close any existing SSE connections (ES5)
            (function(){
                var list = document.querySelectorAll('[sse-connect]');
                for (var i=0; i<list.length; i++) {
                    var el = list[i];
                    if (el.htmx && el.htmx.sseSource) {
                        try { el.htmx.sseSource.close(); } catch(_){ }
                    }
                }
            })();
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
