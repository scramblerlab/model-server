# aimodel — Unified Local LLM Server

Manages a single Ollama instance tuned for Apple Silicon (M4 Pro, 64 GB) and routes requests from multiple apps through a stats-logging proxy, with a dedicated port per app.

```
generative-radio backend ──→ :11430 ─╮
                                       aimodel proxy → Ollama :11434
logger backend           ──→ :11431 ─╯
```

## Quick Start

```bash
cd aimodel
./start.sh                        # uses model from .env (default: qwen3.5:4b)
./start.sh --model gemma4:12b     # override model at launch
```

Then start each app — they connect to the proxy, not directly to Ollama:

```bash
cd ../generative-radio && ./scripts/start.sh
cd ../logger && ./start_local.sh
```

Stop everything:

```bash
./stop.sh
```

## Configuration

Edit `aimodel/.env` before first run:

```
OLLAMA_MODEL=qwen3.5:4b          # model to load at startup
AIMODEL_OLLAMA_PORT=11434        # Ollama internal port
AIMODEL_PROXY_PORT_RADIO=11430   # generative-radio connects here
AIMODEL_PROXY_PORT_LOGGER=11431  # logger connects here
```

## Console Output

The proxy prints one line per completed LLM response:

```
2026-06-10 14:23:01  [generative-radio]  in:  847 tok @ 38.5 tok/s   out: 312 tok @ 44.2 tok/s
2026-06-10 14:23:45  [logger          ]  in:  523 tok @ 41.2 tok/s   out:  89 tok @ 48.1 tok/s
```

- **in tok/s** — prompt processing speed; a near-infinite value means Ollama reused a cached prefix
- **out tok/s** — generation speed; use this to compare models

## Performance Settings (M4 Pro, 64 GB)

| Variable | Value | Rationale |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | `4` | 4 concurrent KV slots fit easily within 64 GB |
| `OLLAMA_FLASH_ATTENTION` | `1` | Cuts KV cache memory 30–50 %; no quality cost |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Halves KV memory vs f16; perplexity impact is undetectable in chat |
| `OLLAMA_KEEP_ALIVE` | `-1` | Model stays loaded permanently |
| `OLLAMA_CONTEXT_LENGTH` | `8192` | Speed/memory sweet spot for chat workloads (`OLLAMA_NUM_CTX` is not a real variable — Ollama ignores it and loads the model at its native max context) |

> **Important:** these settings only apply to the Ollama instance `start.sh`
> launches. The Ollama **menu-bar app** (launch-at-login) starts its own server
> with defaults (flash attention off, f16 KV, 5-min keep-alive, native max
> context). `start.sh` detects and restarts such an instance, but to avoid the
> churn disable *Start at login* in the Ollama app settings.

## KV Cache Type Reference

| Type | Memory | Quality impact |
|------|--------|---------------|
| `f16` (default) | 1× baseline | None |
| `q8_0` ← recommended | ~½ of f16 | Negligible (+0.002–0.05 perplexity) |
| `q4_0` | ~¼ of f16 | Small (~7.6% perplexity increase; may affect long contexts) |

Change `OLLAMA_KV_CACHE_TYPE` in `start.sh` to switch.

## RAM Budget (Qwen3.5-4B, 4 parallel slots, q8_0, 8192 ctx)

```
KV cache formula:
  bytes/token = 2 × layers × kv_heads × head_dim × 2 bytes (f16)
  Qwen3.5-4B: 2 × 36 × 8 × 128 × 2 = 144 KB/token
  per slot (8192 ctx, q8_0): 144 KB × 8192 × 0.5 ≈ 576 MB
```

| Component | Memory |
|---|---|
| Model weights (Q4_K_M) | ~2.3 GB |
| KV cache: 4 × 576 MB (q8_0) | ~2.3 GB |
| Ollama overhead | ~0.5 GB |
| **Ollama total** | **~5.1 GB** |
| ACE-Step XL turbo DiT | ~9.0 GB |
| ACE-Step 1.7B LM (MLX fp16) | ~3.4 GB |
| ACE-Step VAE + embedding + overhead | ~2.7 GB |
| **ACE-Step peak** | **~15–16 GB** |
| System / other processes | ~4 GB |
| **Grand total** | **~24–25 GB** |
| **Headroom on 64 GB** | **~39 GB free** |

For a 12B model (Q4_K_M, ~7.5 GB): Ollama total ≈ 10.9 GB → grand total ≈ 30–32 GB, still well within budget.

## macOS GPU Memory Ceiling

macOS caps Metal GPU memory at ~75% of RAM (~48 GB on 64 GB). The start script shows the current value. To raise it to 90% (~58 GB) for sessions with heavy concurrent load:

```bash
sudo sysctl iogpu.wired_limit_mb=58982
```

Resets on reboot. Run once per session if you observe memory pressure.

## Adding a New App

1. Add a port to `aimodel/.env`:
   ```
   AIMODEL_PROXY_PORT_MYAPP=11432
   ```
2. Add to `APP_PORTS` in `proxy/server.py`:
   ```python
   APP_PORTS = {
       11430: "generative-radio",
       11431: "logger",
       11432: "my-new-app",   # add this line
   }
   ```
3. Set the new app's Ollama base URL to `http://localhost:11432`.

No Ollama or start.sh changes needed.

## Alternatives to Ollama

| Runtime | Concurrency | Apple Silicon perf | Model format | Complexity |
|---------|------------|-------------------|--------------|-----------|
| **Ollama** ← current | Queue + prefix cache | ~40–58 tok/s | GGUF (wide choice) | Simple |
| llama-server | Basic queue | ~60–80 tok/s | GGUF | Moderate |
| mlx-lm serve | Single-request only | ~80–120 tok/s | MLX format | Simple |
| vllm-mlx | Continuous batching (3.4× at 5 req) | ~42 tok/s / >400 tok/s aggregate | MLX format | Moderate |

**Recommendation:** Ollama is the right choice for 2 apps with 2–4 concurrent requests. `OLLAMA_NUM_PARALLEL=4` handles this load without issue.

**Migrate to vllm-mlx** when: concurrent load grows beyond 5–6 overlapping requests. The proxy's backend URL (`AIMODEL_OLLAMA_PORT`) is a one-line change to point at a different server.
