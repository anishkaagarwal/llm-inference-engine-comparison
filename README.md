# LLM Inference Engine Comparison: vLLM vs. llama.cpp

A hands-on benchmark comparing two LLM serving stacks — **vLLM** (GPU, FP16, batched serving) and **llama.cpp** (CPU, quantized GGUF) — using Qwen2.5-1.5B-Instruct on a Colab T4 GPU. Also includes a standalone investigation into **KV-cache memory growth**, which explains why vLLM's PagedAttention exists in the first place.

## What's inside

- `llm_inference_comparison.ipynb` — the full notebook: setup, benchmarks, and analysis

## Setup

- **Model:** Qwen2.5-1.5B-Instruct
- **Hardware:** T4 GPU (Google Colab)
- **Stacks compared:**
  - vLLM — FP16, GPU, batched serving
  - llama.cpp — GGUF Q4_K_M, CPU, sequential

## Key findings

**Model size**
| | vLLM (FP16) | llama.cpp (GGUF Q4_K_M) |
|---|---|---|
| Size | ~3.09 GB | ~1.04 GB (~3x smaller) |

**Throughput ≠ latency**
vLLM batched (20 prompts at once) hit **328.54 tok/s** aggregate throughput. Sequential single-request latency ranged **1.88s–7.71s** per request. These are different measurements answering different questions — batching optimizes total system throughput, not the speed of any one request.

**Quantization's memory impact**
Loading the same model in FP16 vs. 4-bit directly (via Transformers/bitsandbytes) showed 4-bit cuts GPU memory to roughly a quarter of FP16 — independent of which inference engine is used downstream.

**KV-cache growth**
Cache memory scales with both sequence length *and* batch size, not just model weight size. In this test (1.5B params, ≤1000 tokens) the cache was tiny (~0.02–0.04 GB), but at scale — 70B models, 32K+ context windows, many concurrent users — KV-cache can balloon past the model weights themselves. This is the exact bottleneck vLLM's **PagedAttention** was designed to solve.

## When to use which

- **vLLM** — high-throughput serving, many concurrent users, GPU available, aggregate speed matters most
- **llama.cpp** — cheap CPU-only or edge deployment, offline/on-device apps, single-user scenarios where footprint and no-GPU-dependency matter more than raw speed

## Note

The notebook title references TensorRT-LLM as a third engine for comparison, but the current notebook only implements and benchmarks vLLM and llama.cpp — TensorRT-LLM is not yet included.

## Requirements

See `requirements.txt`. Note: this notebook was built for Google Colab and uses `google.colab.userdata` to read a Hugging Face token (`HF_TOKEN`) — outside Colab, set `HF_TOKEN` as an environment variable instead.

## License

Add a license of your choice (MIT is a common default for benchmark/research repos like this).
