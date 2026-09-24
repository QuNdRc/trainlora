# LoRA Fine-Tuning Pipeline for Apartment Layout Planning

Domain-specific LoRA adapters trained on Google Gemma-4 (4.6B) to control
an apartment layout engine via structured JSON commands — from dataset to
llama.cpp-ready GGUF-LoRA files.

## What this repository is

A complete, self-contained pipeline that takes planning examples, fine-tunes a
**4.6-billion parameter language model** (Gemma-4-E2B-it) on **CPU-only**
hardware, and produces tiny (~112 MB) LoRA adapters ready for llama.cpp
deployment.  The adapters generate layout-fixing commands
(`move_target`, `swap`, `add_door`) in a strict two-part format:
human-readable reasoning followed by machine-parseable JSON.

Three adapters cover the full planning workflow:

| Adapter | Role | Dataset |
|---------|------|---------|
| `expert` | Norm controller — validates and corrects room layouts against standards | `dataset_norm.json` |
| `planner` | Layout generator — proposes room placement from scratch | `dataset_planer.json` |
| `pluginist` | Command executor — translates high-level instructions into engine commands | `dataset_pluginist.json` |

## Architecture

```
dataset.json  ──→  train_lora.py  ──→  *_lora_ready/
  (JSON)             (PEFT/HF)            (safetensors)

  *_lora_ready/  ──→  convert_adapters_to_gguf.py  ──→  *.gguf
  (safetensors)        (llama.cpp subprocess)             (GGUF-LoRA)

  *.gguf  ──→  llama-server --lora *.gguf  ──→  HTTP API
```

- **Model**: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) — text-only backbone (Gemma4ForCausalLM)
- **Training**: PEFT LoRA (r=16, alpha=32) on 7 target modules per layer
- **Hardware**: CPU, 16+ GB RAM, bfloat16 weights (~14 GB)
- **Format**: two-part response — `"Решение: …\n{JSON}"` — loss masked on prompt tokens (-100)

## Quick start

### 1. Environment

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers==5.15.1 peft==0.20.0 datasets safetensors
```

Download Gemma-4 weights (safetensors) to `C:/ai/weight/model.safetensors`.

### 2. Train an adapter

```bash
python train_lora.py --dataset dataset_norm.json --output ./expert_lora_ready --adapter-name expert
```

One command per adapter. Training runs at ~13 seconds per step on
a modern CPU — roughly 1 hour per 50-example dataset × 5 epochs.

### 3. Convert to GGUF-LoRA

```bash
python convert_adapters_to_gguf.py --adapter expert_lora_ready
```

Requires the [llama.cpp](https://github.com/ggml-org/llama.cpp) source tree
with `conversion/` and `gguf-py/` available at `C:/ai/llama-src/llama.cpp-master`.

### 4. Run with llama.cpp

```bash
llama-server.exe -m gemma-4-base.gguf --lora expert_lora.gguf --port 8080
```

Or hot-swap at runtime via `POST /lora-adapters`.

## Files

| File | Purpose |
|------|---------|
| [`train_lora.py`](train_lora.py) | Main LoRA training script — builds model, loads weights, runs PEFT |
| [`convert_datasets.py`](convert_datasets.py) | Preprocessing: splits raw outputs into text + JSON parts |
| [`convert_adapters_to_gguf.py`](convert_adapters_to_gguf.py) | Batch converter: PEFT safetensors → GGUF-LoRA via llama.cpp |
| [`bench_step.py`](bench_step.py) | Micro-benchmark: measures seconds-per-step on real hardware |
| `dataset_*.json` | Pre-converted training datasets |
| `plans/` | Architecture docs, pipeline design, research notes |

## The response format

The model is trained to produce a two-part answer:

```
Решение: Кухня пересекает гостиную в точке (3.2, 1.5), сдвигаю её вправо.
{"commands":[{"action":"move_target","room_id":"r1","new_tx":5.5,"new_ty":4.0}],"explanation":"…"}
```

This design separates the model's chain-of-thought reasoning (for
debugging and human review) from the machine-executable JSON (parsed
by the layout engine).

## License

MIT