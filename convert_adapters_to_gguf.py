"""
Convert trained PEFT adapters (safetensors) to GGUF-LoRA format for llama.cpp.

Requires the llama.cpp source tree with the conversion/ and gguf-py/ packages
available. The llama.cpp convert_lora_to_gguf.py tool is invoked as a subprocess
with the adapter path and the HuggingFace model id for configuration lookup.

Usage:
    python convert_adapters_to_gguf.py --adapter expert_lora_ready
    python convert_adapters_to_gguf.py                          # all adapters
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

LLAMA_SRC = r"C:/ai/llama-src/llama.cpp-master"
CONVERTER = os.path.join(LLAMA_SRC, "convert_lora_to_gguf.py")

BASE_MODEL_ID = "google/gemma-4-E2B-it"

ADAPTERS: dict[str, str] = {
    "expert_lora_ready": "expert_lora",
    "planner_lora_ready": "planner_lora",
    "pluginist_lora_ready": "pluginist_lora",
}


def convert_one(adapter_dir: str, out_name: str, base: str) -> bool:
    adapter_abs = os.path.abspath(adapter_dir)
    if not os.path.isfile(CONVERTER):
        print(f"[x] Converter not found: {CONVERTER}")
        return False
    if not os.path.isdir(adapter_abs):
        print(f"[x] Adapter directory not found: {adapter_abs}")
        return False

    out_file = os.path.join(LLAMA_SRC, out_name + ".gguf")
    cmd = [
        sys.executable, CONVERTER,
        adapter_abs,
        "--outfile", out_file,
        "--base-model-id", base,
    ]

    print("=" * 70)
    print(f"  Converting: {adapter_dir}")
    print(f"  Output:     {out_file}")
    print("=" * 70)

    proc = subprocess.run(cmd, cwd=LLAMA_SRC)
    if proc.returncode == 0:
        print(f"  [OK] {adapter_dir} -> {out_file}")
        return True
    print(f"  [x] Conversion failed for {adapter_dir}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", help="Single adapter directory name")
    parser.add_argument("--base", default=BASE_MODEL_ID)
    args = parser.parse_args()

    if not os.path.isdir(LLAMA_SRC):
        print(f"[x] llama.cpp sources not found: {LLAMA_SRC}")
        sys.exit(1)

    targets = [args.adapter] if args.adapter else list(ADAPTERS)
    ok = True

    for name in targets:
        if name not in ADAPTERS:
            print(f"[x] Unknown adapter: {name}")
            sys.exit(1)
        ok = convert_one(name, ADAPTERS[name], args.base) and ok

    print("=" * 70)
    print("  Done. GGUF-LoRA adapters saved to:", LLAMA_SRC)
    for name in targets:
        print(f"    - {os.path.join(LLAMA_SRC, ADAPTERS[name] + '.gguf')}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()