"""
Convert raw planning datasets into the two-part format expected by train_lora.py.

Each source file is a JSON array where the "output" field is a combined string:
    "Human-readable text\\n{\"commands\":[...]}"

The script splits this into:
    output_text  → human-readable portion
    output       → parsed JSON object

Usage:
    python convert_datasets.py
"""

from __future__ import annotations

import json
import os


def convert(src: str, dst: str) -> None:
    if not os.path.isfile(src):
        print(f"  skip {src}: file not found")
        return

    with open(src, encoding="utf-8") as f:
        raw = json.load(f)

    result: list[dict] = []
    errors = 0

    for i, item in enumerate(raw):
        instruction = item.get("instruction", "")
        output_raw = item.get("output", "")

        output_text = ""
        json_part = ""
        if isinstance(output_raw, str):
            head, sep, tail = output_raw.partition("\n")
            output_text = head.strip()
            json_part = tail
        else:
            json_part = output_raw

        try:
            output_obj = json.loads(json_part) if json_part else {}
        except json.JSONDecodeError as exc:
            errors += 1
            print(f"  [{src}#{i}] JSON parse error: {exc}")
            continue

        result.append({
            "instruction": instruction,
            "input_lines": [],
            "output_text": output_text,
            "output": output_obj,
        })

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = f"{src} → {dst}: {len(result)} examples"
    if errors:
        summary += f" ({errors} skipped)"
    print(f"  {summary}")


if __name__ == "__main__":
    convert("norm.txt", "dataset_norm.json")
    convert("planer.txt", "dataset_planer.json")