#!/usr/bin/env python
"""Build the frozen pilot prompt set: 200 harmful (HarmBench) + 200 benign (XSTest safe).

Sources are pinned so the set is reproducible from this script alone:
  harmful: HarmBench "standard" text behaviors (all 200), centerforaisafety/HarmBench @ 8e1604d
  benign : XSTest "safe" prompts, 20 per type (10 types), Paul/XSTest @ f600c99, seed below

Writes data/pilot/prompts.jsonl (git-ignored) and prints its sha256; record that in the log.

Usage:
  python scripts/pilot_prompts.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

HARMBENCH_URL = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
                 "8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/"
                 "harmbench_behaviors_text_all.csv")
XSTEST_URL = ("https://huggingface.co/datasets/Paul/XSTest/resolve/"
              "f600c994b256f12867dfa5b3eb3d545a3e62f8b5/xstest_prompts.csv")
XSTEST_PER_TYPE = 20
SEED = 0
OUT = Path("data/pilot/prompts.jsonl")


def fetch_csv(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))


def main() -> int:
    hb = [r for r in fetch_csv(HARMBENCH_URL) if r["FunctionalCategory"] == "standard"]
    assert len(hb) == 200, f"expected 200 HarmBench standard behaviors, got {len(hb)}"
    harmful = [{"id": f"hb-{r['BehaviorID']}", "split": "harmful", "source": "harmbench",
                "category": r["SemanticCategory"], "prompt": r["Behavior"]} for r in hb]

    xs = [r for r in fetch_csv(XSTEST_URL) if r["label"] == "safe"]
    by_type = defaultdict(list)
    for r in xs:
        by_type[r["type"]].append(r)
    rng = random.Random(SEED)
    benign = []
    for t in sorted(by_type):
        for r in rng.sample(sorted(by_type[t], key=lambda r: int(r["id"])), XSTEST_PER_TYPE):
            benign.append({"id": f"xs-{r['id']}", "split": "benign", "source": "xstest",
                           "category": t, "prompt": r["prompt"]})
    assert len(benign) == 200, f"expected 200 XSTest safe prompts, got {len(benign)}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in harmful + benign)
    OUT.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"[prompts] wrote {OUT}: {len(harmful)} harmful + {len(benign)} benign, sha256 {sha[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
