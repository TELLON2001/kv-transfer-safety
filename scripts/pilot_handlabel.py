#!/usr/bin/env python
"""Blinded hand-label sample for checking the judge, and Cohen's kappa against it.

  export    per condition: a random sample, plus top-ups so each label the judge used appears at least
            --min-per-label times (rare labels such as incoherent would otherwise be missed). Items are
            shuffled together; the sheet shows only request and response (no condition, no judge label).
            Writes <sheet>_sheet.csv (fill the `your_label` column) and <sheet>_key.json.
            Works before the judge has run: with no judgments.jsonl it samples per condition, split
            evenly between harmful and benign, with no top-ups. Refuses to overwrite a sheet that
            already has labels in it.
  followup  a further sheet for after the judge has run, when an earlier sheet was exported without
            judge labels: every item the judge left unlabeled (judge refusals, parse errors), plus
            top-ups so each judge label appears at least --min-per-label times per condition across
            all sheets. Skips items already on any sheet.
  score     reads every filled sheet in the folder and reports kappa vs the judge (labels read from
            judgments.jsonl at scoring time, so a sheet can be exported first): on the random part
            (unbiased) and on everything, plus the confusion matrix and per-condition agreement.
            Writes hand_labels.jsonl, which pilot_stats.py uses for items the judge left unlabeled.

Usage:
  python scripts/pilot_handlabel.py export   --gen results/pilot/<pair>/<mapper>/generations.jsonl
  python scripts/pilot_handlabel.py followup --gen ... --sheet handlabel2
  python scripts/pilot_handlabel.py score    --gen ...
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("refuse", "comply", "incoherent")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def latest_judgments(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    return {(r["id"], r["condition"]): r for r in read_jsonl(path)}  # last line per key wins


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def sheets(folder: Path) -> dict[str, dict]:
    """Existing sheet name -> its key."""
    return {p.name[:-len("_key.json")]: json.loads(p.read_text()) for p in sorted(folder.glob("*_key.json"))}


def write_sheet(gen_path: Path, name: str, picked: list, gens: dict, judg: dict) -> None:
    sheet = gen_path.parent / f"{name}_sheet.csv"
    if sheet.exists():
        with sheet.open(encoding="utf-8-sig", newline="") as fh:
            if any(r["your_label"].strip() for r in csv.DictReader(fh)):
                raise SystemExit(f"[handlabel] {sheet} already has labels; move it aside to re-export")
    with sheet.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "request", "response", "truncated", "your_label", "notes"])
        for i, (k, _) in enumerate(picked, 1):
            g = gens[k]
            w.writerow([i, g["prompt"], g["response"], "yes" if g["hit_limit"] else "", "", ""])
    key = {str(i): {"id": k[0], "condition": k[1], "split": gens[k]["split"], "how": how,
                    "judge_label": judg.get(k, {}).get("label")} for i, (k, how) in enumerate(picked, 1)}
    (gen_path.parent / f"{name}_key.json").write_text(json.dumps(key, indent=1))
    print(f"[handlabel] {len(picked)} items -> {sheet}")
    print(f"[handlabel] fill `your_label` with one of {', '.join(LABELS)}; do not open {name}_key.json first")
    # balance of the random part only: counts of judge-selected items by condition would leak results
    print("[handlabel] random items by condition:",
          dict(Counter(v["condition"] for v in key.values() if v["how"] == "random")))


def export(gen_path: Path, name: str, per_cond: int, min_per_label: int, seed: int) -> None:
    gens = {(r["id"], r["condition"]): r for r in read_jsonl(gen_path)}
    judg = latest_judgments(gen_path.parent / "judgments.jsonl")
    rng = random.Random(seed)
    picked = []  # (key, how)
    if not judg:
        # judge not run yet: random per condition, half harmful / half benign
        by_cs = defaultdict(list)
        for k in sorted(gens):
            by_cs[(k[1], gens[k]["split"])].append(k)
        for cond in sorted({c for c, _ in by_cs}):
            splits = sorted(s for c, s in by_cs if c == cond)
            for j, s in enumerate(splits):
                n = per_cond // len(splits) + (j < per_cond % len(splits))
                picked += [(k, "random") for k in rng.sample(by_cs[(cond, s)], n)]
        print("[handlabel] no judgments yet: random sample only, no label top-ups")
    by_cond = defaultdict(list)
    for k in sorted(gens):
        if k in judg and judg[k]["label"] is not None:
            by_cond[k[1]].append(k)
    for cond in sorted(by_cond):
        keys = by_cond[cond]
        rand = rng.sample(keys, min(per_cond, len(keys)))
        picked += [(k, "random") for k in rand]
        have = Counter(judg[k]["label"] for k in rand)
        for lab in LABELS:
            pool = [k for k in keys if judg[k]["label"] == lab and k not in rand]
            need = max(0, min(min_per_label, have[lab] + len(pool)) - have[lab])
            picked += [(k, f"topup_{lab}") for k in rng.sample(pool, need)]
    # judge refusals / parse errors always go to hand review
    picked += [(k, "judge_unlabeled") for k, j in sorted(judg.items()) if j["label"] is None and k in gens]
    rng.shuffle(picked)
    write_sheet(gen_path, name, picked, gens, judg)


def followup(gen_path: Path, name: str, min_per_label: int, seed: int) -> None:
    gens = {(r["id"], r["condition"]): r for r in read_jsonl(gen_path)}
    judg = latest_judgments(gen_path.parent / "judgments.jsonl")
    if not judg:
        raise SystemExit("[handlabel] no judgments yet; followup is for after pilot_judge.py collect")
    prior = {(v["id"], v["condition"]) for s, key in sheets(gen_path.parent).items() if s != name
             for v in key.values()}
    rng = random.Random(seed)
    picked = [(k, "judge_unlabeled") for k, j in sorted(judg.items())
              if j["label"] is None and k in gens and k not in prior]
    for cond in sorted({k[1] for k in gens}):
        have = Counter(judg[k]["label"] for k in prior if k[1] == cond and k in judg)
        for lab in LABELS:
            pool = [k for k in sorted(judg) if k[1] == cond and judg[k]["label"] == lab and k not in prior]
            need = min(max(0, min_per_label - have[lab]), len(pool))
            picked += [(k, f"topup_{lab}") for k in rng.sample(pool, need)]
    rng.shuffle(picked)
    print(f"[handlabel] {len(prior)} items already on other sheets, skipped")
    write_sheet(gen_path, name, picked, gens, judg)


def score(gen_path: Path) -> None:
    judg = latest_judgments(gen_path.parent / "judgments.jsonl")
    items, hand = [], []  # items: (sheet:item, your label, key entry)
    for name, key in sheets(gen_path.parent).items():
        for v in key.values():  # current judge labels win over whatever existed at export time
            j = judg.get((v["id"], v["condition"]))
            if j is not None:
                v["judge_label"] = j["label"]
        with (gen_path.parent / f"{name}_sheet.csv").open(encoding="utf-8-sig", newline="") as fh:
            rows = {r["item"]: r["your_label"].strip().lower() for r in csv.DictReader(fh)}
        missing = [i for i, lab in rows.items() if lab not in LABELS]
        print(f"[handlabel] {name}: {len(rows) - len(missing)}/{len(rows)} labeled"
              + (f" (first missing: {missing[:5]})" if missing else ""))
        for i, lab in rows.items():
            if lab in LABELS:
                v = key[i]
                hand.append({"id": v["id"], "condition": v["condition"], "label": lab, "sheet": name,
                             "how": v["how"]})
                if v["judge_label"] in LABELS:
                    items.append((f"{name}:{i}", lab, v))
    (gen_path.parent / "hand_labels.jsonl").write_text(
        "".join(json.dumps(h) + "\n" for h in hand), encoding="utf-8")
    print(f"[handlabel] {len(hand)} hand labels -> hand_labels.jsonl "
          f"({sum(h['how'] == 'judge_unlabeled' for h in hand)} fill items the judge left unlabeled)")
    if not items:
        raise SystemExit("[handlabel] no items with both a hand label and a judge label yet")
    for name, sel in (("random sample (unbiased)", [x for x in items if x[2]["how"] == "random"]),
                      ("all labeled items", items)):
        h, j = [x[1] for x in sel], [x[2]["judge_label"] for x in sel]
        agree = sum(a == b for a, b in zip(h, j)) / len(sel) if sel else float("nan")
        print(f"\n[handlabel] {name}: n={len(sel)}  agreement={agree:.3f}  kappa={cohen_kappa(h, j):.3f}")
    print("\n[handlabel] confusion (rows = you, cols = judge), all items:")
    conf = Counter((x[1], x[2]["judge_label"]) for x in items)
    print("            " + "".join(f"{l:>12s}" for l in LABELS))
    for a in LABELS:
        print(f"{a:>12s}" + "".join(f"{conf[(a, b)]:>12d}" for b in LABELS))
    print("\n[handlabel] agreement by condition (all items):")
    for cond in sorted({x[2]["condition"] for x in items}):
        sel = [x for x in items if x[2]["condition"] == cond]
        print(f"  {cond:10s} n={len(sel):3d}  agreement={sum(x[1] == x[2]['judge_label'] for x in sel) / len(sel):.3f}")
    inc = conf[("incoherent", "comply")]
    if inc:
        print(f"\n[handlabel] WARNING: {inc} item(s) you called incoherent were judged comply "
              "(the failure mode the plan warns about)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "followup", "score"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--sheet", default=None, help="sheet name (default handlabel; handlabel2 for followup)")
    ap.add_argument("--per-condition", type=int, default=12)
    ap.add_argument("--min-per-label", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gen = Path(args.gen)
    if args.mode == "export":
        export(gen, args.sheet or "handlabel", args.per_condition, args.min_per_label, args.seed)
    elif args.mode == "followup":
        followup(gen, args.sheet or "handlabel2", args.min_per_label, args.seed)
    else:
        score(gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
