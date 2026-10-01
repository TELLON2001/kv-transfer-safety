#!/usr/bin/env python
"""Label pilot generations refuse / comply / incoherent with a Claude judge (fixed rubric).

The rubric is configs/judge_rubric.md, committed before any pilot output existed. The judge sees
only the request and the response, never the condition. Output is constrained to a JSON schema.

  submit   send every not-yet-judged generation as one Message Batch (50% price, async);
           the batch id and custom_id map are saved to judge_state.json
  collect  fetch finished batches and append labels to judgments.jsonl (safe to re-run)
  sync     judge the first N unjudged items directly (for testing the setup)

Credentials: ANTHROPIC_API_KEY or an `ant auth login` profile.

Usage:
  python scripts/pilot_judge.py submit  --gen results/pilot/<pair>/<mapper>/generations.jsonl
  python scripts/pilot_judge.py collect --gen results/pilot/<pair>/<mapper>/generations.jsonl
  python scripts/pilot_judge.py sync    --gen ... --n 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

RUBRIC_PATH = Path(__file__).resolve().parents[1] / "configs" / "judge_rubric.md"
JUDGE_MODEL = "claude-sonnet-5"
EFFORT = "medium"
MAX_TOKENS = 4000
LABELS = ("refuse", "comply", "incoherent")
SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}


def key(r: dict) -> str:
    return f"{r['id']}|{r['condition']}"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def user_message(r: dict) -> str:
    cut = "\n(The response was cut off at the token limit.)" if r.get("hit_limit") else ""
    return (f"<request>\n{r['prompt']}\n</request>\n\n"
            f"<response>\n{r['response'] or '(empty response)'}\n</response>{cut}")


def params(r: dict, rubric: str) -> dict:
    return {
        "model": JUDGE_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": rubric,
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": user_message(r)}],
    }


def parse(msg) -> dict:
    """Judgment fields from a Message; judge refusals and unparseable output get label None."""
    out = {"judge_model": msg.model, "stop_reason": msg.stop_reason}
    if msg.stop_reason == "refusal":
        return {**out, "label": None, "reason": "judge refused", "judge_status": "judge_refused"}
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        j = json.loads(text)
        if j.get("label") not in LABELS:
            raise ValueError(j.get("label"))
        return {**out, "label": j["label"], "reason": j.get("reason", ""), "judge_status": "ok"}
    except (json.JSONDecodeError, ValueError) as e:
        return {**out, "label": None, "reason": f"unparseable: {e}", "judge_status": "parse_error"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["submit", "collect", "sync"])
    ap.add_argument("--gen", required=True, help="generations.jsonl from pilot_generate.py")
    ap.add_argument("--n", type=int, default=5, help="sync mode: how many items")
    args = ap.parse_args()

    import anthropic

    gen_path = Path(args.gen)
    out_path = gen_path.parent / "judgments.jsonl"
    state_path = gen_path.parent / "judge_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    gens = {key(r): r for r in read_jsonl(gen_path)}
    # any judgment is final, including judge refusals / parse errors (those go to hand review),
    # so a refused item is never resubmitted in a loop
    judged = {key(r) for r in read_jsonl(out_path)}
    pending = {k for b in state["batches"] if not b.get("collected") for k in b["ids"].values()}
    client = anthropic.Anthropic()

    def write(rows: list[dict]) -> None:
        with out_path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.mode in ("submit", "sync"):
        todo = [k for k in gens if k not in judged and k not in pending]
        if args.mode == "sync":
            todo = todo[:args.n]
        print(f"[judge] {len(gens)} generations, {len(judged)} judged, {len(pending)} pending, "
              f"{len(todo)} to {args.mode}")
        if not todo:
            return 0
        if args.mode == "sync":
            rows = []
            for k in todo:
                r = gens[k]
                msg = client.messages.create(**params(r, rubric))
                rows.append({"id": r["id"], "condition": r["condition"], **parse(msg)})
                print(f"  {k}: {rows[-1]['label']}  ({rows[-1]['reason'][:80]})")
            write(rows)
            return 0
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        ids = {f"r{i}": k for i, k in enumerate(todo)}  # custom_id must be short and [A-Za-z0-9_-]
        batch = client.messages.batches.create(requests=[
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params(gens[k], rubric)))
            for cid, k in ids.items()])
        state["batches"].append({"batch_id": batch.id, "ids": ids, "rubric_sha256": hashlib.sha256(rubric.encode("utf-8")).hexdigest(),
                                 "submitted": datetime.now().isoformat(timespec="seconds")})
        state_path.write_text(json.dumps(state, indent=1))
        print(f"[judge] submitted batch {batch.id} with {len(ids)} requests; run `collect` later")
        return 0

    # collect
    for b in state["batches"]:
        if b.get("collected"):
            continue
        batch = client.messages.batches.retrieve(b["batch_id"])
        c = batch.request_counts
        print(f"[judge] {b['batch_id']}: {batch.processing_status} "
              f"(processing {c.processing}, ok {c.succeeded}, errored {c.errored}, expired {c.expired})")
        if batch.processing_status != "ended":
            continue
        rows, failed = [], 0
        for res in client.messages.batches.results(b["batch_id"]):
            k = b["ids"][res.custom_id]
            r_id, cond = k.split("|", 1)
            if res.result.type == "succeeded":
                rows.append({"id": r_id, "condition": cond, **parse(res.result.message)})
            else:
                failed += 1  # errored / canceled / expired: left unjudged, the next submit retries it
        write(rows)
        b["collected"] = datetime.now().isoformat(timespec="seconds")
        state_path.write_text(json.dumps(state, indent=1))
        bad = sum(1 for r in rows if r["label"] is None)
        print(f"[judge]   wrote {len(rows)} judgments ({bad} without a label); {failed} failed, resubmit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
