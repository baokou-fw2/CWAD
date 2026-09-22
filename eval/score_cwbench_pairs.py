#!/usr/bin/env python3
"""Score a CWBench run: per-world accuracy, CWPA and CWFR.

Every pair is the same question against two images -- member A the original,
member B the counterfactual edit -- and the two members never share an answer.
A pair therefore counts as correct only when *both* members are answered
correctly: a model that ignores the image and always emits one letter scores half
of member A and nothing at all on pairs, which is the property that makes this
column discriminating.

    CWPA = both members correct          (higher is better)
    CWFR = A Acc + B Acc - 2 * CWPA      (lower is better)

so CWFR is the part of the two per-world accuracies that does not survive being
asked about both worlds: it is exactly the pairs where one member was right and
the other wrong. A model that answers from the question alone scores high
per-world accuracy, zero CWPA, and puts all of it in CWFR.

Input is eval/infer.py's answer file (JSONL, one record per item). The records
carry the gold ``response``, the ``pair_id`` and the ``member`` through from
prepare_cwbench.py, so no separate benchmark file is needed.

    python3 eval/score_cwbench_pairs.py model_answer/cwbench/<model-name>_answer.jsonl
    python3 eval/score_cwbench_pairs.py answers.jsonl --json out.json

The queries ask for the option letter alone, so scoring is a letter match: the
answer is unwrapped from any thinking block or <answer> marker, and the first
standalone A-E in it is taken. Unparseable and API-error answers count as wrong
and are reported separately, never dropped.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LETTER = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")


def read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def unwrap(answer: str) -> str:
    """Drop chain-of-thought and answer markers, keeping the part that decides."""
    think_end = answer.rfind("</think>")
    if think_end != -1:
        answer = answer[think_end + len("</think>"):]
    start = answer.rfind("<answer>")
    end = answer.find("</answer>", start + len("<answer>")) if start != -1 else -1
    if start != -1 and end != -1:
        answer = answer[start + len("<answer>"):end]
    if "Answer:" in answer:
        answer = answer[answer.rfind("Answer:") + len("Answer:"):]
    return answer.strip()


def predicted_letter(answer: str) -> str | None:
    if not isinstance(answer, str) or not answer.strip():
        return None
    if answer.lstrip().startswith(("[API_ERROR]", "[FUTURE_ERROR]")):
        return None
    body = unwrap(answer)
    match = LETTER.search(body)
    if match is None:
        return None
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", type=Path, help="eval/infer.py answer file (JSONL or JSON)")
    parser.add_argument("--json", type=Path, help="also write these numbers as JSON")
    args = parser.parse_args()

    records = read_records(args.answers)
    for field in ("pair_id", "member", "response"):
        if any(field not in record for record in records):
            raise SystemExit(f"{args.answers.name} has records without {field!r}; "
                             f"build it with eval/prepare_cwbench.py")

    members: dict[str, dict[str, bool]] = defaultdict(dict)
    unparseable = 0
    errors = 0
    for record in records:
        letter = predicted_letter(record.get("model_answer", ""))
        if str(record["model_answer"] or "").lstrip().startswith(("[API_ERROR]", "[FUTURE_ERROR]")):
            errors += 1
        elif letter is None:
            unparseable += 1
        pair_id = str(record["pair_id"])
        members[pair_id][str(record["member"])] = (
            letter is not None and letter == str(record["response"]).strip().upper()
        )

    complete = {p: v for p, v in members.items() if {"A", "B"} <= set(v)}
    if not complete:
        raise SystemExit("no complete pairs in the answer file")
    dropped = sorted(set(members) - set(complete))

    total = len(complete)
    a_correct = sum(1 for v in complete.values() if v["A"])
    b_correct = sum(1 for v in complete.values() if v["B"])
    both = sum(1 for v in complete.values() if v["A"] and v["B"])
    only_a = sum(1 for v in complete.values() if v["A"] and not v["B"])
    only_b = sum(1 for v in complete.values() if v["B"] and not v["A"])
    neither = total - both - only_a - only_b

    def pct(count: int) -> str:
        return f"{100.0 * count / total:6.2f}% ({count}/{total})"

    member_a = 100.0 * a_correct / total
    member_b = 100.0 * b_correct / total
    cwpa = 100.0 * both / total
    # CWFR = A Acc + B Acc - 2 * CWPA, i.e. exactly the pairs where the model got
    # one world right and the other wrong. Computed from the counts, so it is
    # independent of the rounding of the three percentages above.
    cwfr = 100.0 * (only_a + only_b) / total

    print(f"== {args.answers.name} ==")
    print(f"  items            {len(records)}   complete pairs {total}"
          + (f"   incomplete {len(dropped)} (not counted)" if dropped else ""))
    print(f"  unanswered/error {errors}   unparseable letter {unparseable}")
    print(f"  member A (original)      {pct(a_correct)}")
    print(f"  member B (counterfactual){pct(b_correct)}")
    print(f"  CWPA  (both correct, higher better)   {pct(both)}")
    print(f"  CWFR  (A+B-2*CWPA, lower better)     {cwfr:6.2f}% ({only_a + only_b}/{total})")
    print(f"  A only {only_a}   B only {only_b}   neither {neither}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "answers": str(args.answers),
                    "items": len(records),
                    "pairs": total,
                    "member_a_accuracy": member_a,
                    "member_b_accuracy": member_b,
                    "cwpa": cwpa,
                    "cwfr": cwfr,
                    "only_a": only_a,
                    "only_b": only_b,
                    "neither": neither,
                    "unanswered": errors,
                    "unparseable": unparseable,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
