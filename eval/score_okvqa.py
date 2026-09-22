#!/usr/bin/env python3
"""Score OK-VQA by VQA count-accuracy.

Usage: score_okvqa.py <model_answer.jsonl> <okvqa.json> [--tag NAME]

pred = min(#refs matching / 3, 1), normalized (lower, drop punct/articles/space),
averaged over questions -- the standard OK-VQA / VQA-v1 metric.
Also prints the 5-bin histogram (fraction scoring 0,.1,..,1) like the official tool.
"""
import argparse, json, re
from collections import Counter


def norm(s):
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answer", help="infer.py model_answer.jsonl")
    ap.add_argument("benchmark", help="the converted okvqa.json eval/infer.py was run on")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    bench = json.load(open(args.benchmark))
    # key -> refs ; infer.py sets sample_uid from index/question_id
    refs = {}
    for it in bench:
        for k in (str(it.get("question_id")), str(it.get("index"))):
            refs[k] = [norm(r) for r in it["refs"]]

    rows = []
    txt = open(args.answer).read().strip()
    if txt[0] == "[":
        rows = json.loads(txt)
    else:
        rows = [json.loads(l) for l in txt.splitlines() if l.strip()]

    accs, binsum, n = [], Counter(), 0
    seen = set()
    for r in rows:
        key = str(r.get("sample_uid") or r.get("question_id") or r.get("index"))
        if key not in refs:
            continue
        seen.add(key)
        pred = norm(str(r.get("model_answer", "")))
        # take the answer up to newline (model may ramble); use first non-empty line
        if not pred:
            for line in str(r.get("model_answer", "")).splitlines():
                if line.strip():
                    pred = norm(line); break
        cnt = sum(1 for x in refs[key] if x == pred)
        a = min(cnt / 3.0, 1.0)
        accs.append(a)
        binsum[round(a, 1)] += 1
        n += 1

    if n == 0:
        print("no matched records; check sample_uid/question_id keys"); return
    acc = sum(accs) / n * 100
    print(f"== {args.tag or 'okvqa'} ==  n={n}/{len(refs)//2 if len(refs)//2>0 else n}  matched={len(seen)}")
    print(f"  VQA count-accuracy: {acc:.2f}%")
    print("  5-bin (0/.1/.2/1): " + "  ".join(f"{b}={binsum.get(b,0)/n*100:.1f}%" for b in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))


if __name__ == "__main__":
    main()
