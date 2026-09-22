"""补判 judge 文件里的 llm_error 条目 (judge=="ERROR")。

复用 judge_qwenlm.py 的 PROMPT_TEMPLATE / extract_answer / normalize_judge_response 保证同源。
只重判 ERROR 条目, 写回原文件 (in-place)。并发 4 + 重试 6。

用法: python retry_judge_errors.py --benchmark seedbench --model cwad-4b_seed42 \
        --api_base http://127.0.0.1:PORT/v1 --judge_model <judge served name>
"""
import argparse
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# 复用 judge_qwenlm.py 的同源函数
from judge_qwenlm import (
    PROMPT_TEMPLATE,
    extract_answer,
    normalize_judge_response,
    JUDGE_ERROR,
    MCQ_BENCHMARKS,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--model", required=True, help="MODEL_NAME (含 _seed42)")
    p.add_argument("--api_base", required=True)
    p.add_argument("--api_key", default="EMPTY")
    p.add_argument("--judge_model", required=True)
    p.add_argument("--judge_max_tokens", default=512, type=int)
    p.add_argument("--parallel_workers", default=4, type=int)
    p.add_argument("--max_retries", default=6, type=int)
    p.add_argument("--judge_enable_thinking", default="False", choices=["True", "False"])
    args = p.parse_args()

    path = f"judge/{args.benchmark}/{args.model}_answer.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 找出所有 ERROR 条目
    err_indices = [i for i, x in enumerate(data) if x.get("judge") == "ERROR"]
    print(f"total={len(data)}, ERROR to retry={len(err_indices)}")
    if not err_indices:
        print("no ERROR to retry, exit.")
        return

    # 构造 prompt (与 judge_qwenlm.py main() 同逻辑)
    is_mcq = args.benchmark in MCQ_BENCHMARKS
    prompts = []
    n_trunc = 0
    for i in err_indices:
        item = data[i]
        question = item["query"].replace("<image>", "")
        extracted = item.get("extracted_answer") or extract_answer(item["model_answer"])
        # 超长 model_answer (如 rambling 数数 5万-19万字符) 会让 judge prompt 超 9B 上下文.
        # 截断到 2000 字符 (实测截断后 455 条全部能装进 max-model-len 16384).
        if isinstance(extracted, str) and len(extracted) > 2000:
            extracted = extracted[:2000]
            n_trunc += 1
        gt = item["response"]
        prompts.append(PROMPT_TEMPLATE.format(gt=gt, response=extracted, question=question))
    if n_trunc:
        print(f"truncated {n_trunc} overlong extracted_answer to 2000 chars")

    from openai import OpenAI
    thread_local = threading.local()
    enable_thinking = args.judge_enable_thinking == "True"

    def get_client():
        c = getattr(thread_local, "client", None)
        if c is None:
            c = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=600)
            thread_local.client = c
        return c

    def call_one(idx_in_err, prompt):
        client = get_client()
        for attempt in range(args.max_retries):
            try:
                extra = {}
                if enable_thinking is not None:
                    extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
                resp = client.chat.completions.create(
                    model=args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=args.judge_max_tokens,
                    **extra,
                )
                return idx_in_err, (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == 0:
                    print(f"[call_one {idx_in_err}] exception: {type(e).__name__}: {str(e)[:200]}", flush=True)
                if attempt < args.max_retries - 1:
                    time.sleep(2.0)
                else:
                    return idx_in_err, JUDGE_ERROR

    results = [""] * len(err_indices)
    with ThreadPoolExecutor(max_workers=args.parallel_workers) as ex:
        futures = {ex.submit(call_one, k, pr): k for k, pr in enumerate(prompts)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Retry judge"):
            k, text = fut.result()
            results[k] = text

    # 写回
    still_err = 0
    for k, text in enumerate(results):
        orig_i = err_indices[k]
        data[orig_i]["judge_raw"] = text
        if str(text).strip() == JUDGE_ERROR:
            still_err += 1
            data[orig_i]["judge"] = "ERROR"
            data[orig_i]["judge_source"] = "llm_error"
        else:
            data[orig_i]["judge"] = normalize_judge_response(text)
            data[orig_i]["judge_source"] = "llm"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    from collections import Counter
    c = Counter(x.get("judge") for x in data)
    print(f"retry done. still ERROR={still_err}/{len(err_indices)}")
    print(f"final judge dist: {dict(c)}")
    yes = c.get("Yes", 0)
    print(f"Acc: {yes}/{len(data)} = {100*yes/len(data):.2f}%")


if __name__ == "__main__":
    main()
