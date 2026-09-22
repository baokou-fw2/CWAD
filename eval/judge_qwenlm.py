import argparse
import json
import os
import re
import sys
import threading
import time

from tqdm import tqdm

MCQ_BENCHMARKS = [
    "realworldqa",
    "hrbench-4k",
    "hrbench-8k",
    "vstar",
    "mme-realworld",
    "mme-realworld-cn",
    "mme-realworld-lite",
    "mmstar",
    "cv-bench",
    "seedbench",
    "dualworld",
    "bwdhd",
    "bwdhd_hisc",
    "lres_839hi",
    "mmbench-dev-en",
    "oracle596",
    "vstar2",
    "vstar2_hiA",
    "vstar3",
    "vstar3_A",
    "vstar2_B512",
    "vstar2_hiA_B512",
]

GENERAL_BENCHMARKS = [
    "mmvet",
]

POPE_BENCHMARKS = [
    "pope",
    "pope_adv",
    "pope_pop",
    "pope_random",
]

MMVP_BENCHMARKS = [
    "mmvp",
]

PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)


def extract_first_option(text):
    if not text:
        return ""
    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])[\.\)\s]", text)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z])", text)
    if match:
        return match.group(1)
    return ""


def extract_mcq_option(answer):
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip()
    pattern = r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))"
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    return ""


def pope_extract(text):
    if not isinstance(text, str) or not text:
        return ""
    t = text.lstrip("*").strip()
    if t.lower().startswith("answer"):
        t = t[len("answer") :].lstrip(":").lstrip("*").strip()
    t_lower = t.lower()
    if t_lower.startswith("yes"):
        return "yes"
    if t_lower.startswith("no"):
        return "no"
    last = t.rstrip(".").rstrip("*").strip().split()[-1] if t.strip() else ""
    last = last.lstrip("*").rstrip("*").lower()
    if last in ("yes", "no"):
        return last
    return text.strip()


def mmvp_extract(text):
    if not isinstance(text, str) or not text:
        return ""
    t = text.strip().lower()
    match = re.search(r"\(([ab])\)", t)
    if match:
        return f"({match.group(1)})"
    match = re.search(r"\b([ab])\b", t)
    if match:
        return f"({match.group(1)})"
    return ""


def first_letter_match(gt, answer):
    gt_val = extract_mcq_option(gt)
    pred_val = extract_first_option(answer)
    return bool(gt_val and pred_val and gt_val == pred_val)


def extract_answer(model_answer_raw):
    if "<answer>" in model_answer_raw:
        start = model_answer_raw.find("<answer>")
        end = model_answer_raw.find("</answer>")
        if start != -1 and end != -1:
            return model_answer_raw[start + len("<answer>") : end].strip()
    if "Answer:" in model_answer_raw:
        return model_answer_raw[model_answer_raw.find("Answer:") :].strip()
    return model_answer_raw.strip()


# Sentinel for "the judge backend never answered". It must NOT be "No": once
# written into the results file a failed call is indistinguishable from a real
# negative verdict, and the damage is invisible -- every LLM-judged item of one
# checkpoint came back "No" this way and scored as a catastrophic model collapse
# when the model was in fact fine.
JUDGE_ERROR = "__judge_error__"

# Above this share of failed calls the whole run is untrustworthy; refuse to
# emit a score rather than reporting one that silently counts every failed call
# as a wrong answer.
JUDGE_ERROR_ABORT_RATE = 0.05


def normalize_judge_response(text):
    text = str(text or "").strip()
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :].strip()
    match = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else text


def judge_via_api(
    prompts,
    api_base,
    api_key,
    judge_model,
    judge_max_tokens,
    parallel_workers=32,
    enable_thinking=None,
):
    from openai import OpenAI

    thread_local = threading.local()

    def get_client():
        c = getattr(thread_local, "client", None)
        if c is None:
            c = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
            thread_local.client = c
        return c

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [""] * len(prompts)

    def call_one(idx, prompt):
        client = get_client()
        for attempt in range(3):
            try:
                extra_kwargs = {}
                if enable_thinking is not None:
                    extra_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": enable_thinking}
                    }
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=judge_max_tokens,
                    **extra_kwargs,
                )
                return idx, (resp.choices[0].message.content or "").strip()
            except Exception:
                if attempt < 2:
                    time.sleep(1.0)
        # Exhausted the retries: report the failure as itself instead of
        # fabricating a "No".
        return idx, JUDGE_ERROR

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(call_one, i, p) for i, p in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
            idx, text = future.result()
            results[idx] = text

    return results


def judge_via_vllm(prompts, judge_model_path, judge_max_tokens, enable_thinking=None):
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(max_tokens=judge_max_tokens, temperature=0)
    llm = LLM(model=judge_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path)

    chat_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        chat_prompts.append(text)

    outputs = llm.generate(chat_prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def main():
    parser = argparse.ArgumentParser(description="LLM-based judge for benchmark evaluation")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--judge_model_path", default=None, type=str, help="Local model path for vLLM-based judging")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI-compatible API base URL for judging")
    parser.add_argument("--api_key", default="EMPTY", type=str)
    parser.add_argument("--judge_model", default=None, type=str, help="Model name for API-based judging")
    parser.add_argument("--judge_max_tokens", default=2048, type=int)
    parser.add_argument(
        "--parallel_workers",
        default=32,
        type=int,
        help="Concurrent judge requests, API path only (the vLLM path keeps its "
        "own worker count). 32 suits a local server; an external API dropped 16%% "
        "of calls at that width, while 4 workers recovered 48/48.",
    )
    parser.add_argument(
        "--judge_enable_thinking",
        default=None,
        choices=["True", "False"],
        help="Pass enable_thinking to Qwen-family judge chat templates.",
    )
    args = parser.parse_args()

    if not args.api_base and not args.judge_model_path:
        print(
            "ERROR: Either --api_base (for API-based judging) or --judge_model_path "
            "(for local vLLM judging) must be provided.",
            file=sys.stderr,
        )
        sys.exit(1)

    answer_path = f"model_answer/{args.benchmark}/{args.model}_answer.jsonl"
    save_path = f"judge/{args.benchmark}/{args.model}_answer.jsonl"
    os.makedirs(f"judge/{args.benchmark}", exist_ok=True)
    is_mcq = args.benchmark in MCQ_BENCHMARKS
    is_pope = args.benchmark in POPE_BENCHMARKS
    is_mmvp = args.benchmark in MMVP_BENCHMARKS
    is_general = args.benchmark in GENERAL_BENCHMARKS
    judge_enable_thinking = (
        None if args.judge_enable_thinking is None else args.judge_enable_thinking == "True"
    )

    data_list = []
    with open(answer_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))

    try:
        from mathruler.grader import grade_answer

        has_mathruler = True
    except ImportError:
        has_mathruler = False

    to_llm_indices = []
    prompt_lists = []

    for i, item in enumerate(tqdm(data_list, desc="Rule-based grading")):
        question = item["query"].replace("<image>", "")
        model_answer_raw = item["model_answer"]
        extracted_answer = extract_answer(model_answer_raw)
        gt = item["response"]
        item["extracted_answer"] = extracted_answer

        is_correct = False

        if is_pope:
            pred = pope_extract(extracted_answer)
            if pred == gt.strip().lower():
                is_correct = True
                item["judge"] = "Yes"
                item["judge_source"] = "pope_exact"

        if not is_correct and is_mmvp:
            pred = mmvp_extract(extracted_answer)
            gt_norm = gt.strip().lower()
            if pred and pred == gt_norm:
                is_correct = True
                item["judge"] = "Yes"
                item["judge_source"] = "mmvp_option"

        if not is_correct and has_mathruler:
            try:
                is_correct = grade_answer(gt, extracted_answer)
            except Exception:
                is_correct = False

        is_letter_correct = False
        if not is_correct and is_mcq:
            try:
                is_letter_correct = first_letter_match(gt, extracted_answer)
            except Exception:
                is_letter_correct = False

        if is_correct and "judge" not in item:
            item["judge"] = "Yes"
            item["judge_source"] = "mathruler"
        elif is_letter_correct:
            item["judge"] = "Yes"
            item["judge_source"] = "first letter"
        else:
            prompt = PROMPT_TEMPLATE.format(gt=gt, response=extracted_answer, question=question)
            to_llm_indices.append(i)
            prompt_lists.append(prompt)

    if prompt_lists:
        print(f"Calling LLM judge for {len(prompt_lists)} remaining cases...")
        if args.api_base:
            judge_model_name = args.judge_model or "default"
            results = judge_via_api(
                prompt_lists,
                args.api_base,
                args.api_key,
                judge_model_name,
                args.judge_max_tokens,
                parallel_workers=args.parallel_workers,
                enable_thinking=judge_enable_thinking,
            )
        else:
            results = judge_via_vllm(
                prompt_lists,
                args.judge_model_path,
                args.judge_max_tokens,
                enable_thinking=judge_enable_thinking,
            )

        judge_errors = 0
        for idx_in_llm, response_text in enumerate(results):
            original_idx = to_llm_indices[idx_in_llm]
            data_list[original_idx]["judge_raw"] = response_text
            if str(response_text).strip() == JUDGE_ERROR:
                judge_errors += 1
                data_list[original_idx]["judge"] = "ERROR"
                data_list[original_idx]["judge_source"] = "llm_error"
                continue
            data_list[original_idx]["judge"] = normalize_judge_response(response_text)
            data_list[original_idx]["judge_source"] = "llm"

    print(f"Total: {len(data_list)}, LLM used: {len(prompt_lists)}")
    if prompt_lists and judge_errors:
        rate = judge_errors / len(prompt_lists)
        print(f"WARNING: {judge_errors}/{len(prompt_lists)} LLM judge calls failed ({rate:.1%})")

    with open(save_path, "w", encoding="utf-8") as out_file:
        json.dump(data_list, out_file, ensure_ascii=False, indent=4)
    print(f"Saved judge results to: {save_path}")

    # Written after the results are on disk so the failure is inspectable, but
    # the run must still fail: a score built on a dead judge backend is worse
    # than no score, because it looks like a result.
    if prompt_lists and judge_errors / len(prompt_lists) > JUDGE_ERROR_ABORT_RATE:
        raise SystemExit(
            f"judge backend failed for {judge_errors}/{len(prompt_lists)} requests "
            f"({judge_errors / len(prompt_lists):.1%} > {JUDGE_ERROR_ABORT_RATE:.0%}); "
            "refusing to report a score that counts those as wrong answers "
            f"(see {save_path})"
        )


if __name__ == "__main__":
    main()
