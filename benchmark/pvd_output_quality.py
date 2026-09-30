"""Fixed held-out GSM8K/HotpotQA subset, full-path collection and answer scoring.

Uses a shared frozen chat-template prompt file for every arm. Official answer
EM/F1 definitions are applied to a declared FINAL answer extraction; missing
FINAL and generation truncation are reported separately. No model judge.
"""

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import string
import time
import urllib.request


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(predicted, gold):
    a, b = normalize(predicted), normalize(gold)
    em = float(a == b)
    if (a in ("yes", "no", "noanswer") or b in ("yes", "no", "noanswer")) and a != b:
        return em, 0.0
    common = sum((Counter(a.split()) & Counter(b.split())).values())
    if not common:
        return em, 0.0
    precision, recall = common / len(a.split()), common / len(b.split())
    return em, 2 * precision * recall / (precision + recall)


def extract_final(text):
    matches = re.findall(r"(?im)^\s*(?:\*\*)?FINAL(?:\*\*)?\s*:\s*(.+)", text)
    return matches[-1].strip().strip("* ") if matches else None


def numeric(text):
    if text is None:
        return None
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0])
    except InvalidOperation:
        return None


def gsm_answer(text):
    final = extract_final(text)
    if numeric(final) is not None:
        return str(numeric(final)), "FINAL"
    # Match the project's GSM8K last-number convention, retaining decimals and
    # signs. Keep format compliance separate from the numeric correctness score.
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return (str(Decimal(matches[-1])), "last_number") if matches else (None, "missing")


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    gsm = [json.loads(line) for line in args.gsm.read_text().splitlines() if line.strip()]
    if args.hotpot.suffix == ".parquet":
        import pyarrow.parquet as pq
        hotpot = pq.read_table(args.hotpot).to_pylist()
        for row in hotpot:
            row["_id"] = row["id"]
            row["context"] = list(zip(row["context"]["title"], row["context"]["sentences"]))
    else:
        hotpot = json.loads(args.hotpot.read_text())
    excluded = set()
    for line in args.training_source.read_text().splitlines():
        for question in json.loads(line):
            excluded.add(question.removeprefix("Question: ").strip())
    rng = random.Random(20260930)
    gsm_indices, hotpot_indices = list(range(len(gsm))), list(range(len(hotpot)))
    rng.shuffle(gsm_indices)
    rng.shuffle(hotpot_indices)
    items = []
    for benchmark, source, indices, count, limit in (
        ("gsm8k", gsm, gsm_indices, 16, 512),
        ("hotpotqa", hotpot, hotpot_indices, 24, 128),
    ):
        accepted = 0
        for index in indices:
            row = source[index]
            if row["question"].strip() in excluded:
                continue
            if benchmark == "gsm8k":
                user = ("Solve this math problem. Show a brief step-by-step solution, "
                        "then write the final numeric answer on its own line as FINAL: <number>.\n\n"
                        + row["question"])
                gold = row["answer"].rsplit("####", 1)[-1].strip()
            else:
                context = "\n\n".join(title + ":\n" + "".join(sentences)
                                        for title, sentences in row["context"])
                user = ("Use the following passages to answer the question. Give a brief "
                        "explanation, then put only the short answer on its own line as "
                        "FINAL: <answer>.\n\nPassages:\n" + context
                        + "\n\nQuestion: " + row["question"])
                gold = row["answer"]
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user},
            ], tokenize=False, add_generation_prompt=True)
            tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if tokens + limit + 8 > 2304 or (benchmark == "hotpotqa" and tokens < 600):
                continue  # Keep all passages intact; never cut away answer evidence.
            items.append({"id": f"{benchmark}:{row.get('_id', index)}",
                          "benchmark": benchmark, "source_index": index,
                          "question": row["question"], "gold_answer": gold,
                          "prompt": prompt, "prompt_sha256": digest(prompt.encode()),
                          "prompt_tokens": tokens, "max_new_tokens": limit})
            accepted += 1
            if accepted == count:
                break
        if accepted != count:
            raise ValueError(f"not enough fitting {benchmark} examples")
    # Mix task order identically across the three arms; selection uses no model outputs.
    rng.shuffle(items)
    payload = {"schema": "pvd.output_quality.dataset.v1", "seed": 20260930,
               "selection": "seeded shuffle, intact context fits context+output+horizon<=2304; "
                            "HotpotQA prompt>=600; exclude repository ReAct questions",
               "sources": {"gsm_test_sha256": digest(args.gsm.read_bytes()),
                           "hotpot_dev_sha256": digest(args.hotpot.read_bytes()),
                           "excluded_training_source_sha256": digest(args.training_source.read_bytes())},
               "items": items}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"count": len(items), "sha256": digest(args.output.read_bytes()),
                      "token_range": [min(x["prompt_tokens"] for x in items),
                                      max(x["prompt_tokens"] for x in items)]}))


def generate(url, item):
    payload = {"text": item["prompt"], "stream": True, "sampling_params": {
        "temperature": 0, "max_new_tokens": item["max_new_tokens"], "ignore_eos": False}}
    req = urllib.request.Request(url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start_unix, start = time.time(), time.perf_counter()
    first, final, event_count = None, {}, 0
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if line.startswith(b"data:") and line.strip() != b"data: [DONE]":
                first = first if first is not None else time.perf_counter() - start
                final = json.loads(line[5:].strip())
                event_count += 1
    meta = final.get("meta_info", {})
    if final.get("error") or meta.get("prompt_tokens") != item["prompt_tokens"]:
        raise ValueError(f"failed response or tokenization mismatch: {final}")
    text = final.get("text", "")
    answer = extract_final(text)
    gsm_value, gsm_method = gsm_answer(text) if item["benchmark"] == "gsm8k" else (None, None)
    return {"id": item["id"], "benchmark": item["benchmark"],
            "started_unix": start_unix, "first_event_seconds": first,
            "wall_seconds": time.perf_counter() - start,
            "prompt_sha256": item["prompt_sha256"], "prompt_tokens": meta["prompt_tokens"],
            "max_new_tokens": item["max_new_tokens"],
            "completion_tokens": meta.get("completion_tokens"),
            "finish_reason": meta.get("finish_reason"), "request_id": meta.get("id"),
            "text": text, "final_answer": answer, "gold_answer": item["gold_answer"],
            "gsm_numeric_answer": gsm_value, "gsm_extraction": gsm_method,
            "truncated": meta.get("completion_tokens") == item["max_new_tokens"],
            "events": event_count}


def collect(args):
    dataset = json.loads(args.dataset.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for item in dataset["items"]:
            row = generate(args.url, item)
            row.update(arm=args.arm, dataset_sha256=digest(args.dataset.read_bytes()))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({key: row[key] for key in (
                "arm", "id", "completion_tokens", "wall_seconds", "final_answer", "truncated")}),
                flush=True)


def compare(args):
    arms = {}
    for arm in ("full", "target", "joint"):
        rows = [json.loads(line) for line in (args.folder / f"{arm}.jsonl").read_text().splitlines()]
        if len(rows) != 40 or len({r["id"] for r in rows}) != 40 or any(r["arm"] != arm for r in rows):
            raise ValueError(f"incomplete or mislabeled {arm}")
        arms[arm] = {r["id"]: r for r in rows}
    for arm in ("target", "joint"):
        if list(arms[arm]) != list(arms["full"]):
            raise ValueError("unmatched requests/order")
        for key, baseline in arms["full"].items():
            row = arms[arm][key]
            if any(row[k] != baseline[k] for k in (
                "dataset_sha256", "prompt_sha256", "prompt_tokens", "max_new_tokens", "gold_answer")):
                raise ValueError("unmatched dataset or generation bounds")
    metrics = {}
    for arm, rows in arms.items():
        metrics[arm] = {}
        for benchmark in ("gsm8k", "hotpotqa"):
            subset = [row for row in rows.values() if row["benchmark"] == benchmark]
            scores = [answer_scores(r["final_answer"] or "", r["gold_answer"]) if benchmark == "hotpotqa"
                      else (float(numeric(r["gsm_numeric_answer"]) is not None
                                  and numeric(r["gsm_numeric_answer"]) == numeric(r["gold_answer"])), 0)
                      for r in subset]
            metrics[arm][benchmark] = {
                "n": len(subset), "answer_em": statistics.mean(s[0] for s in scores),
                "answer_f1": statistics.mean(s[1] for s in scores) if benchmark == "hotpotqa" else None,
                "correct": sum(s[0] for s in scores),
                "strict_final_em": statistics.mean(float(numeric(r["final_answer"]) is not None
                    and numeric(r["final_answer"]) == numeric(r["gold_answer"])) for r in subset)
                    if benchmark == "gsm8k" else statistics.mean(s[0] for s in scores),
                "missing_final": sum(r["final_answer"] is None for r in subset),
                "truncated": sum(r["truncated"] for r in subset),
                "mean_output_tokens": statistics.mean(r["completion_tokens"] for r in subset),
                "median_client_seconds": statistics.median(r["wall_seconds"] for r in subset),
            }
    payload = {"metrics": metrics, "paired_rows": [
        {"id": key, "benchmark": base["benchmark"], "gold": base["gold_answer"],
         "answers": {arm: (rows[key]["gsm_numeric_answer"] if base["benchmark"] == "gsm8k"
                           else rows[key]["final_answer"]) for arm, rows in arms.items()},
         "output_matches_full": {arm: rows[key]["text"] == base["text"]
                                 for arm, rows in arms.items()}}
        for key, base in arms["full"].items()],
        "caveat": "Fixed 40-question subset, not full public benchmark; GSM8K FINAL then "
                  "last-number extraction, strict FINAL metric also reported; HotpotQA strict "
                  "FINAL extraction, missing FINAL scores zero; generated lengths can differ."}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("gsm", "hotpot", "training-source", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--tokenizer", required=True)
    sample = sub.add_parser("collect")
    sample.add_argument("--dataset", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--arm", choices=("full", "target", "joint"), required=True)
    sample.add_argument("--url", default="http://10.10.1.2:8001")
    cmp = sub.add_parser("compare")
    cmp.add_argument("--folder", type=Path, required=True)
    cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
