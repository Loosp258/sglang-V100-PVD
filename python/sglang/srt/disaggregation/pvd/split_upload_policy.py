"""A predictor-generated, bounded two-chunk Prompt upload experiment policy."""

import json
from pathlib import Path


class SplitUploadPolicy:
    def __init__(self, path: str, *, page_size: int):
        if not isinstance(path, str) or not path:
            raise ValueError("split upload policy path is required")
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("positive Prompt page size required")
        payload = json.loads(Path(path).read_text())
        self.profile = None
        if payload.get("schema") == "pvd-exact16-parametric-v1":
            self.profile = validate_profile(payload, page_size=page_size)
            self.choices = {}
            return
        if payload.get("schema") != "pvd-exact16-split-policy-v1":
            raise ValueError("unsupported Prompt split policy")
        self.choices = {}
        for key, value in payload.get("choices", {}).items():
            n = int(key)
            p = value.get("prefix")
            if (
                type(p) is not int or n < 256 or p < 0 or p >= n
                or p % page_size or (p and (p < 256 or n - p < 64))
            ):
                raise ValueError(f"invalid calibrated split for Prompt {key}")
            self.choices[n] = p

    def plan_for(self, prompt_tokens: int) -> dict:
        """Decide from length alone; unsupported lengths use the full graph."""
        if self.profile is not None:
            return plan_split(self.profile, prompt_tokens)
        prefix = self.choices.get(prompt_tokens, 0)
        return {"selected_prefix": prefix, "best_prefix": prefix,
                "predicted_gain_seconds": None}

    def prefix_for(self, prompt_tokens: int) -> int:
        return self.plan_for(prompt_tokens)["selected_prefix"]


def validate_profile(payload: dict, *, page_size: int) -> dict:
    """Validate a hardware-specific, pure-Python online timing profile."""
    config = payload.get("config", {})
    domain = payload.get("domain", {})
    chunk = config.get("prefill_chunk_tokens")
    if (
        config.get("graph_degree") != 16
        or config.get("group_heads") != 4
        or type(chunk) is not int
        or chunk <= 0
        or chunk % page_size
        or config.get("page_size") != page_size
        or type(domain.get("min_prompt_tokens")) is not int
        or type(domain.get("max_prompt_tokens")) is not int
        or domain["min_prompt_tokens"] < 256
        or domain["max_prompt_tokens"] < domain["min_prompt_tokens"]
        or type(domain.get("min_tail_tokens")) is not int
        or domain["min_tail_tokens"] < 64
        or type(domain.get("max_prefix_tokens")) is not int
        or domain["max_prefix_tokens"] < chunk
        or len(payload.get("ranks", [])) != 2
    ):
        raise ValueError("invalid exact-16 split timing profile")
    for rank in payload["ranks"]:
        if any(
            not isinstance(rank.get(name), list)
            or len(rank[name]) != count
            or any(type(coefficient) not in (int, float) for coefficient in rank[name])
            for name, count in (("arrival", 2), ("build", 3), ("extend", 3))
        ):
            raise ValueError("invalid split timing coefficients")
    uncertainty = payload.get("uncertainty_seconds")
    if type(uncertainty) not in (int, float) or uncertainty < 0:
        raise ValueError("invalid split timing uncertainty")
    return payload


def plan_split(profile: dict, prompt_tokens: int) -> dict:
    """Predict both-rank READY for every reachable one-extend boundary."""
    domain = profile["domain"]
    if not domain["min_prompt_tokens"] <= prompt_tokens <= domain["max_prompt_tokens"]:
        return {"selected_prefix": 0, "best_prefix": 0, "reason": "uncalibrated length", "choices": []}

    def arrival(rank, count):
        x = count / 1024
        a, b = rank["arrival"]
        return max(0.0, a + b * x)

    def build(rank, count):
        x = count / 1024
        a, b, c = rank["build"]
        return max(0.0, a + b * x + c * x * x)

    def extend(rank, count):
        x = count / 1024
        a, b, c = rank["extend"]
        return max(0.0, a + b * x + c * x * x)

    full = max(
        arrival(rank, prompt_tokens) + build(rank, prompt_tokens)
        for rank in profile["ranks"]
    )
    choices = []
    chunk = profile["config"]["prefill_chunk_tokens"]
    for prefix in range(chunk, prompt_tokens, chunk):
        tail = prompt_tokens - prefix
        if prefix > domain["max_prefix_tokens"] or tail < domain["min_tail_tokens"]:
            continue
        ready = max(
            max(
                arrival(rank, prefix) + build(rank, prefix),
                arrival(rank, prompt_tokens),
            ) + extend(rank, tail)
            for rank in profile["ranks"]
        )
        choices.append({"prefix": prefix, "both_ready_seconds": ready})
    choices.sort(key=lambda item: item["both_ready_seconds"])
    best = choices[0] if choices else None
    gain = full - best["both_ready_seconds"] if best else 0.0
    selected = best["prefix"] if best and gain > profile["uncertainty_seconds"] else 0
    return {
        "selected_prefix": selected,
        "best_prefix": best["prefix"] if best else 0,
        "predicted_gain_seconds": gain,
        "full_ready_seconds": full,
        "choices": choices,
    }
