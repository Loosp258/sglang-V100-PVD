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

    def prefix_for(self, prompt_tokens: int) -> int:
        """Uncalibrated lengths use the exact full-graph path."""
        return self.choices.get(prompt_tokens, 0)
