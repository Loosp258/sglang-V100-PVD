"""Check whether a target final hidden state is finite for a Prompt prefix."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--attn", choices=("eager", "sdpa"), required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tokenizer.encode("Case 40. " + "EEFTRITON " * 220,
                           add_special_tokens=False)[:512]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16,
        attn_implementation=args.attn, local_files_only=True,
    ).to("cuda:0").eval()
    with torch.inference_mode():
        output = model.model(
            input_ids=torch.tensor(ids, device="cuda:0")[None],
            use_cache=False,
        )
    hidden = output.last_hidden_state[0, -1]
    print({"attn": args.attn, "tokens": len(ids),
           "finite": bool(torch.isfinite(hidden).all()),
           "nonfinite_values": int((~torch.isfinite(hidden)).sum())},
          flush=True)


if __name__ == "__main__":
    main()
