---
library_name: transformers
license: apache-2.0
language:
  - en
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
  - eagle3
  - speculative-decoding
  - sglang
  - draft-model
  - jax
  - tpu
  - qwen2.5
pipeline_tag: text-generation
---

# EAGLE3 Draft Head — Qwen2.5-7B-Instruct

A speculative decoding draft head for [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), trained using the [EAGLE3](https://arxiv.org/abs/2503.01840) method on Google Cloud TPU with the [SpecJAX](https://github.com/tails-mpt/SpecJAX) framework.

EAGLE3 draft heads accelerate autoregressive generation by proposing multiple tokens per step that a target model then verifies in parallel — typically achieving 2-3x throughput gains with no change in output quality.

## Usage

### SGLang (GPU)

> **Note**: Qwen2.5 EAGLE3 support requires a small patch to SGLang (adding `set_eagle3_layers_to_capture()` to the Qwen2 model). See the [SpecJAX inference guide](https://github.com/tails-mpt/SpecJAX/tree/main/inference) for details.

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path thoughtworks/Qwen2.5-7B-Instruct-Eagle3 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --dtype bfloat16
```

### sglang-jax (TPU)

> **Note**: Requires the same Qwen2 EAGLE3 patch applied to sglang-jax. The sglang-jax EAGLE3 pipeline is functional but not yet performance-optimized.

```bash
python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path thoughtworks/Qwen2.5-7B-Instruct-Eagle3 \
    --speculative-eagle-topk 1 \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4 \
    --tp-size 4 --dtype bfloat16
```

### Python (SGLang client)

```python
import sglang as sgl

llm = sgl.LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    speculative_algorithm="EAGLE3",
    speculative_draft_model_path="thoughtworks/Qwen2.5-7B-Instruct-Eagle3",
    speculative_num_steps=5,
    speculative_eagle_topk=4,
    dtype="bfloat16",
)
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Framework | [SpecJAX](https://github.com/tails-mpt/SpecJAX) — pure JAX, no Flax/PyTorch |
| Hardware | Google Cloud TPU v4-32 (4 hosts x 4 chips, TP=4, DP=4) |
| Dataset | 54K mixed: ShareGPT (45%) + UltraChat-200K (35%) + Open-PerfectBlend (20%) |
| Epochs | 3 |
| Steps | 9,966 total |
| Optimizer | AdamW, cosine LR decay, 3% warmup |
| Learning rate | 8e-4 |
| Batch size | B=4, sequence length T=1024, gradient accumulation 4 |
| TTT length | 7 (multi-step speculative rollout) |
| Training time | ~5.5 hours |
| Precision | bfloat16 |

### Training Method

This model uses [EAGLE3](https://arxiv.org/abs/2503.01840)'s Test-Time Training (TTT) objective with a rollout length of 7. At each training step, the draft head autoregressively proposes 7 tokens; the target model provides ground-truth hidden states and logits for all positions; a geometric loss (0.8^k weighting) trains the draft to match the target at each position.

## Performance

Token acceptance rates on generic instruction-following data (ShareGPT-style prompts):

| Position | Acceptance Rate |
|----------|----------------|
| acc_0 (1st draft token) | **61.8%** |
| acc_1 | 56.9% |
| acc_2 | 54.6% |

*Measured on held-out evaluation data. Actual throughput gains depend on hardware, prompt distribution, and runtime version.*

## Model Architecture

The draft head is a single-layer transformer that operates on the target model's hidden states:

| Parameter | Value |
|-----------|-------|
| Architecture | `LlamaForCausalLM` (1 decoder layer) |
| Hidden size | 3584 |
| Attention heads | 28 (GQA: 4 KV heads) |
| Vocabulary size | 152,064 (full target vocab) |
| Draft vocab size | 32,000 (top tokens by training frequency) |
| Parameters | ~300M |

### Auxiliary Layer Indices

This head uses multi-layer feature fusion from layers `{1, 13, 24}` of the 28-layer Qwen2.5-7B model (SpecForge convention: `{1, L//2-1, L-4}`).

## Limitations

- Trained on English-dominant instruction data; performance may degrade on non-English inputs or highly domain-specific content.
- Acceptance rates are measured on generic chat data and will vary by prompt distribution.
- This is a v1 checkpoint trained on generic data. A v2 with target-model-regenerated training data is planned.

## License

This model is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), consistent with the base model's license.

## References

```bibtex
@article{li2025eagle3,
  title={EAGLE3: Scalable Speculative Decoding with Training-Free Multi-Draft Speculation},
  author={Li, Yuhui and Wei, Fangyun and Zhang, Chao and Zhang, Hongyang},
  journal={arXiv preprint arXiv:2503.01840},
  year={2025}
}
```
