"""Independent real tiny SGLang draft in the actual-Q/search/decode CPU loop.

Random local model/tokenizer fixtures only. No checkpoint choice, GPU support,
quality, loading-time budget enforcement or latency-hiding claim.
"""

import copy
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace

import torch


def validate_real_draft_loop(target, port):
    from pvd_batch_decode_smoke import validate_batch_decode
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_hf import VocabularySignature
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import (
        SGLangDraftRunnerFactory,
    )
    from sglang.srt.disaggregation.pvd.draft_sglang import (
        DraftCapabilities,
        DraftPlacement,
        SGLangDraftProvider,
    )
    from sglang.srt.disaggregation.pvd.prediction import DraftConfig
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import (
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GenerationConfig, LlamaConfig, PreTrainedTokenizerFast

    def tensors(runner):
        return [
            *runner.model.parameters(),
            *runner.model.buffers(),
            runner.req_to_token_pool.req_to_token,
            *runner.token_to_kv_pool.k_buffer,
            *runner.token_to_kv_pool.v_buffer,
        ]

    def capacity(runner):
        return (
            len(runner.req_to_token_pool.free_slots),
            runner.token_to_kv_pool_allocator.available_size(),
        )

    def worker(runner):
        return SimpleNamespace(
            get_memory_pool=lambda: (
                runner.req_to_token_pool,
                runner.token_to_kv_pool_allocator,
            ),
            model_config=runner.model_config,
            device=runner.device,
        )

    global_args = get_global_server_args()
    with tempfile.TemporaryDirectory(prefix="pvd-independent-draft-") as directory:
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            architectures=["LlamaForCausalLM"],
            tie_word_embeddings=False,
        )
        config.save_pretrained(directory)
        GenerationConfig(bos_token_id=1, eos_token_id=2).save_pretrained(directory)
        args = copy.deepcopy(target.server_args)
        args.model_path = args.tokenizer_path = directory
        args.max_running_requests = 1
        # ModelRunner construction sets process-global server args. Isolate this
        # fixture's initialization and RNG, restore the target before execution.
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9128)
                draft = ModelRunner(
                    ModelConfig.from_server_args(args),
                    mem_fraction_static=args.mem_fraction_static,
                    gpu_id=0,
                    tp_rank=0,
                    tp_size=1,
                    moe_ep_rank=0,
                    moe_ep_size=1,
                    pp_rank=0,
                    pp_size=1,
                    nccl_port=port,
                    server_args=args,
                    is_draft_worker=True,
                )
                with torch.no_grad():
                    for name, parameter in draft.model.named_parameters():
                        if "norm" in name:
                            parameter.fill_(1.0)
                        else:
                            parameter.normal_(mean=0.0, std=0.12)
        finally:
            set_global_server_args_for_scheduler(global_args)

        vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}
        vocab.update({f"token{i}": i for i in range(3, 64)})
        tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )
        signature = VocabularySignature.from_tokenizer(tokenizer)
        adapter = DraftForwardAdapter(
            draft,
            architecture="LlamaForCausalLM",
            attention_backend="torch_native",
            bytes_per_token=64,
            device="cpu",
            transient_bytes_bound=1 << 20,
        )
        allocator = PrivatePoolAllocator(
            draft.req_to_token_pool, draft.token_to_kv_pool_allocator
        )
        # Explicit retained tensor footprint, not a claim to account the entire
        # runtime's allocator or peak model-loading workspace.
        storage = {}
        for tensor in tensors(draft) + [
            draft.token_to_kv_pool_allocator.free_pages,
            draft.token_to_kv_pool_allocator.release_pages,
        ]:
            s = tensor.untyped_storage()
            storage[s.data_ptr()] = s.nbytes()
        retained = sum(storage.values())
        factory = SGLangDraftRunnerFactory(
            adapter,
            allocator,
            capabilities=DraftCapabilities(
                architectures=("LlamaForCausalLM",),
                attention_backends=("torch_native",),
                max_prefix_tokens=32,
                max_predict_tokens=2,
            ),
            persistent_bytes=retained,
            max_tokens=2,
        )
        before = capacity(draft)
        predictions = []

        class AuditedProvider(SGLangDraftProvider):
            @contextmanager
            def branch(self):
                original = [t.clone() for t in tensors(target)]
                rng = torch.get_rng_state().clone()
                target_capacity = capacity(target)
                with super().branch():
                    yield self
                for actual, saved in zip(tensors(target), original, strict=True):
                    torch.testing.assert_close(
                        actual, saved, rtol=0, atol=0, equal_nan=True
                    )
                assert torch.equal(rng, torch.get_rng_state())
                assert capacity(target) == target_capacity
                assert capacity(draft) == before
                assert self.active_branches == 0 and not self.degraded
                assert self.scratch_budget.snapshot()["used_staging_bytes"] == 0

            def predict(self, prefix, max_tokens):
                result = super().predict(prefix, max_tokens)
                predictions.append((prefix.request_id, result.tokens))
                return result

        provider = AuditedProvider(
            DraftConfig(directory, device="cpu", dtype="float32", predict_tokens=2),
            DraftPlacement(
                scratch_budget_bytes=2 << 20, persistent_budget_bytes=retained
            ),
            factory,
            worker=worker(draft),
            target_worker=worker(target),
            draft_vocabulary=signature,
            target_vocabulary=signature,
        )
        assert provider.pool_ownership.storage_verified
        evidence = validate_batch_decode(
            target, scheduled_results=True, draft_provider=provider
        )
        assert len(predictions) == 1 and predictions[0][0] == "old"
        assert (
            adapter.forward_count == 2
        )  # prefill + one continuation for two predictions
        assert capacity(draft) == before
        assert get_global_server_args() is global_args
        assert provider.persistent_budget.snapshot()["used_staging_bytes"] == retained
        evidence.update(
            draft_forwards=adapter.forward_count,
            draft_predictions=predictions,
            target_state_rng_unchanged_during_draft=True,
            private_backing_storage_verified=True,
            draft_pool_capacity_restored=True,
            committed_boundary_fallback_did_not_call_draft=True,
            draft_retained_tensor_bytes=retained,
            draft_persistent_charge_lifetime="until this standalone test process exits",
            fixture="two independent random Llamas and a shared toy tokenizer",
            model_quality_gpu_latency_validated=False,
        )
        return evidence
