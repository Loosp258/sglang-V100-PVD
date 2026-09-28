"""Search equivalence and build avoidance for the isolated prefix experiment."""

from dataclasses import dataclass

import pytest
import torch

from benchmark.pvd_prefix_index_reuse import PrefixIndexReuseExperiment


@dataclass
class ExactIndex:
    handle: torch.Tensor
    count: int
    metric: str


class CountingExact:
    def __init__(self):
        self.build_calls = 0

    def build(self, vectors, *, vector_space, metric):
        self.build_calls += 1
        return ExactIndex(vectors.clone(), len(vectors), metric)

    def search(self, index, queries, *, top_k):
        if index.metric == "ip":
            scores = queries @ index.handle.T
        else:
            scores = -torch.cdist(queries, index.handle)
        ordered, rows = torch.sort(scores, dim=1, descending=True, stable=True)
        return rows[:, :top_k], ordered[:, :top_k]

    def synchronize(self):
        pass

    def dispose(self, index):
        pass


@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_shared_prefix_avoids_rebuild_and_matches_full_exact(metric):
    backend = CountingExact()
    experiment = PrefixIndexReuseExperiment(
        backend, vector_space="same-model", metric=metric
    )
    generator = torch.Generator().manual_seed(412)
    common = {
        (0, 0): torch.randn(20, 16, generator=generator),
        (0, 1): torch.randn(20, 16, generator=generator),
    }
    entries = []
    for _ in range(3):
        vectors = {
            head: torch.cat((prefix, torch.randn(7, 16, generator=generator)))
            for head, prefix in common.items()
        }
        entry, timing = experiment.build(
            vectors, prefix_rows=20, prefix_identity="shared-token-prefix"
        )
        entries.append(entry)
        assert timing.prefix_builds == (2 if len(entries) == 1 else 0)
        assert timing.tail_builds == 2
        for head, value in vectors.items():
            full = backend.build(value, vector_space="same-model", metric=metric)
            query = torch.randn(4, 16, generator=generator)
            full_rows, full_scores = backend.search(full, query, top_k=5)
            split_rows, split_scores = experiment.search(entry, head, query, top_k=5)
            assert torch.equal(split_rows, full_rows)
            torch.testing.assert_close(split_scores, full_scores)
    # Six reference full builds, two shared prefix builds, six tail builds.
    assert backend.build_calls == 14
    for entry in entries:
        experiment.close(entry)
    assert not experiment.groups


def test_same_identity_with_changed_k_refuses_reuse():
    backend = CountingExact()
    experiment = PrefixIndexReuseExperiment(backend, vector_space="model", metric="ip")
    base = {(0, 0): torch.arange(48, dtype=torch.float32).reshape(12, 4)}
    entry, _ = experiment.build(base, prefix_rows=8, prefix_identity="prefix")
    changed = {key: value.clone() for key, value in base.items()}
    changed[(0, 0)][3, 0] += 1
    with pytest.raises(ValueError, match="does not match stored K"):
        experiment.build(changed, prefix_rows=8, prefix_identity="prefix")
    assert backend.build_calls == 2
    experiment.close(entry)


def test_sequential_entries_reuse_idle_prefix_until_explicit_eviction():
    backend = CountingExact()
    experiment = PrefixIndexReuseExperiment(
        backend, vector_space="model", metric="ip", retain_idle=True
    )
    common = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    for tail_value in (1.0, 2.0, 3.0):
        vectors = {(0, 0): torch.cat((common, torch.full((3, 4), tail_value)))}
        entry, timing = experiment.build(
            vectors, prefix_rows=8, prefix_identity="sequential-prefix"
        )
        assert timing.prefix_builds == (1 if tail_value == 1.0 else 0)
        experiment.close(entry)
    assert backend.build_calls == 4
    assert experiment.groups["sequential-prefix"].users == 0
    experiment.evict("sequential-prefix")
    assert not experiment.groups


def test_identical_complete_prompt_reuses_graph_without_tail_index():
    backend = CountingExact()
    experiment = PrefixIndexReuseExperiment(
        backend, vector_space="model", metric="ip", retain_idle=True
    )
    vectors = {(0, 0): torch.randn(12, 4), (0, 1): torch.randn(12, 4)}
    query = torch.randn(2, 4)
    for expected_builds in (2, 0, 0):
        entry, timing = experiment.build(
            vectors, prefix_rows=12, prefix_identity="entire-prompt"
        )
        assert timing.prefix_builds == expected_builds
        assert timing.tail_builds == 0
        for head, value in vectors.items():
            rows, scores = experiment.search(entry, head, query, top_k=4)
            reference_scores = query @ value.T
            expected_scores, expected_rows = torch.sort(
                reference_scores, dim=1, descending=True, stable=True
            )
            assert torch.equal(rows, expected_rows[:, :4])
            torch.testing.assert_close(scores, expected_scores[:, :4])
        experiment.close(entry)
    assert backend.build_calls == 2
    experiment.evict("entire-prompt")
