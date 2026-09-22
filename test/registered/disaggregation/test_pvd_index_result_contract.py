"""Backend substitution must not weaken logical selection's result contract."""

import pytest
import torch
from sglang.srt.disaggregation.pvd.index_search import (
    BruteForceIndexBackend,
    IdMapping,
    IndexSearchError,
    select,
)


class ResultBackend(BruteForceIndexBackend):
    def __init__(self, result, *, device="cpu"):
        super().__init__(device=device)
        self.result = result
        self.calls = 0

    def search(self, *args, **kwargs):
        self.calls += 1
        return self.result


def select_result(result, *, device="cpu", top_k=2, queries=None):
    backend = ResultBackend(result, device=device)
    index = BruteForceIndexBackend().build(
        torch.eye(4), vector_space="test", metric="ip"
    )
    return select(
        backend,
        index,
        torch.ones(1, 4) if queries is None else queries,
        layer=0,
        kv_head=0,
        mapping=IdMapping("map", (10, 11, 12, 13), 2),
        top_k=top_k,
    )


@pytest.mark.parametrize(
    "rows,scores,message",
    [
        (torch.tensor([[0.5, 1.0]]), torch.ones(1, 2), "integer"),
        (torch.tensor([[False, True]]), torch.ones(1, 2), "integer"),
        (torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.int32), "floating"),
        (torch.tensor([[0, 1]]), torch.ones(1, 1), "shape"),
        (torch.tensor([0, 1]), torch.ones(2), "shape"),
        (torch.tensor([[0, 1], [2, 3]]), torch.ones(2, 2), "shape"),
        (torch.tensor([[0, 1]]), torch.tensor([[1.0, float("nan")]]), "finite"),
        (torch.tensor([[0, 1]]), torch.tensor([[1.0, float("inf")]]), "finite"),
        (torch.tensor([[0, 1]]), torch.tensor([[1.0, -float("inf")]]), "finite"),
        (torch.tensor([[0, 0]]), torch.ones(1, 2), "duplicate"),
        (torch.tensor([[0, -1]]), torch.ones(1, 2), "outside"),
        (torch.tensor([[0, 4]]), torch.ones(1, 2), "outside"),
        ([[0, 1]], torch.ones(1, 2), "tensors"),
        (torch.tensor([[0, 1]]), [[1.0, 2.0]], "tensors"),
    ],
)
def test_malformed_results_are_refused_before_mapping(rows, scores, message):
    with pytest.raises(IndexSearchError, match=message):
        select_result((rows, scores))


@pytest.mark.parametrize("result", [None, (), (torch.zeros(1),)])
def test_wrong_return_arity_is_a_contract_error(result):
    with pytest.raises(IndexSearchError, match="rows, scores"):
        select_result(result)


def test_output_device_must_match_backend_declaration_without_gpu_execution():
    with pytest.raises(IndexSearchError, match="device"):
        select_result((torch.tensor([[0, 1]]), torch.ones(1, 2)), device="meta")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.uint32])
def test_valid_results_keep_cross_query_union_and_best_score(dtype):
    selection = select_result(
        (
            torch.tensor([[2, 0], [1, 2]], dtype=dtype),
            torch.tensor([[0.3, 0.8], [0.9, 0.7]]),
        ),
        queries=torch.ones(2, 4),
    )
    assert selection.token_ids == (11, 10, 12)
    assert selection.scores == pytest.approx((0.9, 0.8, 0.7))
    assert selection.page_ids == (5, 6)
    assert (selection.layer, selection.kv_head) == (0, 0)


def test_exact_l2_score_is_negative_euclidean_not_negative_squared_distance():
    backend = BruteForceIndexBackend()
    index = backend.build(torch.tensor([[3.0, 4.0]]), vector_space="test", metric="l2")
    _, scores = backend.search(index, torch.zeros(1, 2), top_k=1)
    assert scores.item() == -5.0  # a future squared-L2 adapter must take sqrt


@pytest.mark.parametrize("top_k", [True, 0, 1.5, 5])
def test_invalid_top_k_cannot_be_accepted_by_a_lenient_backend(top_k):
    with pytest.raises(IndexSearchError, match="top_k"):
        select_result((torch.tensor([[0, 1]]), torch.ones(1, 2)), top_k=top_k)


@pytest.mark.parametrize("queries", [torch.empty(0, 4), torch.ones(1, 3), [1, 2]])
def test_invalid_query_shape_cannot_be_accepted_by_a_lenient_backend(queries):
    with pytest.raises(IndexSearchError, match="queries"):
        select_result((torch.tensor([[0, 1]]), torch.ones(1, 2)), queries=queries)
