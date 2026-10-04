"""Consecutive logical token ranges in an already validated manifest order.

Source KV heads can be interleaved: these ranges permit strided tensor copies,
not contiguous original-Entry RDMA reads. No token is inserted or reordered.
"""


def consecutive_token_runs(token_ids):
    start = 0
    for stop in range(1, len(token_ids) + 1):
        if stop == len(token_ids) or token_ids[stop] != token_ids[stop - 1] + 1:
            yield start, token_ids[start], stop - start
            start = stop
