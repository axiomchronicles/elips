"""The GIL must not serialize CPU-bound native work.

The C++ engine uses a `shared_mutex`: readers share, writers exclude. The
binding used to hold the GIL across every CPU-bound call, so no Python caller
could observe that concurrency -- searches from four threads ran 1.10x faster
than one thread, against a documented promise that a thread pool can share one
vault.

These tests assert the property rather than a speed. A throughput assertion
would be flaky on a loaded CI box, so the check is that real parallelism is
observable at all: measured CPU time across N threads must exceed what a
fully-GIL-serialized run could produce.
"""

from __future__ import annotations

import random
import threading
import time

import elips
import pytest

# Matched to the audit's measurement fixture. Smaller vectors make each seek
# so cheap that per-call binding overhead, not the search, dominates the
# timing -- which muddies the signal this test is looking for.
DIM = 256
COUNT = 6_000
QUERIES = 400
TOP_K = 50


@pytest.fixture(scope="module")
def populated_vault():
    rng = random.Random(7)
    db = elips.open(":memory:", dimension=DIM, metric="cosine")
    vault = db.vault("bench")
    for _ in range(COUNT):
        vault.place([rng.gauss(0.0, 1.0) for _ in range(DIM)])
    yield vault
    db.abandon()


def _search_across(vault, threads: int, queries: list[list[float]]):
    """Run `queries` split across `threads`, returning (wall, cpu) seconds."""

    barrier = threading.Barrier(threads)

    def worker(chunk: list[list[float]]) -> None:
        barrier.wait()  # start together so the threads actually overlap
        for query in chunk:
            vault.seek(query, TOP_K)

    workers = [
        threading.Thread(target=worker, args=(queries[i::threads],))
        for i in range(threads)
    ]

    wall0, cpu0 = time.perf_counter(), time.process_time()
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    return time.perf_counter() - wall0, time.process_time() - cpu0


def test_concurrent_searches_use_more_than_one_core(populated_vault):
    """CPU time must outrun wall time -- impossible while the GIL is held."""

    rng = random.Random(11)
    queries = [[rng.gauss(0.0, 1.0) for _ in range(DIM)] for _ in range(QUERIES)]

    _search_across(populated_vault, 1, queries[:20])  # warm up
    wall, cpu = _search_across(populated_vault, 4, queries)

    cores_busy = cpu / wall
    # A fully serialized binding pins this near 1.0 even with four threads
    # running (it measured 0.40 before the GIL was released). Anything
    # meaningfully above 1.0 proves the searches genuinely overlapped.
    assert cores_busy > 1.3, (
        f"searches appear serialized: {cores_busy:.2f} cores busy across 4 "
        f"threads (wall={wall:.3f}s cpu={cpu:.3f}s)"
    )


def test_a_slow_search_does_not_block_other_python_threads(populated_vault):
    """A native search must let unrelated Python code keep running."""

    rng = random.Random(13)
    queries = [[rng.gauss(0.0, 1.0) for _ in range(DIM)] for _ in range(QUERIES)]

    ticks = 0
    stop = threading.Event()

    def counter() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1

    pulse = threading.Thread(target=counter)
    pulse.start()
    try:
        for query in queries:
            populated_vault.seek(query, TOP_K)
    finally:
        stop.set()
        pulse.join()

    # With the GIL held across every seek, a pure-Python loop gets only the
    # interpreter's switch-interval slices. Releasing it lets the counter run
    # freely for the whole search duration.
    assert ticks > 0, "a pure-Python thread made no progress during searches"
