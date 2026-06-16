"""End-to-end ELIPS GPU showcase.

Run from the repo root after building the Python module:

    PYTHONPATH=bindings/python python3 bindings/python/test.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import elips


def section(title: str) -> None:
    print(f"\n== {title} ==")


def toy_embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        vectors.append(
            [
                1.0 if "alpha" in lowered else 0.0,
                1.0 if "beta" in lowered else 0.0,
            ]
        )
    return vectors


def make_chunk(key: str, ordinal: int, start: int, end: int) -> elips.ChunkInfo:
    chunk = elips.ChunkInfo()
    chunk.document_key = key
    chunk.ordinal = ordinal
    chunk.char_start = start
    chunk.char_end = end
    return chunk


def make_lineage(provider: str, stage: str) -> elips.EmbeddingLineage:
    lineage = elips.EmbeddingLineage()
    lineage.provider = provider
    lineage.model = "toy-embed-v1"
    lineage.revision = "2026-06-16"
    lineage.attributes = {"stage": stage}
    return lineage


def build_gpu_config() -> elips.GpuConfig:
    if not elips._has_gpu:
        raise RuntimeError("ELIPS was built without GPU bindings")

    gpu = elips.GpuConfig()
    gpu.policy = elips.GpuPolicy.require_gpu
    gpu.build_mode = elips.IndexBuildMode.gpu_build_gpu_serve
    gpu.algorithm = elips.GpuIndexAlgorithm.brute_force
    gpu.precision = elips.GpuPrecision.auto
    gpu.device_memory_pool_mb = 256
    gpu.fp16_search = True
    gpu.unified_memory = True
    gpu.batch_window_us = 250
    gpu.max_batch_size = 128
    gpu.ef_search = 64
    gpu.profiling = True
    return gpu


def build_config() -> elips.Config:
    return (
        elips.Config()
        .dimension(2)
        .metric("cosine")
        .index("exact")
        .segmented_storage(True)
        .metadata_acceleration(True)
        .text_embedder(toy_embed, provider="demo", model="toy-embed-v1")
        .gpu(build_gpu_config())
    )


def print_result_hits(label: str, hits) -> None:
    print(label)
    for hit in hits:
        print(f"  id={hit.id} distance={hit.distance:.4f}")


def print_modern_hits(label: str, hits) -> None:
    print(label)
    for hit in hits:
        print(
            f"  key={hit.key} distance={hit.distance:.4f} "
            f"text={hit.text!r}"
        )


def verify_live_gpu() -> elips.GpuDeviceInfo:
    section("Module GPU Snapshot")
    module_device = elips.GpuDeviceInfo()
    print("compile_time_gpu_bindings:", elips._has_gpu)
    print("module_device:", module_device)
    if module_device.backend == "cpu":
        raise RuntimeError(
            "This showcase requires a live GPU backend. "
            "ELIPS reported CPU fallback instead."
        )
    return module_device


def run_raw_core_worker() -> None:
    module_device = verify_live_gpu()

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "gpu_raw_core"
        config = build_config()

        section("Open GPU Database")
        db = elips.open_with_config(str(db_path), config)
        docs = db.vault("documents")
        tasks = db.vault("tasks")

        db_device = db.gpu_info()
        print("db_device:", db_device)
        print("db_gpu_stats:", db.gpu_stats())
        assert db_device.backend == module_device.backend
        assert db_device.backend != "cpu"

        section("Transactions And CRUD")
        with db.begin_transaction() as txn:
            tx_docs = txn.vault("documents")
            tx_docs.place([1.0, 0.0], {"kind": "design", "priority": 1})
            tx_docs.place([0.0, 1.0], {"kind": "ops", "priority": 2})

        inline_id = docs.place(
            [0.6, 0.8],
            {"kind": "reference", "priority": 3, "owner": "platform"},
            document=elips.DocumentAttachment(
                "alpha beta reference",
                uri="memory://documents/reference",
            ),
            chunk=make_chunk("doc-inline", 0, 0, 20),
            lineage=make_lineage("manual", "inline-place"),
        )
        docs.place_many(
            [
                {
                    "vector": [0.9, 0.1],
                    "data": {"kind": "design", "priority": 4},
                    "document": elips.DocumentAttachment(
                        "alpha architecture review",
                        uri="memory://documents/batch-a",
                    ),
                    "chunk": make_chunk("doc-batch-a", 0, 0, 25),
                    "lineage": make_lineage("batch", "vector-record"),
                },
                {
                    "text": "beta escalation playbook",
                    "data": {"kind": "ops", "priority": 5},
                    "chunk": make_chunk("doc-batch-b", 0, 0, 24),
                    "lineage": make_lineage("batch", "text-record"),
                },
            ]
        )

        fetched = docs.fetch(inline_id)
        assert fetched is not None
        assert fetched["document"].text == "alpha beta reference"
        print("documents_count:", docs.count())
        print("documents_info:", docs.info())

        section("Vector, Filter, Text, And Hybrid Search")
        design_filter = elips.Filter().field("kind").equals("design")
        priority_filter = elips.Filter().field("priority").gte(2)

        vector_hits = docs.seek([1.0, 0.0], top=3, where=design_filter)
        text_hits = docs.seek_text("alpha", top=3)
        hybrid_hits = docs.seek_hybrid(
            [0.0, 1.0],
            "beta",
            top=3,
            where=priority_filter,
            lexical_weight=0.40,
        )

        print_result_hits("vector_hits:", vector_hits)
        print_result_hits("text_hits:", text_hits)
        print_result_hits("hybrid_hits:", hybrid_hits)

        plan = docs.explain_seek(
            [1.0, 0.0],
            top=3,
            where=design_filter,
            has_text_component=True,
        )
        print(
            "plan:",
            {
                "strategy": plan.strategy.name,
                "candidate_count": plan.candidate_count,
                "gpu_index": plan.gpu_index,
                "index_type": plan.index_type,
                "metadata_accelerated": plan.metadata_accelerated,
            },
        )
        assert plan.gpu_index is True
        assert plan.index_type.startswith("gpu_")

        section("EQL And Utilities")
        eql = "seek in tasks nearest $q top 3 where stage >= 1 yield"
        elips.validate_eql(eql)
        tokens = elips.tokenize_eql(eql)
        print("first_tokens:", [token.text for token in tokens[:6]])
        print(
            "cosine_distance(alpha, beta):",
            elips.distance("cosine", [1.0, 0.0], [0.0, 1.0]),
        )

        inserted = db.query(
            'place in tasks vector [1.0, 0.0] data {"name": "alpha task", "stage": 1}'
        )
        db.query(
            'place in tasks vector [0.0, 1.0] data {"name": "beta task", "stage": 2}'
        )
        inserted_id = inserted[0].id
        eql_hits = db.query(eql, {"q": [1.0, 0.0]})
        eql_fetch = db.query(f'fetch from tasks id "{inserted_id}" yield')
        eql_scan = db.query("scan in tasks where stage >= 1 limit 10 yield")
        print_result_hits("eql_hits:", eql_hits)
        print("eql_fetch_count:", len(eql_fetch))
        print("eql_scan_count:", len(eql_scan))
        db.query(f'erase from tasks id "{inserted_id}"')
        assert tasks.fetch(inserted_id) is None

        db.close()


def run_raw_persistence_worker() -> None:
    verify_live_gpu()

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "gpu_raw_persistence"
        config = build_config()
        db = elips.open_with_config(str(db_path), config)
        docs = db.vault("documents")
        docs.place_document("alpha design note", {"kind": "design"})
        docs.place_document("beta incident runbook", {"kind": "ops"})

        section("Maintenance And Reopen")
        docs.rebuild_index()
        db.checkpoint()
        db.compact()
        db.close()

        with elips.open(str(db_path), access_mode="read_only") as reader:
            reader_docs = reader.vault("documents")
            reader_hits = reader_docs.seek_text("beta", top=2)
            print("vaults:", reader.list_vaults())
            print("reader_gpu_info:", reader.gpu_info())
            print_result_hits("reader_hits:", reader_hits)
            assert reader_hits


def run_modern_document_worker() -> None:
    section("Modern Document Session")
    verify_live_gpu()
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "modern_document")
        config = build_config()
        engine = elips.connect_with_config(
            db_path,
            config,
            embedder=toy_embed,
            embedder_provider="demo",
            embedder_model="toy-embed-v1",
        )
        arena = engine.arena("arena_docs")
        keys = arena.ingest(
            texts=["alpha launch brief", "beta incident review"],
            meta=[{"team": "product"}, {"team": "ops"}],
            chunks=[
                make_chunk("arena-alpha", 0, 0, 18),
                make_chunk("arena-beta", 0, 0, 20),
            ],
            lineages=[
                make_lineage("arena", "ingest-alpha"),
                make_lineage("arena", "ingest-beta"),
            ],
        )

        pulled = arena.pull(keys, include_vectors=True)
        probe_hits = arena.probe([1.0, 0.0], top=3, include_vectors=True)
        probe_text_hits = arena.probe_text("alpha", top=2, include_vectors=True)

        print("engine_gpu_info:", engine.raw.gpu_info())
        print("pulled_rows:", len(pulled))
        print("arena_count:", arena.count())
        print_modern_hits("arena_probe_hits:", probe_hits)
        print_modern_hits("arena_probe_text_hits:", probe_text_hits)
        engine.close()


def run_modern_hybrid_worker() -> None:
    section("Modern Hybrid Session")
    verify_live_gpu()
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "modern_hybrid")
        config = build_config()
        engine = elips.connect_with_config(
            db_path,
            config,
            embedder=toy_embed,
            embedder_provider="demo",
            embedder_model="toy-embed-v1",
        )
        arena = engine.arena("arena_docs")
        arena.ingest(
            texts=["alpha launch brief", "beta incident review"],
            meta=[{"team": "product"}, {"team": "ops"}],
        )
        probe_hybrid_hits = arena.probe_hybrid(
            [0.0, 1.0],
            "beta",
            top=3,
            include_vectors=True,
        )
        plan = arena.explain(
            [1.0, 0.0],
            top=1,
            where=elips.Filter().field("team").equals("product"),
            has_text_component=True,
        )

        print_modern_hits("arena_probe_hybrid_hits:", probe_hybrid_hits)
        print(
            "arena_plan:",
            {
                "strategy": plan.strategy.name,
                "candidate_count": plan.candidate_count,
                "gpu_index": plan.gpu_index,
                "index_type": plan.index_type,
                "metadata_accelerated": plan.metadata_accelerated,
            },
        )
        engine.close()


def run_modern_merge_worker() -> None:
    section("Modern Merge Session")
    verify_live_gpu()
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "modern_merge")
        config = build_config()
        engine = elips.connect_with_config(
            db_path,
            config,
            embedder=toy_embed,
            embedder_provider="demo",
            embedder_model="toy-embed-v1",
        )
        arena = engine.arena("arena_merge")
        key = arena.write(
            vector=[1.0, 0.0],
            text="alpha draft",
            meta={"team": "drafts"},
        )
        arena.merge(
            vectors=[[0.0, 1.0]],
            texts=["beta release checklist"],
            meta=[{"team": "release"}],
            keys=[key],
        )

        merged_rows = arena.pull([key], include_vectors=True)
        merged_hits = arena.probe([0.0, 1.0], top=5)
        print("merged_rows:", len(merged_rows))
        print_modern_hits("merged_hits:", merged_hits)

        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ELIPS GPU showcase")
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "raw",
            "modern",
            "raw_core",
            "raw_persistence",
            "modern_document",
            "modern_hybrid",
            "modern_merge",
        ),
        default="all",
        help="which showcase to run",
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_raw_core_worker()
        run_raw_persistence_worker()
        run_modern_document_worker()
        run_modern_hybrid_worker()
        run_modern_merge_worker()
        section("Completed")
        print("All GPU showcases finished successfully.")
        return

    if args.mode == "raw":
        run_raw_core_worker()
        run_raw_persistence_worker()
        section("Completed")
        print("Raw GPU showcases finished successfully.")
        return

    if args.mode == "modern":
        run_modern_document_worker()
        run_modern_hybrid_worker()
        run_modern_merge_worker()
        section("Completed")
        print("Modern GPU showcases finished successfully.")
        return

    if args.mode == "raw_core":
        run_raw_core_worker()
        section("Completed")
        print("Raw core session finished successfully.")
        return

    if args.mode == "raw_persistence":
        run_raw_persistence_worker()
        section("Completed")
        print("Raw persistence session finished successfully.")
        return

    if args.mode == "modern_document":
        run_modern_document_worker()
        section("Completed")
        print("Modern document session finished successfully.")
        return

    if args.mode == "modern_hybrid":
        run_modern_hybrid_worker()
        section("Completed")
        print("Modern hybrid session finished successfully.")
        return

    run_modern_merge_worker()
    section("Completed")
    print("Modern merge session finished successfully.")


if __name__ == "__main__":
    main()
