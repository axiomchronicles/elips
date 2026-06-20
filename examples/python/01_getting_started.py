"""ELIPS v2 getting started with the modern Python API.

Run from the repo root after building the bindings:
    cmake -S . -B build -G Ninja -DELIPS_BUILD_PYTHON=ON
    cmake --build build --target elips_pymodule
    PYTHONPATH=bindings/python python3 examples/python/01_getting_started.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import elips


def build_chunk(key: str, ordinal: int, start: int, end: int) -> elips.ChunkInfo:
    chunk = elips.ChunkInfo()
    chunk.document_key = key
    chunk.ordinal = ordinal
    chunk.char_start = start
    chunk.char_end = end
    return chunk


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "demo"

        engine = elips.connect(
            str(db_path),
            dimension=2,
            metric="cosine",
            segmented_storage=True,
            metadata_acceleration=True,
        )
        arena = engine.arena("documents")
        embedder = engine.raw.config.text_embedder_info
        print(f"text embedder: {embedder.provider}/{embedder.model}@{embedder.revision}")

        arena.write_many(
            [
                elips.RecordInput(
                    text="alpha design note",
                    meta={"kind": "design"},
                    chunk=build_chunk("doc-alpha", 0, 0, 17),
                ),
                {
                    "text": "beta incident runbook",
                    "meta": {"kind": "ops"},
                    "chunk": build_chunk("doc-beta", 0, 0, 21),
                },
            ]
        )

        print("text probe:")
        for hit in arena.probe_text("alpha", top=2, include_vectors=True):
            print(f"  {hit.text!r} distance={hit.distance:.3f} meta={hit.meta}")

        print("\nhybrid probe:")
        for hit in arena.probe_hybrid([0.0, 1.0], "alpha", top=2):
            print(f"  {hit.text!r} distance={hit.distance:.3f}")

        plan = arena.explain(
            [1.0, 0.0],
            top=1,
            where=elips.Filter().field("kind").equals("design"),
            has_text_component=True,
        )
        print(
            f"\nplanner: strategy={plan.strategy.name} "
            f"metadata_accelerated={plan.metadata_accelerated}"
        )

        engine.checkpoint()
        engine.compact()
        engine.close()

        with elips.open(str(db_path), access_mode="read_only") as reader:
            docs = reader.vault("documents")
            print("\nread-only reopen:")
            print(" ", reader.list_vaults())
            print(" ", docs.seek_text("beta", top=1)[0].document.text)


if __name__ == "__main__":
    main()
