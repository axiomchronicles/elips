"""Regression coverage for ELIPS local/default text embedding workflows."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

import elips


def toy_embed(texts):
    rows = []
    for text in texts:
        lowered = text.lower()
        rows.append(
            [
                1.0 if "alpha" in lowered else 0.0,
                1.0 if "beta" in lowered else 0.0,
                1.0 if "database" in lowered else 0.0,
                1.0 if "python" in lowered else 0.0,
            ]
        )
    return rows


def test_default_local_embedder_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "default-local")

        db = elips.open(db_path, dimension=64, metric="cosine")
        assert db.config.has_text_embedder is True
        info = db.config.text_embedder_info
        assert info is not None
        assert info.kind == elips.TextEmbedderKind.local_builtin
        assert info.auto_attached is True
        assert info.rehydratable is True

        docs = db.vault("docs")
        first = docs.place_document("vector database handbook", {"kind": "db"})
        docs.place_document("python language guide", {"kind": "py"})
        db.checkpoint()
        db.close()

        assert os.path.exists(os.path.join(db_path, "TEXT_EMBEDDER.manifest"))
        assert os.path.exists(info.storage_path)

        reopened = elips.open(db_path)
        assert reopened.config.has_text_embedder is True
        reopened_info = reopened.config.text_embedder_info
        assert reopened_info is not None
        assert reopened_info.storage_path == info.storage_path
        assert reopened_info.auto_attached is True

        hit = reopened.vault("docs").seek_text("vector database", top=1)[0]
        assert hit.id == first
        reopened.close()

    print("  PASS test_default_local_embedder_roundtrip")


def test_disable_default_embedder_requires_vector_first():
    db = elips.open(
        ":memory:",
        dimension=64,
        metric="cosine",
        use_default_text_embedder=False,
    )
    assert db.config.has_text_embedder is False

    docs = db.vault("docs")
    try:
        docs.place_document("alpha note")
        assert False, "place_document should require a text embedder"
    except elips.ConfigError as error:
        message = str(error)
        assert "place()" in message
        assert "local text embedder" in message

    docs.place([1.0] + [0.0] * 63, document=elips.DocumentAttachment(text="vector note"))

    try:
        docs.seek_text("alpha", top=1)
        assert False, "seek_text should require a text embedder"
    except elips.ConfigError as error:
        assert "seek()" in str(error)

    print("  PASS test_disable_default_embedder_requires_vector_first")


def test_explicit_local_config_and_missing_artifact():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "explicit-local")
        artifact_path = os.path.join(td, "models", "default.localembed")

        local = elips.LocalEmbedderConfig(
            model="default",
            revision="v1",
            storage_path=artifact_path,
            dimension=64,
        )

        db = elips.open(
            db_path,
            dimension=64,
            metric="cosine",
            embedder=local,
            use_default_text_embedder=False,
        )
        info = db.config.text_embedder_info
        assert info is not None
        assert info.storage_path == artifact_path
        docs = db.vault("docs")
        docs.place_document("vector search reference", {"kind": "db"})
        db.close()

        os.remove(artifact_path)
        try:
            elips.open(db_path)
            assert False, "missing artifact should fail reopen"
        except elips.StorageError:
            pass

    print("  PASS test_explicit_local_config_and_missing_artifact")


def test_non_rehydratable_python_embedder_requires_explicit_reopen():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "python-embedder")

        db = elips.open(
            db_path,
            dimension=4,
            metric="cosine",
            embedder=toy_embed,
            embedder_provider="pytest",
            embedder_model="toy",
            embedder_revision="r1",
            use_default_text_embedder=False,
        )
        docs = db.vault("docs")
        docs.place_document("alpha database")
        db.close()

        reopened = elips.open(db_path)
        assert reopened.config.has_text_embedder is False
        info = reopened.config.text_embedder_info
        assert info is not None
        assert info.kind == elips.TextEmbedderKind.external
        assert info.provider == "pytest"
        assert info.model == "toy"
        assert info.rehydratable is False

        try:
            reopened.vault("docs").place_document("beta database")
            assert False, "reopen without explicit embedder should fail"
        except elips.ConfigError as error:
            assert "matching embedder" in str(error)
        reopened.close()

        explicit = elips.open(
            db_path,
            dimension=4,
            metric="cosine",
            embedder=toy_embed,
            embedder_provider="pytest",
            embedder_model="toy",
            embedder_revision="r1",
            use_default_text_embedder=False,
        )
        hit = explicit.vault("docs").seek_text("alpha", top=1)[0]
        assert hit.document.text == "alpha database"
        explicit.close()

    print("  PASS test_non_rehydratable_python_embedder_requires_explicit_reopen")


def test_modern_wrapper_uses_default_native_local_embedder():
    engine = elips.connect(":memory:", dimension=64, metric="cosine")
    assert engine.raw.config.has_text_embedder is True
    info = engine.raw.config.text_embedder_info
    assert info is not None
    assert info.kind == elips.TextEmbedderKind.local_builtin
    assert info.auto_attached is True

    arena = engine.arena("docs")
    keys = arena.ingest(
        texts=["vector database handbook", "python language guide"],
        meta=[{"kind": "db"}, {"kind": "py"}],
    )
    assert len(keys) == 2

    hit = arena.probe_text("vector database", top=1)[0]
    assert hit.text == "vector database handbook"
    assert hit.meta["kind"] == "db"
    engine.close()

    print("  PASS test_modern_wrapper_uses_default_native_local_embedder")


if __name__ == "__main__":
    tests = [
        test_default_local_embedder_roundtrip,
        test_disable_default_embedder_requires_vector_first,
        test_explicit_local_config_and_missing_artifact,
        test_non_rehydratable_python_embedder_requires_explicit_reopen,
        test_modern_wrapper_uses_default_native_local_embedder,
    ]

    for test in tests:
        test()

    print("All local embedder tests passed.")
