# Changelog

All notable changes to the `elips` Python package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-22

### Added
- Native C++23 core compilation via CMake integration in `setup.py`.
- Modern Python facades: `connect()`, `Engine`, `Arena`, `Hit`, `Row`.
- Built-in local text embedder support with automatic default provisioning.
- HNSW and Exact vector indices support.
- Scoped query and transaction facades with auto-commit.
- Metadata filters and acceleration.
- WAL durability configurations (paranoid, standard, relaxed, ephemeral).
- Apple Metal GPU acceleration support on macOS.
- Complete Python packaging metadata (`pyproject.toml`, `setup.py`).
- Automatic CI linting with Ruff and build verification.
