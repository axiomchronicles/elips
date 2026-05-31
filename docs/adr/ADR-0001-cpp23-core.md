# ADR-0001: C++23 as the core language

**Status:** Accepted
**Date:** 2024-08-01

## Context
ELIPS needs predictable, allocation-controlled performance for SIMD distance
kernels and graph traversal, plus first-class embeddability in both C++ and
Python processes with no runtime.

## Decision
Implement the core in C++23. Use modern facilities (`std::span`, `std::variant`,
`std::expected`-style error types via exceptions, `<=>`, concepts where useful)
and the C++ Core Guidelines (RAII, value semantics, immutability by default).

## Consequences
- Zero-overhead abstractions and direct SIMD access.
- Trivial embedding as a static/shared library and as a Python extension.
- Requires a recent toolchain (Clang 17+/GCC 13+).

## Alternatives Considered
- **Rust:** excellent safety, but a heavier interop story for a drop-in C++ SDK.
- **C:** maximal portability, but loses RAII and type-safety guarantees.
