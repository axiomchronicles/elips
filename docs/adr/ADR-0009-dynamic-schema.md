# ADR-0009: Dynamic metadata schema

**Status:** Accepted
**Date:** 2024-08-01

## Context
Workloads attach heterogeneous metadata to vectors and evolve it over time.
Requiring an upfront schema adds friction.

## Decision
Metadata is a dynamic `Payload` (`map<string, variant<int64, double, bool,
string>>`). No schema declaration is required; keys are observed lazily. Filters
compare values by type (numeric cross-compare int/double; `contains` is a
substring match on string fields).

## Consequences
- Frictionless ingestion and schema evolution; new keys are always safe.
- Missing keys evaluate to "no match" in filters (null-like semantics).
- Columnar storage, attribute B-trees, and inverted indexes are deferred; v1.0
  evaluates filters over in-memory payloads.

## Alternatives Considered
- **Declared schema:** enables tighter encoding/validation, more upfront cost.
- **Opaque JSON blob:** simplest, but not filterable.
