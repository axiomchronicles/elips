import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/cpp/overview")({
  head: () => ({
    meta: [
      { title: "C++ API Overview — ELIPS Documentation" },
      {
        name: "description",
        content:
          "High-performance C++23 native vector engine API: core concepts, headers, RAII memory management, thread-safety, and CMake integration.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="C++ API"
      title="C++ Engine Overview"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "header-structure", label: "Header Structure" },
        { id: "core-classes", label: "Core Abstractions" },
        { id: "memory-raii", label: "Memory & RAII Semantics" },
        { id: "concurrency", label: "Concurrency & Thread Safety" },
        { id: "cmake-integration", label: "CMake Integration" },
        { id: "quick-example", label: "Quick Example" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS is implemented as a header-first C++23 library designed for embedded
        vector storage, zero-copy graph traversals, and hardware-accelerated
        hybrid search. The native C++ API provides direct, un-sandboxed access to
        all engine primitives.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        The native C++ API is optimized for high-throughput, low-latency applications
        where language binding overhead cannot be tolerated. Built with modern C++23,
        it relies on RAII ownership, move semantics, std::shared_mutex for concurrent
        reads, and zero-allocation query execution paths wherever possible.
      </p>

      <h2 id="header-structure">Header Structure</h2>
      <p>
        All public C++ headers reside under the <code>elips/</code> include path.
        Including <code>&lt;elips/elips.hpp&gt;</code> pulls in the full surface area of
        the engine.
      </p>
      <CodeBlock lang="cpp">{`// Master header providing ElipsInstance, Vault, Transaction, and Domain types
#include <elips/elips.hpp>

// Specific subsystem headers (included automatically by elips.hpp)
#include <elips/Config.hpp>
#include <elips/domain/Record.hpp>
#include <elips/domain/SearchResult.hpp>
#include <elips/domain/Vector.hpp>
#include <elips/metadata/Filter.hpp>`}</CodeBlock>

      <h2 id="core-classes">Core Abstractions</h2>
      <p>
        The C++ surface is organized around five primary classes:
      </p>
      <ul>
        <li>
          <Link to="/docs/cpp/elips-instance"><code>elips::ElipsInstance</code></Link> — Top-level database handle managing storage, WAL, locking, and multi-vault registries.
        </li>
        <li>
          <Link to="/docs/cpp/vault"><code>elips::Vault</code></Link> — A named partition owning vector graph indices, metadata inverted indices, and record stores.
        </li>
        <li>
          <Link to="/docs/cpp/domain"><code>elips::Record</code> & <code>Vector</code></Link> — Domain primitives representing vectors, payloads, and search results.
        </li>
        <li>
          <Link to="/docs/cpp/config"><code>elips::Config</code></Link> — Fluent configuration specifying dimensions, distance metrics, HNSW parameters, and durability policies.
        </li>
        <li>
          <Link to="/docs/cpp/transaction"><code>elips::Transaction</code></Link> — Multi-vault atomic transactional write batching with automatic rollback capability.
        </li>
      </ul>

      <h2 id="memory-raii">Memory & RAII Semantics</h2>
      <p>
        All resource ownership in ELIPS C++ API strictly follows Resource Acquisition Is Initialization (RAII).
      </p>
      <ul>
        <li>
          <code>elips::open()</code> returns a <code>std::unique_ptr&lt;ElipsInstance&gt;</code>.
        </li>
        <li>
          When <code>ElipsInstance</code> is destroyed, it flushes pending WAL logs, executes a database checkpoint (for persistent databases), and releases file lock handles.
        </li>
        <li>
          <code>Vault</code> objects are owned by their parent <code>ElipsInstance</code> and returned by reference.
        </li>
      </ul>

      <h2 id="concurrency">Concurrency & Thread Safety</h2>
      <p>
        ELIPS enforces multi-level concurrency protection:
      </p>
      <ul>
        <li>
          <strong>Process Isolation:</strong> An advisory <code>flock</code> lock manager prevents multiple processes from opening the same disk database in Read-Write mode simultaneously.
        </li>
        <li>
          <strong>Thread Safety:</strong> <code>ElipsInstance</code> uses a recursive mutex for vault registry lookups. Each <code>Vault</code> contains a <code>std::shared_mutex</code> allowing arbitrary concurrent readers (e.g. <code>seek</code> calls across threads) while serializing mutations (<code>place</code>, <code>erase</code>).
        </li>
      </ul>

      <h2 id="cmake-integration">CMake Integration</h2>
      <p>
        Link against ELIPS in your <code>CMakeLists.txt</code> using the target <code>elips::elips</code>:
      </p>
      <CodeBlock lang="cmake">{`cmake_minimum_required(VERSION 3.22)
project(my_vector_app CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(elips REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE elips::elips)`}</CodeBlock>

      <h2 id="quick-example">Quick Example</h2>
      <CodeBlock lang="cpp">{`#include <elips/elips.hpp>
#include <iostream>

int main() {
    // 1. Configure a 128-dimensional Cosine HNSW database
    elips::Config config;
    config.dimension = 128;
    config.metric = elips::Metric::cosine;
    config.index_type = elips::IndexType::hnsw;

    // 2. Open an in-memory database instance
    auto db = elips::open(":memory:", config);
    auto& vault = db->vault("embeddings");

    // 3. Place a vector with metadata payload
    elips::Vector vec(128, 0.5f);
    elips::Payload meta;
    meta["category"] = std::string("finance");
    meta["year"] = int64_t(2026);

    elips::RecordID id = vault.place(vec, meta);
    std::cout << "Inserted record ID: " << id << std::endl;

    // 4. Perform top-K similarity search
    auto results = vault.seek(vec, 5);
    for (const auto& hit : results) {
        std::cout << "Hit ID: " << hit.id << " Score: " << hit.score << std::endl;
    }

    return 0;
}`}</CodeBlock>
    </DocsShell>
  );
}
