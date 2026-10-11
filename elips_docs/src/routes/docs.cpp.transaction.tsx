import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/cpp/transaction")({
  head: () => ({
    meta: [
      { title: "C++ Transactions — Atomic Multi-Collection Operations" },
      {
        name: "description",
        content:
          "Complete reference for C++ elips::Transaction and elips::TransactionCollection: pending operations, atomic two-phase commits, undo rollback logs, and exception safety.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="C++ API"
      title="C++ Transactions"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "transaction-class", label: "Transaction Class" },
        { id: "transaction-vault", label: "TransactionCollection Class" },
        { id: "write-protocol", label: "Two-Phase Write Protocol" },
        { id: "exception-safety", label: "RAII & Exception Safety" },
        { id: "code-example", label: "Complete Code Example" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS supports atomic, multi-collection transaction write batching via <code>elips::Transaction</code> and <code>elips::TransactionCollection</code>.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        Transactions guarantee Atomicity, Consistency, Isolation, and Durability (ACID) for write operations across multiple collections. Either all vector insertions and removals are committed to disk and memory atomically, or none are applied.
      </p>

      <h2 id="transaction-class">Transaction Class</h2>
      <CodeBlock lang="cpp">{`class Transaction {
public:
    explicit Transaction(Database& db);
    ~Transaction(); // Automatically rolls back if commit() was not called

    TransactionCollection collection(const std::string& name);
    void commit();
    void rollback() noexcept;
};`}</CodeBlock>

      <h2 id="transaction-vault">TransactionCollection Class</h2>
      <CodeBlock lang="cpp">{`class TransactionCollection {
public:
    RecordID add(
        const Vector& vector,
        Payload payload = {},
        std::optional<RecordID> id = std::nullopt
    );
    void remove(const RecordID& id);
};`}</CodeBlock>

      <h2 id="write-protocol">Two-Phase Write Protocol</h2>
      <p>
        When <code>txn.commit()</code> is invoked:
      </p>
      <ol>
        <li>
          <strong>Validation Phase:</strong> The transaction checks vault write permissions, validates vector dimensions, and prepares undo records.
        </li>
        <li>
          <strong>WAL Logging:</strong> All batched operations are written to the Write-Ahead Log (WAL) with a transaction sequence marker.
        </li>
        <li>
          <strong>Memory Application:</strong> Vectors are inserted into HNSW indices and payload stores across target collections.
        </li>
        <li>
          <strong>Rollback Recovery:</strong> If any stage fails (e.g. disk write failure), all in-memory changes are rolled back using internal undo buffers.
        </li>
      </ol>

      <h2 id="exception-safety">RAII & Exception Safety</h2>
      <p>
        <code>Transaction</code> follows RAII. If a <code>Transaction</code> goes out of scope before <code>commit()</code> completes successfully, its destructor automatically triggers <code>rollback()</code>.
      </p>

      <h2 id="code-example">Complete Code Example</h2>
      <CodeBlock lang="cpp">{`#include <elips/elips.hpp>
#include <iostream>

void atomic_transfer(elips::Database& db) {
    // Begin transaction handle
    auto txn = db.begin_transaction();

    try {
        // Enqueue operations across multiple collections
        auto users = txn.collection("users");
        auto logs = txn.collection("logs");

        elips::Vector user_vec = {0.1f, 0.2f, 0.3f, 0.4f};
        elips::Vector log_vec = {0.9f, 0.8f, 0.7f, 0.6f};

        users.add(user_vec, {{"user_id", std::string("u_100")}});
        logs.add(log_vec, {{"action", std::string("user_created")}});

        // Commit atomically
        txn.commit();
        std::cout << "Transaction committed successfully!" << std::endl;
    } catch (const elips::ElipsException& ex) {
        std::cerr << "Transaction failed: " << ex.what() << std::endl;
        txn.rollback();
    }
}`}</CodeBlock>
    </DocsShell>
  );
}
