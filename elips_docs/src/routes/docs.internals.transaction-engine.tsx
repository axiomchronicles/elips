import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";
import { SketchCard, TransactionLifecycleDiagram } from "../components/SketchDiagram";

export const Route = createFileRoute("/docs/internals/transaction-engine")({
  head: () => ({
    meta: [
      { title: "Transaction engine — ELIPS Docs" },
      {
        name: "description",
        content:
          "How ELIPS buffers and commits atomic write batches — eager validation, RAII auto-rollback, and the single-writer isolation model.",
      },
      { property: "og:title", content: "Transaction engine — ELIPS" },
      {
        property: "og:description",
        content:
          "Transaction / TransactionVault, eager validation, commit and rollback, and the Python context-manager binding.",
      },
      { property: "og:url", content: "/docs/internals/transaction-engine" },
    ],
    links: [{ rel: "canonical", href: "/docs/internals/transaction-engine" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Internals"
      title="Transaction engine"
      toc={[
        { id: "shape", label: "Class shape" },
        { id: "pendingop", label: "PendingOp queue" },
        { id: "eager", label: "Eager validation" },
        { id: "ids", label: "RecordID pre-generation" },
        { id: "commit", label: "Commit" },
        { id: "rollback", label: "Rollback & RAII" },
        { id: "python", label: "Python context manager" },
        { id: "isolation", label: "Isolation level" },
        { id: "lifecycle", label: "Lifecycle" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS transactions are atomic write batches under the single-writer model.{" "}
        <code>Transaction</code> buffers <code>place</code> and <code>erase</code> calls; commit
        applies them in order, each one WAL-appended before the in-memory mutation. The destructor
        auto-rolls back if neither <code>commit()</code> nor <code>rollback()</code> ran.
      </p>

      <h2 id="shape">Class shape</h2>
      <CodeBlock lang="cpp">
        {`namespace elips {

class TransactionVault {
public:
    RecordID place(const Vector& vector, Payload payload = {},
                   std::optional<RecordID> id = std::nullopt);
    void     erase(const RecordID& id);

private:
    friend class Transaction;
    TransactionVault(Transaction& txn, std::string vault)
        : txn_(&txn), vault_(std::move(vault)) {}

    Transaction* txn_;
    std::string  vault_;
};

class Transaction {
public:
    explicit Transaction(ElipsInstance& db);
    ~Transaction();                          // auto-rolls back if !done_

    TransactionVault vault(std::string name);
    void commit();
    void rollback() noexcept;

private:
    void enqueue_place(std::string vault, const Vector&, Payload, std::optional<RecordID>);
    void enqueue_erase(std::string vault, RecordID);

    ElipsInstance*           db_;
    std::vector<PendingOp>   ops_;
    bool                     done_{false};
};

} // namespace elips`}
      </CodeBlock>

      <h2 id="pendingop">PendingOp queue</h2>
      <CodeBlock lang="cpp">
        {`struct PendingOp {
    bool                       is_erase{false};   // true = erase, false = place
    std::string                vault;             // target vault name
    Vector                     vector;            // empty for erase
    Payload                    payload;           // empty for erase
    std::optional<RecordID>    id;                // pre-generated or explicit
};`}
      </CodeBlock>
      <p>
        Inserts capture vector + payload + pre-generated id; erases capture only the target id. The
        buffer is <code>std::vector&lt;PendingOp&gt;</code> — operations apply in the order they
        were enqueued.
      </p>

      <h2 id="eager">Eager validation</h2>
      <CodeBlock lang="cpp">
        {`void Transaction::enqueue_place(std::string vault, const Vector& vector,
                                Payload payload, std::optional<RecordID> id) {
    if (vector.dimension() != db_->config().dimension()) {
        throw DimensionMismatch{"vector dimension does not match database"};
    }
    if (!all_finite(vector.values())) {
        throw InvalidVector{"vector contains NaN or Inf"};
    }
    ops_.push_back(PendingOp{false, std::move(vault), vector,
                             std::move(payload), std::move(id)});
}`}
      </CodeBlock>
      <p>
        Validating up front means <code>commit()</code> can never fail mid-batch on dimension or
        finiteness grounds — the only failure modes left are storage IO errors. Without this,
        atomicity would be impossible to promise: half the batch could end up applied before a bad
        vector tripped validation.
      </p>

      <h2 id="ids">RecordID pre-generation</h2>
      <CodeBlock lang="cpp">
        {`RecordID TransactionVault::place(const Vector& vector, Payload payload,
                                 std::optional<RecordID> id) {
    const RecordID assigned = id.value_or(RecordID::generate());
    txn_->enqueue_place(vault_, vector, std::move(payload), assigned);
    return assigned;          // returned before commit
}`}
      </CodeBlock>
      <p>
        Generating the UUIDv7 at enqueue time lets callers log or cross-reference the id during the
        transaction, even though the record is not yet visible to readers.
      </p>

      <h2 id="commit">Commit</h2>
      <CodeBlock lang="cpp">
        {`void Transaction::commit() {
    for (auto& op : ops_) {
        Vault& vault = db_->vault(op.vault);
        if (op.is_erase) {
            vault.erase(*op.id);                          // WAL + index + store
        } else {
            vault.place(op.vector, op.payload, op.id);    // prepare + WAL + index + store
        }
    }
    ops_.clear();
    done_ = true;
}`}
      </CodeBlock>
      <p>
        Each operation is WAL-appended individually before the in-memory mutation. A crash
        mid-commit is consistent: WAL replay reapplies every operation that made it to the log;
        everything past the crash is lost cleanly. Under the single-writer lock no other mutator can
        interleave.
      </p>

      <h2 id="rollback">Rollback &amp; RAII</h2>
      <CodeBlock lang="cpp">
        {`void rollback() noexcept { ops_.clear(); done_ = true; }

Transaction::~Transaction() {
    if (!done_) {
        rollback();   // discard buffered ops; none were applied
    }
}`}
      </CodeBlock>
      <p>
        Rollback is trivial — nothing was applied. The destructor calls it on any non-finalised
        transaction, so exceptions and early returns never leave a half-built batch lying around.
      </p>

      <h2 id="python">Python context manager</h2>
      <CodeBlock lang="python">
        {`with db.begin_transaction() as txn:
    v = txn.vault("data")
    v.place(vector1, {"tag": "a"})
    v.place(vector2, {"tag": "b"})
    # clean exit  → __exit__ calls commit()
    # raised exc  → __exit__ does NOT commit; destructor rolls back`}
      </CodeBlock>
      <p>
        The binding uses a <code>TransactionHolder</code> that keeps the Python{" "}
        <code>Database</code> reference alive for the transaction's lifetime, so the C++{" "}
        <code>ElipsInstance</code> outlives the <code>Transaction</code> as required.
      </p>

      <h2 id="isolation">Isolation level</h2>
      <p>
        Atomic batched writes under a single-writer lock — effectively <strong>serializable</strong>{" "}
        isolation. All transactions are serialised by the database lock. Reads within a transaction
        do <em>not</em> see uncommitted writes; <code>seek</code> always reads from the committed
        state.
      </p>
      <ul>
        <li>
          <strong>Atomicity</strong> — all or none, via rollback. WAL provides durability.
        </li>
        <li>
          <strong>Consistency</strong> — eager validation rejects bad inputs before they enter the
          buffer.
        </li>
        <li>
          <strong>Isolation</strong> — single writer prevents dirty reads and lost updates.
        </li>
        <li>
          <strong>Durability</strong> — every operation hits the WAL during commit, flushed per{" "}
          <code>Durability</code>.
        </li>
      </ul>

      <h2 id="lifecycle">Lifecycle</h2>
      <SketchCard caption="A transaction has exactly three terminal paths: commit, rollback, or destructor-rollback. done_ flips once and never flips back.">
        <TransactionLifecycleDiagram />
      </SketchCard>

      <p className="text-muted text-[13px] mt-8">
        Related: <Link to="/docs/internals/lock-manager">Lock manager</Link> ·{" "}
        <Link to="/docs/storage">Storage &amp; recovery</Link> ·{" "}
        <Link to="/docs/cpp-sdk">C++ SDK</Link>.
      </p>
    </DocsShell>
  );
}
