import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";
import { SketchCard, ErrorHierarchyDiagram } from "../components/SketchDiagram";

export const Route = createFileRoute("/docs/internals/lock-manager")({
  head: () => ({
    meta: [
      { title: "Lock manager — ELIPS Docs" },
      {
        name: "description",
        content:
          "How ELIPS enforces single-writer / multi-reader through a POSIX advisory file lock and an RAII LockManager.",
      },
      { property: "og:title", content: "Lock manager — ELIPS" },
      {
        property: "og:description",
        content:
          "RAII flock(2) coordination, exception unwinding behaviour, and cross-platform notes.",
      },
      { property: "og:url", content: "/docs/internals/lock-manager" },
    ],
    links: [{ rel: "canonical", href: "/docs/internals/lock-manager" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Internals"
      title="Lock manager"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "class", label: "Class shape" },
        { id: "semantics", label: "Locking semantics" },
        { id: "open", label: "Use in open()" },
        { id: "raii", label: "RAII guarantees" },
        { id: "platforms", label: "Cross-platform" },
        { id: "errors", label: "Errors" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>LockManager</code> enforces ELIPS' single-writer / many-reader contract for on-disk
        databases through a POSIX advisory file lock. It acquires an exclusive, non-blocking lock at
        open time and holds it for the lifetime of the <code>ElipsInstance</code>.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        A persistent database directory contains a <code>LOCK</code> file. The first read-write
        opener takes an exclusive <code>flock(LOCK_EX | LOCK_NB)</code>; subsequent writers fail
        fast with <code>LockConflict</code>. Read-only opens take a shared lock and may coexist with
        other readers. There is no background coordination — no daemon, no thread pool, just a file
        descriptor.
      </p>

      <h2 id="class">Class shape</h2>
      <CodeBlock lang="cpp">
        {`// include/elips/kernel/LockManager.hpp
namespace elips {

class LockConflict : public ElipsError {
public:
    using ElipsError::ElipsError;
};

class LockManager {
public:
    explicit LockManager(const std::string& lock_path);  // acquires
    ~LockManager();                                       // releases

    LockManager(const LockManager&)            = delete;
    LockManager& operator=(const LockManager&) = delete;
    LockManager(LockManager&&) noexcept;                  // movable
    LockManager& operator=(LockManager&&)      = delete;

private:
    int fd_{-1};
};

} // namespace elips`}
      </CodeBlock>

      <h2 id="semantics">Locking semantics</h2>
      <CodeBlock lang="cpp">
        {`LockManager::LockManager(const std::string& lock_path) {
    fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        throw StorageError{"cannot open lock file: " + lock_path};
    }
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw LockConflict{"database is already open by another writer: " + lock_path};
    }
}

LockManager::~LockManager() {
    if (fd_ >= 0) {
        ::flock(fd_, LOCK_UN);
        ::close(fd_);
    }
}`}
      </CodeBlock>
      <ul>
        <li>
          <strong>Path</strong> — <code>&lt;db_path&gt;/LOCK</code>. The file is a 0-byte target for{" "}
          <code>flock(2)</code>; no data is written.
        </li>
        <li>
          <strong>LOCK_EX</strong> — exclusive lock. No other process can hold any lock on the file
          simultaneously.
        </li>
        <li>
          <strong>LOCK_NB</strong> — non-blocking. Conflicting calls return immediately with{" "}
          <code>EWOULDBLOCK</code>, not a hang.
        </li>
        <li>
          <strong>Release on close</strong> — POSIX releases every <code>flock</code> held on a file
          when any descriptor for it is closed.
        </li>
        <li>
          <strong>Release on process exit</strong> — the kernel cleans up file descriptors, so a
          crashed writer is automatically unlocked.
        </li>
      </ul>

      <h2 id="open">
        Use in <code>open()</code>
      </h2>
      <CodeBlock lang="cpp">
        {`std::unique_ptr<ElipsInstance> open(const std::string& path, const Config& config) {
    // ... path validation ...
    LockManager lock{(fs::path(path) / lock_file).string()};   // single-writer
    // ... identity, snapshot load, WAL replay ...
    auto instance = std::make_unique<ElipsInstance>(path, effective,
                                                    /*persistent=*/true,
                                                    std::move(lock));
    return instance;
}`}
      </CodeBlock>
      <p>
        The lock is created on the stack inside <code>open()</code>, moved into{" "}
        <code>ElipsInstance</code>, and released when the instance is destroyed or{" "}
        <code>close()</code> is called. If any step between acquisition and instance construction
        throws, stack unwinding releases the lock — there is no leakage path.
      </p>

      <h2 id="raii">RAII guarantees</h2>
      <div className="not-prose my-4 rounded-lg border border-hairline overflow-hidden bg-surface">
        <table className="w-full text-[14px]">
          <thead className="bg-canvas-soft">
            <tr>
              <th className="text-left px-4 py-2 font-semibold text-ink">Scenario</th>
              <th className="text-left px-4 py-2 font-semibold text-ink">Lock state</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["Normal open → close", "Acquired on open, released on close"],
              ["ElipsInstance destructor", "Released via ~LockManager() → flock(LOCK_UN)"],
              ["Process crash / SIGKILL", "Released by the kernel"],
              [
                "Exception during open() before instance constructed",
                "Released via stack unwinding",
              ],
              ['In-memory database (":memory:")', "No lock created (no filesystem path)"],
            ].map(([s, l]) => (
              <tr key={s} className="border-t border-hairline align-top">
                <td className="px-4 py-3 text-ink">{s}</td>
                <td className="px-4 py-3 text-body">{l}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2 id="platforms">Cross-platform notes</h2>
      <ul>
        <li>
          <strong>POSIX (macOS, Linux)</strong> — current implementation, BSD-style{" "}
          <code>flock(2)</code>. Lock follows the file descriptor.
        </li>
        <li>
          <strong>Windows</strong> — planned via <code>LockFileEx</code> with{" "}
          <code>LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY</code>. Path handling differs.
        </li>
      </ul>

      <h2 id="errors">Errors</h2>
      <SketchCard caption="Every lock failure surfaces as a typed ElipsError descendant — never a bare std::runtime_error.">
        <ErrorHierarchyDiagram leaves={["LockConflict", "StorageError"]} />
      </SketchCard>

      <CodeBlock lang="python">
        {`try:
    db = elips.open("path/to/db")
except elips.LockConflict as e:
    print(f"Database is locked by another process: {e}")
except elips.StorageError as e:
    print(f"Storage error: {e}")`}
      </CodeBlock>

      <p className="text-muted text-[13px] mt-8">
        Related: <Link to="/docs/architecture">Architecture</Link> ·{" "}
        <Link to="/docs/internals/transaction-engine">Transaction engine</Link>.
      </p>
    </DocsShell>
  );
}
