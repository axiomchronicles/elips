# CLI — `elips`

A local command-line tool for inspection, maintenance, and EQL execution. No
server required. Build produces the `elips` binary.

```bash
elips <command> <db_path> [options]
```

When creating a new database, pass `--dimension` (and optionally `--metric`,
`--index exact` for brute-force; the default is HNSW). Existing databases read
their identity from disk.

## Commands

```bash
# inspect
elips info /my_db
elips collections /my_db
elips stats /my_db
elips verify /my_db                       # replays WAL + validates; prints OK/CORRUPT

# query (EQL from a flag or a file)
elips query /my_db --dimension 3 --eql 'add in docs vector [0.1,0.2,0.3] metadata {"t":"a"}'
elips query /my_db --eql 'search in docs nearest [0.1,0.2,0.3] k 5 project t yield'
elips query /my_db --file query.eql

# maintenance
elips checkpoint /my_db
elips quantize /my_db --collection docs --quantize sq8   # omit --collection for all

# import / export (JSON Lines: {"id","vector","data"})
elips export /my_db --collection docs --output docs.jsonl
elips import /my_db --collection docs --input docs.jsonl --dimension 3

# benchmark
elips bench /tmp/bench_db --count 100000 --dim 768
```

The EQL verbs follow the language guide: `add/search/get/scan/delete`. The old
`place/seek/fetch/erase` spellings still parse as deprecated synonyms.

## Output

`query` prints one JSON object per result row:

```json
{"id":"019e...","distance":0.0061,"data":{"t":"a"}}
```

`export` writes one record per line; `import` reads the same format and assigns
the stored id, then checkpoints.
