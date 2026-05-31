# CLI — `elips`

A local command-line tool for inspection, maintenance, and EQL execution. No
server required. Build produces the `elips` binary.

```bash
elips <command> <db_path> [options]
```

When creating a new database, pass `--dimension` (and optionally `--metric`,
`--index`). Existing databases read their identity from disk.

## Commands

```bash
# inspect
elips info /my_db
elips vaults /my_db
elips stats /my_db
elips verify /my_db                       # replays WAL + validates; prints OK/CORRUPT

# query (EQL from a flag or a file)
elips query /my_db --dimension 3 --eql 'place in docs vector [0.1,0.2,0.3] data {"t":"a"}'
elips query /my_db --eql 'seek in docs nearest [0.1,0.2,0.3] top 5 project t yield'
elips query /my_db --file query.eql

# maintenance
elips checkpoint /my_db

# import / export (JSON Lines: {"id","vector","data"})
elips export /my_db --vault docs --output docs.jsonl
elips import /my_db --vault docs --input docs.jsonl --dimension 3

# benchmark
elips bench /tmp/bench_db --count 100000 --dim 768
```

## Output

`query` prints one JSON object per result row:

```json
{"id":"019e...","distance":0.0061,"data":{"t":"a"}}
```

`export` writes one record per line; `import` reads the same format and assigns
the stored id, then checkpoints.
