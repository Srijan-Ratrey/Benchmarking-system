# runs/

Benchmark outputs land here, one folder per run: `<timestamp>-<run_name>/`
containing `config.yaml` (snapshot), `raw.csv` (per-row predictions),
`summary.csv` / `summary.json` (per-model aggregates), and `run.log`.

Run outputs are gitignored by default since raw.csv can contain dataset
content. Upload reference runs here manually if you want to share them.
