# data/

Upload your labeled dataset CSVs here. They are gitignored (may contain
sensitive conversation data), so each user supplies their own.

Expected shape: one row per example, with the columns referenced by your
YAML config (`dataset.input_columns`, `dataset.gold_column`, and optionally
`dataset.id_column`). See `configs/example.yaml` and the main README.
