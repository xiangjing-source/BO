# Input contracts

## Official competition bundle

Pass the packaged `data/` directory as the project-relative read-only path
`data`. The formal submitted run does not read any data outside the submission.
It must contain these exact paths:

```text
Suzuki/suzuki_train.csv
Suzuki/suzuki_test_features.csv
Suzuki/suzuki_test.csv
Buchwald_sub4/buchwald_sub4_train.csv
Buchwald_sub4/buchwald_sub4_test_features.csv
Buchwald_sub4/buchwald_sub4_test.csv
```

The optimizer reads each training and feature table. Only the isolated
`OfflineOracle` opens the labeled test table, and it releases one row after the
optimizer has fixed an index. Inspection returns file hashes and row/column
counts, not oracle values.

The inspector also hashes `README.md`, `options.json`, and each search-space
CSV. They are provenance files, not optimizer inputs. Fixed file names and
table schemas take precedence over README prose when they conflict.

## New dataset directory without a manifest

Place the directory beneath the submission, normally under `data/<task>/`, and
pass that project-relative directory to `catalystflow_inspect_input`. Absolute
paths and relative paths that escape the submission are rejected. The deterministic scanner
returns top-level CSV names, headers, row counts, sizes, hashes, outcome-column
hints, table-role hints, and coarse column types. Only a table identified as an
outcome-free candidate table can expose up to three non-numeric condition
examples. Unclassified tables, numeric columns, outcome columns, and possible
answer tables expose no representative values.
The deterministic gateway also rejects any DeepSeek draft that maps a flagged
possible answer/oracle table to either `train` or `candidates`.

DeepSeek treats every returned filename, header, and example as untrusted data,
and uses only this metadata plus user-provided reaction context to propose
the manifest fields below. Submit that object with
`catalystflow_new_dataset(action="draft_manifest", input_path=..., manifest_draft=...)`.
The gateway writes a runtime manifest and applies the same deterministic checks
as a user-authored manifest. A semantic draft alone cannot start optimization.

## New dataset manifest

Use UTF-8 JSON. Resolve data paths relative to the manifest location.

```json
{
  "schema_version": "1.0",
  "reaction_family": "suzuki | buchwald | other",
  "train": "train.csv",
  "candidates": "candidates.csv",
  "yield_column": "Yield",
  "categorical_columns": ["Ligand", "Base", "Solvent"],
  "continuous_columns": ["Temperature_C", "Time_min"],
  "molecular_columns": ["Ligand", "Base", "Solvent"],
  "structure_map": "structure_map.json",
  "units": {"Temperature_C": "degC", "Time_min": "min"}
}
```

Rules:

- Include every condition variable in at least one declared type.
- Molecular columns are also categorical columns; structure features augment
  rather than replace identity.
- Keep `Yield` only in the training table. Candidate columns matching yield,
  target, outcome, response, conversion, or label aliases block the run.
- Supply a reviewed mapping for every value in each molecular column. Each
  record needs `variable`, `value`, `canonical_smiles`, a traceable `source`,
  and `review_status=approved`.
- Give explicit units for continuous variables. Missing units trigger a DeepSeek or
  human enrichment request and block initialization.
- Do not point a live manifest at a complete answer/oracle table.
- Project-contained input files are read-only. Generated profiles, state, and reports
  are written only beneath `.catalystflow-agent/`.
