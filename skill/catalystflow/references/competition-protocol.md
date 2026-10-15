# Fixed competition protocol

## Fixed design

- Datasets: `Suzuki`, `Buchwald_sub4`
- Seeds: `100, 200, ..., 2000`
- Budget: 40 sequential, non-repeated queries for every seed
- Batch size: one
- Acquisition: expected improvement
- Configuration source: `code/config/production_config.json`

Suzuki uses the fixed 0.75 random-forest/0.25 categorical-product Gaussian
process ensemble and its fixed decreasing exploration schedule. Buchwald uses
the fixed identity/knowledge/structure random forest with no extra uniform
exploration. Do not alter these branches during reproduction.

The fixed model configuration is determined only from internal validation of
the initial training data. Every seed initializes and trains from scratch.
Stage model files, stage query records, and formal evaluation results do not
enter training, tuning, model selection, or candidate ranking.

## Fresh output

Write each run to a new job directory under
`.catalystflow-agent/jobs/<job_id>/run`. Never write into `results/`, which is
the submitted reference.

A completed job must contain 40 `.pt` trajectories and 1,600 query records.
Each `.pt` file stores primitive Python values only and must pass checks for
dataset/seed identity, consecutive rounds, unique indices, condition/index
agreement, and monotone `best_so_far`.

## Reproduction comparison

Compare new and submitted trajectories by dataset and seed. Require exact
equality for indices, conditions, observed feedback, selection mode, and
best-so-far. Compare finite prediction, uncertainty, and acquisition floats
with `rtol=1e-10` and `atol=1e-10` to tolerate harmless serialization details.

Report `identical`, `numerically_equivalent`, or `different`. If different,
retain the first bounded set of field-level mismatches. Never silently replace
the reference or round a mismatch away.
