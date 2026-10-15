# Capacity and stability policy

## Public-only capacity decision

Choose the structural capacity before reading any candidate outcome:

| Labeled rows | Morgan bits per molecular role |
|---:|---:|
| 2–63 | 32 |
| 64–255 | 64 |
| 256 or more | 128 |

The inputs are limited to labeled-row count, candidate-row count, number of
molecular roles, mapping coverage, reaction family, and declared memory.
Candidate outcomes must never influence the choice. The initialized state
records policy version `coupling-fingerprint-v1`.

The deliberately small capacities avoid an extremely sparse representation
when only a few labeled reactions are known. Every molecular role also receives
eight RDKit physicochemical descriptors. Buchwald requires complete structure
coverage; there is no silent all-zero vector or missing-structure indicator.

## Memory guard

For `R` roles and `B` bits:

`dimensions = R × (B + 8 physicochemical descriptors)`

Features are float32. Conservatively estimate four simultaneous working copies
for generation, model fitting, uncertainty, and acquisition. Use at most 50% of
the declared memory:

`working bytes = candidates × dimensions × 4 bytes × 4 copies`

If the full matrix does not fit, downshift 128 → 64 → 32. If 32 bits still
cannot fit as one matrix, keep every legal candidate and score deterministic
batches. Never filter or subsample the candidate pool to make memory fit.

Initialization blocks when:

- fewer than two labeled rows are available;
- no candidates or molecular roles exist;
- declared memory is below 128 MB;
- Buchwald structure coverage is below 100%; or
- one candidate working set cannot fit the declared budget.

## Surrogates

### Buchwald

- random forest with 100 trees;
- `min_samples_leaf=2`;
- one CPU worker for reproducibility and bounded memory;
- tree disagreement as empirical uncertainty;
- Morgan radius 2 plus eight descriptors per molecular role.

### Suzuki

- the same forest plus a corrected per-variable categorical-product Gaussian
  process for the frozen all-categorical competition route;
- forest weight 0.75 and GP weight 0.25;
- categorical one-hot grouped by source variable;
- kernel hyperparameters optimized without candidate-answer tuning;
- above 800 observed rows, record a forest-only capacity fallback.

The stateful external adapter may use a fixed RBF for mixed categorical and
continuous schemas. Arbitrary unseen molecular identities need a reviewed
structure-feature adapter; do not claim universal transfer.

## Acquisition and state safety

- expected improvement;
- Suzuki exploration probability decreases linearly from 0.15 to 0.01;
- frozen Buchwald exploration probability is zero;
- one recommendation and one real feedback value per round;
- no repeated candidate;
- atomic state writes and frozen SHA-256 hashes for train, candidates, and
  structure map.

## Changing the policy

Change thresholds only with training-label-only validation or a genuinely
independent external dataset. Record the old value, new value, evidence,
decision time, and resulting config hash. Never use competition candidate
answers to tune the policy.
