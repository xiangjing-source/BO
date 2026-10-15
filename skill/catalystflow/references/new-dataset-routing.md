# New-dataset robustness and routing

## Deterministic diagnostics

Compute diagnostics before any model call:

- schema and target validity;
- candidate outcome leakage;
- duplicate full-condition rows;
- per-column unseen category rate;
- unseen full-combination rate;
- per-continuous-column fraction outside the training range and beyond three
  robust scaled deviations;
- reviewed structure-map coverage for molecular variables;
- train/candidate hashes and row counts.

Classify OOD risk as low, medium, high, or blocked. Treat this as a warning and
routing signal, not a calibrated probability of failure.

## Legitimate DeepSeek contribution

Use the Open Science shell to improve semantic robustness by:

- identifying reaction family and chemical roles from documented context;
- resolving unseen names to traceable structures, with human approval;
- identifying units and incompatible encodings;
- retrieving registry and primary-literature evidence for input semantics,
  feasibility, and post-selection mechanisms;
- explaining why a route or fallback is appropriate.

Do not allow DeepSeek to invent features, assign pseudo-yields, change observed
feedback, or apply a hidden preference score. Numeric priors may enter a future
model only after explicit task-local validation and configuration freezing.

## Portable live route

For an outcome-free candidate pool, fit a random-forest or mixed-variable
surrogate to initial labels plus committed feedback. Preserve all categorical
identities and standardize continuous values. Append reviewed
Morgan/physicochemical features whenever the manifest declares reviewed
molecular columns; preserve categorical identity and standardized continuous
variables in the same route. If no molecular columns are declared, state that
the numerical model remains identity/continuous-based. Use tree disagreement
as uncertainty and expected improvement for selection. Keep any exploration
schedule explicit, bounded, and fixed before the first recommendation.

This route is designed to fail clearly on novel inputs. It is not numerically
identical to either official final-model branch and carries no inherited benchmark
guarantee.

For the completed queried Top-5, report DFT status separately. Unless a reviewed,
task-specific DFT mapping is available, use `status=unavailable`, null coverage
and confidence, and `ranking_effect=none`; do not reuse the competition DFT table
for unrelated structures.

External databases and DeepSeek may resolve identities, synonyms, structures,
reaction family, variable roles, and units before deterministic validation. Only
reviewed outcome-free structures and metadata may enter model features. Never use
external or literature yields, activity labels, optimum conditions, unqueried
candidate outcomes, or outcome-derived preferences in fitting, parameter selection,
prompting, EI, or ranking. After the queried Top-5 is fixed, use databases and
primary literature for feasibility and DFT interpretation; fall back to explicitly
labelled LLM inference when no valid evidence is found.
