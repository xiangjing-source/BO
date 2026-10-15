# Frozen model routes

## Buchwald–Hartwig

For the frozen internal task, use the fused identity, label-free chemistry
knowledge, and structure route:

1. Preserve every public condition variable as categorical identity.
2. Add label-free role/name chemistry features.
3. Represent each molecular role independently with Morgan radius-2
   fingerprints plus eight physicochemical descriptors.
4. Select 32, 64, or 128 bits only from initial labeled-row count and declared
   memory, never from candidate outcomes.
5. Fit a random forest with 100 trees and minimum leaf size 2.
6. Use tree disagreement as empirical uncertainty.
7. Select with expected improvement and no additional uniform-random
   exploration for the frozen competition route.
8. Add exactly one returned yield and refit after every round.

For an internal frozen dataset, load the reviewed structure-map JSON and verify
100% `(variable, value)` coverage plus the stored hash. For an external dataset,
preserve all extra condition variables and build a new structure map in this order:

`direct SMILES/InChI → local cache → PubChem → OPSIN → RDKit canonicalization`

The language model orchestrates these tools and records provenance. It must not
write a structure from memory. Block unresolved, invalid, disconnected, or
ambiguous records until chemical review; never substitute an all-zero vector.
When external variable semantics or identities do not match the internal task,
use the structure-only adapter as the conservative fallback rather than
dropping extra columns.

## Suzuki–Miyaura

Use the frozen categorical hybrid route:

1. One-hot encode categorical variables.
2. Group one-hot positions by chemical variable.
3. Fit a random forest and a corrected per-variable categorical-product
   Gaussian process.
4. Combine predictions with forest weight 0.75 and GP weight 0.25.
5. Combine tree disagreement, GP variance, and model-mean disagreement.
6. Select with expected improvement plus the same decreasing exploration.
7. Add one returned yield and refit after every round.

This exact route is for the frozen all-categorical competition schema. External
data must provide semantically reviewed column mappings and initial labeled
experiments. Preserve additional continuous fields and use the stateful
fixed-RBF adapter when the schema is mixed; do not discard extra variables or
claim that this adapter is numerically identical to the frozen route.

## Failure boundaries

- Preserve state during provider, PubChem, OPSIN, or model timeouts.
- Permit one pending recommendation only.
- Freeze hashes for training data, outcome-free candidates, and structure maps.
- Do not use predictions as feedback.
- Do not tune model choices after inspecting competition candidate outcomes.
