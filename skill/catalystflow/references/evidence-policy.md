# Evidence and report policy

## Evidence tiers

Use only these source classes:

1. `official_sds_or_ghs`: current official safety data sheet or authoritative
   hazard registry;
2. `primary_literature`: paper or primary experimental report;
3. `chemical_registry`: traceable identity or property registry;
4. `supplier_catalog`: current catalog availability or quotation evidence;
5. `project_artifact`: a project record whose content directly supports the
   stated claim.

Every source requires `type`, `title`, `locator`, `retrieved_at`, and a concise
`claim`. A generic trajectory or model configuration is not procurement or
safety evidence. Reject evidence containing a proposed yield, a ranking
instruction, or an unsupported causal claim.

## Six feasibility dimensions

Assess every queried Top-5 condition separately for:

- `budget`;
- `procurement_difficulty`;
- `hazard`;
- `operational_complexity`;
- `scale_up_risk`;
- `waste_disposal_burden`.

Every dimension has `level`, `basis`, `confidence`, `rationale`, and `sources`.
Use `basis=verified_evidence` only when a qualifying source supports the stated
dimension. Use `basis=llm_inference` at low or medium confidence only and mark
it for verification. With `basis=unknown`, the level must be `unknown` and the
source list empty.

Verified budget and procurement claims require a supplier catalog or a project
procurement record. Verified hazard requires official SDS/GHS evidence.
Verified operational complexity and scale-up risk require primary experimental
literature or a directly relevant project procedure. Verified waste-disposal
burden requires SDS/GHS evidence or a directly relevant project waste record.
Registries establish identity or properties but do not alone verify price,
stock, handling safety, scale-up safety, or waste acceptance.

## External retrieval protocol

For every Top-5 report, first call `science_list_dbs`. The only exception is an
explicit user request for offline mode, which must be stated in the report. Deduplicate repeated
substances and claims, then use `science_search` with an explicit positive
integer `limit` and `science_fetch` for the selected record. Prefer PubChem or
ChEBI for identity and properties, and Crossref, OpenAlex, Semantic Scholar,
PubMed, or Europe PMC for directly relevant literature when those databases are
listed as available. A search hit is discovery evidence, not verified evidence;
read the record and confirm that it directly supports the claim before adding it
to `sources`; call `provenance_record` for each source actually used.

If the service is unavailable, no directly relevant record is found, or the
available databases do not cover the claim, use `llm_inference` at low or medium
confidence or `unknown`. Do not fabricate a source. Scientific registries and
literature databases do not replace current supplier quotations, official SDS,
SOP review, or EHS approval. Perform this retrieval only after the optimizer has
fixed the queried Top-5; retrieved information cannot enter yield prediction,
EI, candidate scoring, or optimizer ranking.

For new-dataset parsing, the timing differs but the outcome boundary is the same:
database evidence may be retrieved before training to resolve identity, synonyms,
reviewed structures, reaction family, roles, and units. Outcome-free reviewed
metadata may enter feature construction. External or literature yields, activities,
reported optimum conditions, outcome-derived preferences, and unqueried candidate
results may not enter fitting, parameter selection, prompting, EI, or ranking.

The program maps `low=3`, `medium=2`, `high=1`, and `unknown=null`. If any
dimension is unknown, `feasibility_total_score` is null. Observed yield,
optimizer rank, model prediction, acquisition value, and DFT context do not
enter this score.

## Safety gate and final recommendation

The program, not DeepSeek, computes the safety gate:

- `blocked`: `critical_safety_flags` is non-empty, or hazard or scale-up risk
  is high;
- `conditional`: not blocked and at least one dimension is unknown, safety-
  relevant evidence for hazard, scale-up risk, or waste disposal is unverified,
  or waste-disposal burden is high;
- `eligible`: all dimensions are known, safety-relevant evidence is verified,
  and none of the preceding conditions applies.

Assess ranks 1–5 exactly once and preserve `optimizer_rank`. Select one
non-blocked final recommended reaction condition when available. If all five
are blocked, return `no_condition_recommended`. An eligible label never replaces
current SDS, laboratory SOP, process-safety review, or EHS approval.

## DFT evidence boundary

Use `computed` only when a project calculation supports the component and role;
otherwise use `partial`, `planned`, `unavailable`, or `not_computed`. Always
report coverage, confidence, and `ranking_effect=none`. Electronic descriptors
provide mechanistic context, not causal proof.

After Top-5 selection, use chemical registries and primary literature to interpret
reaction roles, descriptor direction, and mechanistic compatibility. Record the
source and supported claim separately from the computed DFT artifact. If no direct
evidence is found, use low- or medium-confidence `llm_inference`; never invent DFT
values, change coverage/status, or claim that a descriptor caused the observed yield.

## Required report sections

Produce Chinese Markdown and JSON containing input hashes, environment and fixed
model configuration, trajectory validation, confidence intervals, reference
comparison, queried Top-5 conditions, ten six-dimensional assessments, final
recommended reaction conditions or an explicit no-recommendation result, DFT
context, limitations, provenance, and required checks. Never include a
competition score or project self-rating.
