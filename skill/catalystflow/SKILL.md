---
name: catalystflow
description: Run and audit CatalystFlow reaction-condition optimization inside Open Science. Use when the packaged organization-provided Suzuki and Buchwald data must be freshly reproduced as 2 datasets × 20 fixed seeds × 40 sequential queries; when a project-contained new reaction directory or manifest needs DeepSeek semantic parsing plus deterministic schema, leakage, novelty, structure-coverage, and out-of-distribution checks; or when completed queries need a queried Top-5, post-selection DFT context, six-dimension DeepSeek-led real-world feasibility assessment, a final recommended reaction condition, and an auditable Chinese report.
---

# CatalystFlow

Use CatalystFlow as the deterministic optimization and audit layer inside the
Open Science agent. Let the language model classify intent, resolve ambiguous
scientific metadata, retrieve sources, and explain results. Never let it read an
oracle table, invent a yield, or replace the numerical optimizer.

## Choose exactly one workflow

1. For the packaged competition bundle at `data/`, run **official reproduction**.
2. For a project-contained new labeled training table plus outcome-free candidate table, run
   **new-dataset onboarding**.
3. For an existing live run, run exactly one **recommend** or **feedback**
   transaction.
4. For a completed reproduction, run **evidence enrichment and final report**.

Call `catalystflow_inspect_input` first when the input type is not already
verified. Read [references/input-contracts.md](references/input-contracts.md)
before asking the user to rename or reshape files.

## Preserve the non-negotiable boundary

- Keep the final model configuration fixed at
  `code/config/production_config.json`.
- Generate official trajectories by executing the optimizer; never treat the
  submitted trajectories as a substitute for a fresh run.
- Permit the isolated `OfflineOracle` to release only the selected candidate's
  yield. Do not send oracle contents, global best values, or unqueried outcomes
  to Open Science or the model.
- Permit database and DeepSeek enrichment for role/type resolution, traceable
  identity and structure resolution, units, reaction-family interpretation,
  literature/SDS evidence, and post-selection interpretation. Reviewed,
  outcome-free structures and metadata may enter feature construction. Keep any
  external yields, activities, optimum conditions, outcome-derived preferences,
  and unreviewed semantic guesses outside fitting, parameter selection, expected
  improvement (EI), candidate scoring, prompting, and ranking.
- Mark DFT as `computed` only when a packaged calculation artifact supports it.
  Otherwise mark it `planned`, `unavailable`, or `not_computed`. The production
  config sets `use_dft=false`, so every DFT explanation has
  `ranking_effect=none`.
- Record real feedback only. Never use a predicted yield as feedback.

Read [references/evidence-policy.md](references/evidence-policy.md) before
making feasibility, safety, price, literature, or mechanistic claims.

## Reproduce the official 2 × 20 × 40 experiments

1. Call `catalystflow_inspect_input` with `input_path=data`. Do not substitute an
   external data path for the formal submitted run.
2. Require `input_kind=official_competition_bundle`, both datasets, the fixed
   configuration, and an isolated oracle boundary.
3. Call `catalystflow_start_reproduction` with the default
   `execution_mode=inline`. It uses the packaged environment and returns only
   after report completion, which is safe for one-shot CLI/container clients.
   Use `detached` only on a host that explicitly preserves child processes, and
   never start a second job while one is running.
4. In detached mode, poll `catalystflow_reproduction_status` until it reports
   `completed` or `failed`. Report progress without claiming completion early.
5. On completion, require all of the following:
   - 40 trajectory files in total;
   - 20 official seeds per dataset;
   - exactly 40 unique, consecutive queries per file;
   - 1,600 total query records;
   - a fresh-versus-submitted comparison result;
   - summary metrics and a generated report path.
6. If comparison differs, report the exact mismatched dataset/seed/field and
   preserve both runs. Do not overwrite the submitted reference trajectories.

When asked for the four official metrics, use `summary_metrics` returned by
`catalystflow_start_reproduction` or `catalystflow_reproduction_status`. Report
the 20-run mean, sample standard deviation, and 95% confidence interval for
`B1`, `B40`, `t95`, and `AUC`. Do not substitute queried Top-5 yields for these
metrics, and do not claim that the metrics are inaccessible when
`summary_metrics_status=available`.

The worker creates a deterministic numerical report and a queried Top-5 feasibility
review request before any DeepSeek call. The Agent must then call
`catalystflow_prepare_feasibility_review`, assess all queried Top-5 conditions, and call
`catalystflow_finalize_report` before presenting the final reaction condition.
The reusable output shape is stored in
[assets/feasibility_assessment_template.json](assets/feasibility_assessment_template.json).
Read
[references/competition-protocol.md](references/competition-protocol.md) for
the fixed seeds, output layout, and equality rules.

## Onboard a new dataset

1. Place the new directory beneath the submission (normally `data/<task>/`). If a
   manifest exists, call `catalystflow_new_dataset` with `action=profile`.
   Otherwise inspect the CSV directory, let DeepSeek draft a manifest from the
   returned metadata and user context, and submit it with
   `action=draft_manifest`.
2. Treat the DeepSeek output as a semantic draft. Only the deterministic
   gateway can accept roles, reject candidate outcome columns, and decide
   readiness.
3. Stop on candidate outcome columns, non-finite/out-of-range training yields,
   duplicate condition rows, missing variables, changed file hashes, or
   unresolved molecular structures.
4. Inspect categorical novelty, combination novelty, continuous range shift,
   structure coverage, and the overall OOD risk. Do not describe a new reaction
   family as validated merely because the file is readable.
5. Before drafting a manifest and again for returned `llm_enrichment_requests`,
   use Open Science chemistry databases and primary-literature tools to resolve
   identities, synonyms, structures, reaction family, variable roles, and units.
   Attach source identifiers and retrieval dates. If no valid source is found,
   let DeepSeek provide a low-confidence semantic draft. Resolve names into
   reviewed structures before numerical use.
6. Call `catalystflow_new_dataset` with `action=init` only after the profile says
   `ready_for_live_optimization=true`.
7. Alternate one `recommend` and one `feedback` call. Require the transaction ID
   returned by `recommend` when recording feedback.
8. When the round budget is complete, use the returned queried Top-5 feasibility
   request to assess all six dimensions for all five queried conditions, then
   select one non-blocked final recommended reaction condition. If all five are
   blocked, return `no_condition_recommended`. Never select from unqueried candidates.

The live adapter preserves all declared variables, combines categorical
identity with reviewed Morgan/physicochemical structure features where
available, standardizes continuous variables, and uses tree disagreement plus
EI. Treat this as a conservative portable route, not as proof that third-party
data will reproduce the official benchmark metrics. Read
[references/new-dataset-routing.md](references/new-dataset-routing.md) before
interpreting OOD results.

## Enrich and finalize the report

After the numerical run is complete, DeepSeek owns the practical assessment layer.
Call `catalystflow_prepare_feasibility_review`, then assess **every** queried Top-5
condition on exactly six dimensions: budget, procurement difficulty,
preliminary hazard, operational complexity, scale-up risk, and waste-disposal
burden. Use `low`, `medium`, `high`, or `unknown`. For each
dimension label its basis as `verified_evidence`, `llm_inference`, or `unknown`
and give confidence, rationale, and sources. DeepSeek-only inference is allowed but
cannot use high confidence and must not be presented as a supplier quote, SDS,
or EHS conclusion.

Use an evidence-first retrieval pass for every Top-5 report:

1. Call `science_list_dbs` once and use only databases returned as available. Skip
   retrieval only when the user explicitly requests offline mode, and label that
   fallback in the report.
2. Deduplicate repeated substances and claims across both Top-5 sets. Prefer
   PubChem or ChEBI for identity, structure, and properties; prefer Crossref,
   OpenAlex, Semantic Scholar, PubMed, or Europe PMC for directly relevant
   literature; use SureChEMBL only when patent evidence is relevant.
3. Call `science_search` with an explicit positive integer `limit` (normally 5),
   select a directly relevant hit, and call `science_fetch` before citing it.
4. Add a source only when the fetched record directly supports the assessed
   claim, and record its title, stable locator, retrieval time, source class,
   and supported claim. Call `provenance_record` for each source actually used.
5. If a database is unavailable, returns no suitable record, or does not cover
   the claim, use `llm_inference` at low or medium confidence or use `unknown`.
   Never invent a citation.

Treat registry and literature records as supporting only the claims they contain.
The currently exposed scientific databases are not supplier quotation systems or
official SDS repositories: they cannot by themselves verify current price, stock,
lead time, regulatory hazard classification, or waste acceptance. Keep all
retrieval after numerical ranking is fixed; never feed retrieved text into the
optimizer.

Use the same database-first rule for post-selection DFT interpretation. Retrieve
chemical-registry or primary-literature evidence for reaction roles, descriptor
direction, and mechanistic compatibility; if no directly relevant evidence exists,
label the explanation as `llm_inference`. Never alter computed DFT values, coverage,
status, or `ranking_effect=none`, and never infer a causal yield effect from a
descriptor or a literature result.

Keep each dimension rationale to one sentence and at most 96 characters, and keep
each candidate summary within 160 characters. Do not repeat complete conditions or
DFT payloads inside the assessment; the deterministic report already preserves them.
For official datasets, call `catalystflow_finalize_report` ten times with
`assessment_context` plus exactly one `candidate_assessment`, then call it once more
with only the two `final_recommendations`. Never place all ten candidate assessments
in one model tool call. Validated partial calls are persisted and may be resumed.

Prefer primary literature, chemical registries, official SDS/GHS documents,
and traceable supplier/catalog records when available. Submit the incremental
10-candidate assessment and final recommendations through
`catalystflow_finalize_report`. The program maps levels to 3/2/1/null, computes
coverage and a safety gate, and rejects a blocked selection. If all five are
blocked, return `no_condition_recommended`.

Keep the two decisions separate:

- `optimizer_rank`: immutable order among queried conditions by observed yield;
- `selected_optimizer_rank`: DeepSeek's final recommended reaction condition from the queried Top-5,
  based on all six dimensions. It is not a new yield
  rank and does not rewrite `optimizer_rank`.

Always include basis/confidence flags, the DFT coverage/confidence, the fixed
model reason for selection, provenance hashes, environment versions, final
pre-experiment checks, and a laboratory SOP/EHS disclaimer. Do not score the
project in the generated or submitted reports. Output Chinese Markdown plus
machine-readable JSON.

## Handle failures

- Preserve a failed reproduction job, log, and partial output; start a new job
  rather than editing a partial trajectory.
- Preserve live optimizer state on provider or literature-tool failure. The
  numerical recommend-feedback loop does not depend on the DeepSeek endpoint.
- Require manual review for ambiguous structures, salts/solvates, conflicting
  registry hits, or missing units.
- Mark unsupported judgments as `llm_inference` or `unknown`; never disguise
  an inference as verified budget, procurement, or hazard evidence.
- Never bypass a blocked profile or report validator with prose.

## Validate the installation

Run the project tests and the bundled smoke test after changing this Skill:

```bash
runtime/.portable_env/bin/python -m unittest discover -s skill/catalystflow/tests -v
runtime/.portable_env/bin/python -m unittest discover -s agent/tests -v
```

Use the packaged Python runtime for numerical forward tests. Report the Skill
validation result, MCP tool list, input hashes, trajectory counts, and any
fallbacks.
