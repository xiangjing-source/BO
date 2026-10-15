# Structure resolution and provenance

## Resolution order

1. Parse supplied SMILES or InChI directly with RDKit.
2. Reuse a successful local cache entry with an identical query hash.
3. Resolve a name through PubChem PUG REST.
4. Fall back to OPSIN for systematic chemical names.
5. Stop for manual review when the name is ambiguous or unresolved.

The language model may identify likely aliases for a reviewer, but it must not
write a guessed structure into the production mapping.

## Required mapping fields

Each record must retain:

- reaction-variable role and original value;
- identifier type and SHA-256 of the query;
- canonical isomeric SMILES;
- InChI and InChIKey;
- source (`input`, `pubchem`, or `opsin`) and source identifier;
- retrieval time, resolution status, and errors.

Keep the raw source name. Deduplicate with canonical structures only after
reviewing salts, protonation, stereochemistry, tautomers, and mixtures.

## Safety

- Contact only fixed HTTPS PubChem and OPSIN hosts.
- URL-encode names and reject control characters or oversized identifiers.
- Use finite timeouts and bounded retries.
- Never send model-provider API keys to chemistry databases.
- Never log environment variables or HTTP authorization headers.
- Write cache and mapping JSON atomically.
- Hash and freeze the mapping at optimizer initialization.

## Chemical review gates

Require review for:

- more than one PubChem CID for a name;
- missing stereochemistry when stereochemistry affects the reagent;
- mixtures, solvates, salts, polymers, supported catalysts, or unspecified
  ligand/metal complexes;
- a structure whose role is chemically implausible;
- RDKit sanitization failure;
- conflicting InChIKeys across sources.
