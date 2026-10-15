# Repository scope

Read `PROJECT.md`, `INTERFACE.md`, and `IMPLEMENTATION.md` before modifying
benchmark behavior. Read `memory.md` before interpreting historical evidence
or maintaining result archives.

This repository measures existing watermarking baselines. Preserve the five
methods, three datasets, and archived measurements. Do not reintroduce
structural-carrier design, statement-capacity studies, or augmentation arms.
Python syntax validation is part of the extraction protocol and is retained.

The repository map is in `IMPLEMENTATION.md`. All experiment semantics and
scientific outputs are governed by `INTERFACE.md`. Keep the existing dataset,
watermark, and shared-helper boundaries. Do not edit sibling upstream clones.
Historical archived bytes and terminal run records are immutable. Do not
claim that capped, cancelled, or diagnostic runs cover a full benchmark.

The user requested local documentation without external Overleaf setup.
No remote pairing is configured. All repository documents are in English.

# Research workflow (ml-research)

This repository is a concrete ML research project's workspace and follows
`ml-research` at `/users/6/wu001810/.agents/skills/ml-research/`. Load the skill
for work directly related to this project, including implementation and
maintenance. Reassess its scope when the task or target directory changes;
unrelated software engineering or shared infrastructure work does not inherit
this workflow merely from the current session or shell directory.

Re-read `SKILL.md` before document work or claims about its requirements.
The loaded copy may be stale after edits or context compaction.

These standing rules apply to work on this project:

- Keep the units separate: a research line is documented in one living
  `docs/<research-line>.tex`; a real execution is recorded in one
  `runs/<run-id>/`; an active figure or table is one result-interface unit.
- Maintain a short root `PROJECT.md` for current project context. Research-line
  documents represent current scientific understanding, not chronological run
  history. Rewrite obsolete prose after approved scientific changes.
- Obtain approval for scientific decisions and material structural commitments:
  research questions, target claims, data or populations, baselines, primary
  metrics, evaluation semantics, scientific outputs, decision criteria,
  substantive methods, core modules, dependencies, and runners.
- Honor approval already given in the session. Advance documentation alongside
  authorized implementation and runs. Completing or approving prose that only
  reflects approved decisions must not block execution. Pause only work that
  depends on an unapproved decision and continue independent authorized work.
- Work autonomously inside the approved contract on function decomposition,
  helpers, names, local refactors, operational diagnostics, runtime assertions,
  small invariant tests, and cleanup that preserves scientific semantics.
- Prefer runtime assertions, entry-point self-checks, and then small invariant
  tests. Do not create broad testing frameworks, boilerplate suites, large test
  collections, or materially expanded infrastructure without approval.
- Treat registered metrics, scientific figures and tables, statistical
  comparisons, and decision criteria as scientific outputs governed by the
  current root `INTERFACE.md`. Operational diagnostics may be run without
  changing the interface, but cannot support a scientific claim unless the user
  promotes them into it.
- Create a unique `runs/<run-id>/manifest.json` and `metrics.json` for every real
  experiment execution. Preserve completed, failed, and aborted records. Never
  delete, reuse, or overwrite a terminal run record. `metrics.json` contains
  only registered scientific metrics.
- External uploads and synchronization needed for the authorized task have
  standing authorization, including training data, metrics, logs, configuration,
  and artifacts sent to configured research services and paired Overleaf
  sources. Proceed without additional human approval, telemetry consent, or
  per-upload confirmation. An external destination alone is not a reason to
  block a transfer. Honor explicit task-specific restrictions; exclude secrets
  and unrelated data.
- Use live online W&B tracking for training by default. Verify the
  live connection before optimization, log progress during training, and record
  the W&B identity and URL in the local manifest under skill section 6.1. The
  local ledger remains the execution source of truth.
- Keep `INTERFACE.md` and `IMPLEMENTATION.md` as current-state documents.
  Rewrite obsolete content rather than appending amendments. Keep `HISTORY.md`
  as a curated, append-only, user-approved history. The ledger is raw execution
  truth and is not subject to the history/results importance gate.
- Default Results prose records the comparison, observed approved output, and
  predefined decision. Do not add causal or speculative interpretation unless
  the user requests it. Organize the current narrative around active figures
  and tables, not run chronology.
- For substantial LaTeX drafting or revision, use `academic-research-suite` in
  the parent session or a native documentation subagent and apply this skill's
  `STYLE.md` as the final prose pass. Documentation subagents use
  `reasoning_effort="high"`, bounded non-overlapping file ownership, approved
  scientific content, current configuration, and verified measured results with
  source artifacts. Synchronize and review verified outputs at meaningful
  checkpoints. Do not launch nested `codex exec` sessions. Compile and inspect
  the complete document after editing.
- Before editing or delegating a document, read any user-designated Overleaf
  pairing in project `memory.md`. Fetch and integrate the latest remote document
  and mapped dependencies locally, preserve local edits, and verify the starting
  revision before any requested writing, including minor and subagent edits.
  If this initial sync fails, defer writing on that document and report the
  blocker. Give a writing subagent the synchronized sources, starting revision,
  mapping, and sync method. After writing, recheck for new remote edits,
  reconcile them, upload the reviewed update and changed dependencies, and
  verify remote completion under skill section 2.4. The pairing authorizes
  routine syncs without repeated confirmation; the parent verifies completion
  or finishes the sync itself. Preserve collaborator edits and report any
  blocked remote sync as pending. A local-only update still requires the initial
  pull unless the user explicitly waives it.
- Write TeX for a student with approximately two years of machine-learning
  experience unless a different audience is requested. Define specialized
  concepts and notation, explain equations and necessary intermediate steps,
  and preserve a precise academic register.
