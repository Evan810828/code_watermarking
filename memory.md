# Durable project context

- The project is a baseline watermarking measurement repository. The local
  directory retains its historical name; the GitHub remote remains
  `Evan810828/code_watermarking`.
- `out/README.md` contains the detailed five-method report. The current
  local LaTeX/PDF report is `docs/baseline_benchmarks.tex`.
- `results/inventory.json` maps preserved original paths to compressed
  archives and SHA-256 identities. Restore with `src/result_archive.py`.
- SeqMark has complete HumanEval and MBPP evaluations for two encoders. Its
  LiveCodeBench run was cancelled; partial artifacts remain available.
- CodeWM includes capped cells, chat-protocol variants, and original-paper
  completion-protocol reproductions. Their task cohorts and quality estimators
  differ and must remain separate.
- Historical SeqMark manifests and source snapshots retain original paths
  and code. Missing historical provenance must not be inferred.
- Documents remain local; there is no Overleaf pairing.
