# Evaluation

Planned: a SWE-bench Verified subset runner plus the A/B ablations that motivate
this repo.

Ablations to run (each on the same instance subset, same model, same seed):

1. **Edit format** — SEARCH/REPLACE vs whole-file rewrite vs unified diff.
   Measure: resolve rate, total tokens, edit-application failure rate.
2. **Repo map** — on vs off, on instances whose fix spans more than one file.
3. **Compaction** — sliding window vs summarisation, on long-horizon instances.

Nothing here has been run yet. Results will be committed as raw trajectories
alongside the summary table so the numbers can be re-derived.
