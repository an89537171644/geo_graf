# NIGHTLY STATUS

## Repository

- Base branch: `main`
- Base SHA: `89aef39952931ed50287a890f2da529118373aeb` (current `main`)
- Working branch: `overnight/scientific-hardening-2026-07-11`
- Draft PR: https://github.com/an89537171644/geo_graf/pull/5
- Phase 00 code head: `85ec0e06689629b250d60ee720a37dde1c4feabf`
- Phase 01 local code head: `0474c80`
- Phase 01 remote head: `74320b5d667f39bd551e3894dfb05aae446de095`
- Phase 02 local code head: `3a8a5456f5d3304807577f2b4f5363608f8b5f58`
- Phase 02 remote head: `62326f596def0044058c1972bbd6dcb7720d2209`

## Phase status

| Phase | Status | Commit | Local gate | GitHub CI | Notes |
|---|---|---|---|---|---|
| 00 CI | COMPLETE / MERGED BY OWNER | `f59370f`, `85ec0e0`, `8a84f7a` | 180 tests; coverage 81.54%; demo verified | [run 29169728497](https://github.com/an89537171644/geo_graf/actions/runs/29169728497): 6/6 matrix + Required CI SUCCESS | PR #4 was merged externally into `main`; Codex did not invoke merge |
| 01 E contract | COMPLETE | local `0474c80`; remote `74320b5` | 215 tests; core coverage 80.97%; CLI demo verified | [run 29184900474](https://github.com/an89537171644/geo_graf/actions/runs/29184900474): 6/6 matrix + Required CI SUCCESS; 6 artifacts | No primary E without an approved profile, confirmed in-range interval and valid positive calculation |
| 02 Pairing | COMPLETE | local `3a8a545`; remote `62326f5` | 233 tests; core coverage 81.73%; CLI demo verified | [run 29186501814](https://github.com/an89537171644/geo_graf/actions/runs/29186501814): 6/6 matrix + Required CI SUCCESS; 6 artifacts | `pair_id` is explicit; invalid, incomplete or ambiguous pairing falls back to independent analysis with a visible reason |
| 03 Indicators/metrology | NOT STARTED | — | — | — | Strict phase order |
| 04 Plots/censoring | NOT STARTED | — | — | — | Strict phase order |
| 05 Stretch | NOT STARTED | — | — | — | Not permitted before phases 0–4 |

## Numerical changes

Phase 00 made no numerical or methodological change.

Phase 01 keeps the formula and legacy numerical values unchanged, but changes their
scientific status: a call without a confirmed range is now
`diagnostic_unapproved_v1`, `review_required`, `is_primary=false`. The existing demo
therefore retains its former `nu=0.30`, `shape_factor=1.00` numerical values while no
longer presenting the whole-curve result as primary.

The new project profile `antonov_round_stamp_v1` fixes `nu=0.30` and
`shape_factor=0.80`. On the synthetic golden linear curve (`D=300 mm`,
`ds/dp=0.01 mm/kPa`) this gives `E=21.84 MPa`; the same calculation with
`shape_factor=1.00` gives `27.30 MPa`, exactly 1.25 times larger. These are test
fixtures, not substituted laboratory results. Antonov plotting and censored-failure
logic were not changed.

Phase 02 does not change Antonov plots, the failure model, or the numerical demo
results. It changes group-comparison eligibility: the old manual adapter could treat
`baseline_group` as evidence of pairing, while the new schema requires an explicit,
complete and one-to-one `pair_id`. Ambiguous, duplicated, incomplete, noncanonical or
non-analyzable pair assignments now use independent analysis with a visible reason.
For a valid paired design, every pressure level is calculated from the same finite
pair subset on both sides, preventing cross-pair means when curve supports differ.
Consequently, comparison numbers may intentionally change only for data that were
previously paired implicitly or had unequal/incomplete pair support.

## Remaining blockers

- Engineering selection/approval of a pressure range for each real primary modulus.
- Engineering verification of real `pair_id` values and group membership.
- Multi-indicator aggregation/metrology and censored-failure presentation.
- Real laboratory acceptance on at least three experiments.
- SQLite archive and clean Windows distribution.
- Owner review and merge decision for Draft PR #5. Codex did not merge or enable auto-merge.

Current status: research beta under author control; not a finished engineering release.
