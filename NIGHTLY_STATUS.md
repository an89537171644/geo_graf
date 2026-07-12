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
- Phase 03 local code head: `cc909901bf52c50a1214793caeb88beccd149618`
- Phase 03 remote head: `148622d1c171dbd404ac5ef572cae6b13afb451d`
- Phase 04 local code head: `420e5f02334a1b747e74c4cf6857ab308aceda7e`
- Phase 04 remote code head: `47488d68d12179dac5b085229b0c2031a248b10c`

## Phase status

| Phase | Status | Commit | Local gate | GitHub CI | Notes |
|---|---|---|---|---|---|
| 00 CI | COMPLETE / MERGED BY OWNER | `f59370f`, `85ec0e0`, `8a84f7a` | 180 tests; coverage 81.54%; demo verified | [run 29169728497](https://github.com/an89537171644/geo_graf/actions/runs/29169728497): 6/6 matrix + Required CI SUCCESS | PR #4 was merged externally into `main`; Codex did not invoke merge |
| 01 E contract | COMPLETE | local `0474c80`; remote `74320b5` | 215 tests; core coverage 80.97%; CLI demo verified | [run 29184900474](https://github.com/an89537171644/geo_graf/actions/runs/29184900474): 6/6 matrix + Required CI SUCCESS; 6 artifacts | No primary E without an approved profile, confirmed in-range interval and valid positive calculation |
| 02 Pairing | COMPLETE | local `3a8a545`; remote `62326f5` | 233 tests; core coverage 81.73%; CLI demo verified | [run 29186501814](https://github.com/an89537171644/geo_graf/actions/runs/29186501814): 6/6 matrix + Required CI SUCCESS; 6 artifacts | `pair_id` is explicit; invalid, incomplete or ambiguous pairing falls back to independent analysis with a visible reason |
| 03 Indicators/metrology | COMPLETE | local `cc90990`; remote `148622d` | 299 tests; core coverage 82.98%; CLI demo verified; indicator demo 11/11 rows | [run 29189319814](https://github.com/an89537171644/geo_graf/actions/runs/29189319814): 6/6 matrix + Required CI SUCCESS; 6 artifacts | Per-channel passports, deterministic verification and an immutable aggregation basis; no scientific settlement while review is required |
| 04 Plots/censoring | COMPLETE | local `420e5f0`; remote `47488d6` | 334 tests; core coverage 85.30%; Ruff/compile/pip check PASS; CLI demo and semantic verifier PASS | [run 29191539241](https://github.com/an89537171644/geo_graf/actions/runs/29191539241): 6/6 matrix + Required CI SUCCESS; 6 artifacts | Explicit repeat selection; coordinate-aware support; individual failure intervals; no default point estimate |
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

Phase 03 changes only the path from raw indicator readings to settlement. Every active
channel now has its own passport, correction factor, initial reading/turn, zero
correction, verification result and optional centre-relative coordinates. A fixed
`settlement_aggregation` policy is resolved before row processing. Means never change
their denominator by skipping missing channels; `plane_center` fits `z=a+bx+cy` and
saves the centre intercept, rank, residual RMS and tilt diagnostics. A reference
indicator is calibrated separately and never enters the vertical mean denominator.

Verification is evaluated against the experiment date, not the computer's current
date. An unknown, expired or not-yet-valid verification and any unconfirmed channel
assignment block indicator-derived settlement with an explicit status. Legacy common
passports migrate losslessly to `manual-entry-draft/1.2`, remain non-effective and
require an audited engineering distribution to channels. The indicator demo preserves
11 raw rows, records four zero crossings and produces 11 successful primary-channel
aggregation rows. Antonov plotting, the failure model and direct supplied settlement
remain unchanged.

Phase 04 changes publication presentation without changing the Antonov scientific
curve model or inventing a failure-capacity estimator. A repeated series now requires
an explicit `mean_curve`, `median_curve`, `manual_representative` or
`individual_curves` decision; a manual representative requires test ID, author, UTC
timestamp and reason. Groups, tests, bootstrap input and curve numbering are sorted
canonically, so row order no longer selects a curve or changes the result.

For F–s, aggregation is permitted only for identical finite stamp diameter and area
and only over the common force support. For p–s, only the common pressure support is
used. For p–s/D, each test is normalized before aggregation and contributes only
inside its own measured support. No extrapolation is performed. Every exported point
stores `n`, `measured_n` and `interpolated_n`; a marker is drawn only where no
interpolated contribution is present.

Failure results use `failure-analysis/1.0`. The demo contains two observed,
interval-censored failures and two right-censored tests. Their individual bounds are
drawn separately; no arithmetic pooling is present and `summary_method=none` leaves
`point_estimate=null`. The final local demo bundle
`work/phase04-publication-demo-final/reproducibility.zip` passed the semantic verifier
with SHA-256 `E98BD0CFBEF51C16AD3912B64E58F3D6BAD2E10982515020D3079EBD8201051F`.

## Remaining blockers

- Engineering selection/approval of a pressure range for each real primary modulus.
- Engineering verification of real `pair_id` values and group membership.
- Real laboratory acceptance on at least three experiments.
- Engineering approval of publication-curve decisions for every real repeated series;
  manual representative decisions require a named author and reason.
- Real channel assignments, instrument passports, coordinates and aggregation policy
  require engineering review; manual input also requires engineering acceptance.
- SQLite archive, approved revisions, backup/restore and clean Windows distribution
  are not implemented and require engineering review.
- Owner review and merge decision for Draft PR #5. Codex did not merge or enable auto-merge.

The existing support-adaptive `compare_groups()` inference remains a separate,
previously accepted comparison method; Phase 04 common-support rules apply to
publication/group-curve aggregation and do not silently redefine that method.

Current status: research beta under author control; not a finished engineering release.
