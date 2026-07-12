# NIGHTLY STATUS

## Repository

- Base branch: `main`
- Base SHA: `668e5fa652d9b070f597a095b1dc5d5042a76ede`
- Working branch: `overnight/scientific-hardening-2026-07-11`
- Draft PR: https://github.com/an89537171644/geo_graf/pull/4
- Phase 00 code head: `85ec0e06689629b250d60ee720a37dde1c4feabf`
- Phase 01 local code head: `0474c80`

## Phase status

| Phase | Status | Commit | Local gate | GitHub CI | Notes |
|---|---|---|---|---|---|
| 00 CI | COMPLETE | `f59370f`, `85ec0e0` | 180 tests; coverage 81.54%; demo verified | [run 29169591778](https://github.com/an89537171644/geo_graf/actions/runs/29169591778): 6/6 matrix + Required CI SUCCESS | Only verified CI infrastructure; scientific modules unchanged |
| 01 E contract | LOCAL COMPLETE / CI PENDING | `0474c80` | 215 tests; core coverage 80.97%; CLI demo verified | Pending push and GitHub matrix | Owner explicitly authorised the next phase; no primary E without an approved profile, confirmed range and valid positive calculation |
| 02 Pairing | NOT STARTED | — | — | — | Strict phase order |
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

## Remaining blockers

- GitHub matrix acceptance for Phase 01 after push.
- Engineering selection/approval of a pressure range for each real primary modulus.
- Pairing schema, multi-indicator aggregation/metrology, and censored-failure presentation.
- Real laboratory acceptance on at least three experiments.
- SQLite archive and clean Windows distribution.
- Owner review and merge decision. No merge or auto-merge was performed.

Current status: research beta under author control; not a finished engineering release.
