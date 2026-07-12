# Методическая трассируемость `antonov_round_stamp_v1`

Карточка описывает runtime-профиль условного расчёта. Она не заменяет
библиографический источник и не означает инженерную или release-приёмку.

```yaml
profile_id: antonov_round_stamp_v1
profile_version: "1.0"
formula: "E_stamp_app = (1 - nu^2) * K_shape * D * dp/ds"
nu: 0.30
shape_factor: 0.80
stamp_shape: circle
source_title: null
author: null
year: null
page_or_section: null
source_file_hash_or_reference: null
applicability: "Conditional modulus for a circular rigid stamp within a confirmed range."
limitations: >-
  Conditional project calculation only; requires a confirmed circular stamp,
  an explicit approved pressure range, valid positive inputs and engineering
  completion of the bibliographic reference. It must not be presented as a
  normative deformation modulus or as an approved release method while the
  source and reviewer fields remain unfilled.
reviewer: ""
review_status: review_required_for_release
```

## Runtime traceability

- Authoritative profile definition: `soilstamp/methodology.py`, constant
  `_FORMULA` and registry entry `MODULUS_METHOD_PROFILES["antonov_round_stamp_v1"]`.
- Result trace: `moduli.csv` records `profile_id`, `profile_version`, `nu`,
  `shape_factor`, the selected pressure range, `is_primary`, `review_status` and
  `methodology_note`; the same method context is included in the report package.
- Verification trace: `scripts/verify_demo_artifacts.py` checks the required
  methodology columns and rejects a primary modulus without a finite positive
  `E_stamp_app_kPa`.

## Test and acceptance traceability

- `tests/test_methodology.py::test_antonov_profile_uses_fixed_coefficients_and_approved_explicit_range`
  checks the registered profile version and fixed coefficients together with an
  explicitly approved range.
- `tests/test_methodology.py::test_custom_shape_factor_changes_modulus_by_exact_ratio`
  checks the numerical dependence on `shape_factor` without changing the profile.
- `tests/test_methodology.py::test_antonov_profile_requires_confirmed_round_stamp_shape`
  checks that an unconfirmed stamp shape downgrades the result to engineering review.
- `tests/test_cli_methodology.py` checks CLI propagation of method decisions into
  production artifacts.
- TASK 06 acceptance outputs `acceptance_report.json`, `acceptance_report.md` and
  `acceptance_report.html` must record comparison of the E profile, range and
  scientific status for each applicable case.

`approved_for_conditional_calculation` in the runtime registry permits a conditional
calculation only when its per-experiment gates are satisfied. It is not equivalent to
`review_status: approved` for release. Until the missing source fields are completed
from a real reference and a named engineer fills `reviewer`, this document and the
profile remain `review_required_for_release`.
