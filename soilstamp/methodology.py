"""Versioned methodology contract for the conditional stamp modulus.

The resolver deliberately works with raw metadata instead of
``data.metadata_for_test``: injected compatibility defaults must never look
like an engineer-approved scientific decision.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence


APPROVED = "approved"
REVIEW_REQUIRED = "review_required"
RANGE_SOURCES = ("explicit", "accepted_pcr", "project_profile")


@dataclass(frozen=True, slots=True)
class ModulusMethodProfile:
    """Immutable, versioned definition of a modulus calculation profile."""

    profile_id: str
    profile_version: str
    formula: str
    source_description: str
    applicability: str
    approval_status: str
    nu: float | None
    shape_factor: float | None
    stamp_shape: str | None
    requires_explicit_range: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModulusOverrides:
    """One explicit decision layer supplied by the GUI or CLI."""

    profile_id: str | None = None
    p_range_kpa: tuple[float, float] | None = None
    p_range_source: str | None = None
    nu: float | None = None
    shape_factor: float | None = None
    approval_status: str | None = None
    author: str | None = None
    timestamp_utc: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class ModulusResolution:
    """Fully resolved parameters plus their provenance and review state."""

    profile: ModulusMethodProfile
    nu: float
    shape_factor: float
    p_min_kpa: float | None
    p_max_kpa: float | None
    p_range_source: str
    nu_source: str
    shape_factor_source: str
    profile_source: str
    p_range_origin: str
    is_primary: bool
    review_status: str
    methodology_note: str
    approval_author: str | None = None
    approval_timestamp_utc: str | None = None
    approval_reason: str | None = None
    accepted_pcr_kpa: float | None = None

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    @property
    def profile_version(self) -> str:
        return self.profile.profile_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_definition": self.profile.to_dict(),
            "nu": self.nu,
            "shape_factor": self.shape_factor,
            "p_min_kPa": self.p_min_kpa,
            "p_max_kPa": self.p_max_kpa,
            "p_range_source": self.p_range_source,
            "nu_source": self.nu_source,
            "shape_factor_source": self.shape_factor_source,
            "profile_source": self.profile_source,
            "p_range_origin": self.p_range_origin,
            "is_primary": self.is_primary,
            "review_status": self.review_status,
            "methodology_note": self.methodology_note,
            "approval_author": self.approval_author,
            "approval_timestamp_utc": self.approval_timestamp_utc,
            "approval_reason": self.approval_reason,
            "accepted_pcr_kPa": self.accepted_pcr_kpa,
        }


_FORMULA = "E_stamp_app = (1 - nu^2) * K_shape * D * dp/ds"

MODULUS_METHOD_PROFILES: dict[str, ModulusMethodProfile] = {
    "antonov_round_stamp_v1": ModulusMethodProfile(
        profile_id="antonov_round_stamp_v1",
        profile_version="1.0",
        formula=_FORMULA,
        source_description=(
            "Project-specified Antonov round-stamp profile; exact bibliographic "
            "citation is pending engineering review."
        ),
        applicability="Conditional modulus for a circular rigid stamp within a confirmed range.",
        approval_status="approved_for_conditional_calculation",
        nu=0.30,
        shape_factor=0.80,
        stamp_shape="circle",
        requires_explicit_range=True,
    ),
    "custom_v1": ModulusMethodProfile(
        profile_id="custom_v1",
        profile_version="1.0",
        formula=_FORMULA,
        source_description="Project-defined coefficients and range with recorded approval.",
        applicability="Conditional project calculation after all parameters are specified.",
        approval_status="requires_project_approval",
        nu=None,
        shape_factor=None,
        stamp_shape=None,
        requires_explicit_range=True,
    ),
    "diagnostic_unapproved_v1": ModulusMethodProfile(
        profile_id="diagnostic_unapproved_v1",
        profile_version="1.0",
        formula=_FORMULA,
        source_description="Compatibility-only diagnostic profile; not an approved method.",
        applicability="Numerical diagnostics and legacy compatibility only.",
        approval_status="unapproved",
        nu=0.30,
        shape_factor=1.00,
        stamp_shape=None,
        requires_explicit_range=False,
    ),
}


def get_modulus_profile(profile_id: str) -> ModulusMethodProfile:
    """Return a registered profile or fail without guessing a replacement."""

    try:
        return MODULUS_METHOD_PROFILES[str(profile_id)]
    except KeyError as exc:
        available = ", ".join(MODULUS_METHOD_PROFILES)
        raise ValueError(f"Неизвестный профиль модуля {profile_id!r}; доступны: {available}.") from exc


def modulus_profile_ids() -> tuple[str, ...]:
    return tuple(MODULUS_METHOD_PROFILES)


def modulus_profile_definitions() -> list[dict[str, Any]]:
    return [profile.to_dict() for profile in MODULUS_METHOD_PROFILES.values()]


def parse_pressure_range(value: str | Sequence[float]) -> tuple[float, float]:
    """Parse and validate a closed pressure range without interpolation."""

    if isinstance(value, str):
        parts = [part.strip().replace(",", ".") for part in value.split(":")]
        if len(parts) != 2 or not all(parts):
            raise ValueError("Диапазон E задаётся как P_MIN:P_MAX.")
        raw_lower, raw_upper = parts
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) != 2:
            raise ValueError("Диапазон E должен содержать ровно две границы.")
        raw_lower, raw_upper = value
    else:
        raise ValueError("Диапазон E задаётся как P_MIN:P_MAX.")
    try:
        lower, upper = float(raw_lower), float(raw_upper)
    except (TypeError, ValueError) as exc:
        raise ValueError("Границы диапазона E должны быть числами.") from exc
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("Границы диапазона E должны быть конечными.")
    if not lower < upper:
        raise ValueError("Для диапазона E требуется P_MIN < P_MAX.")
    return lower, upper


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, ModulusOverrides):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("Слой параметров методики должен быть объектом.")


def _method_config(raw: Mapping[str, Any], *, decision_layer: bool = False) -> dict[str, Any]:
    nested = raw.get("modulus_method")
    if isinstance(nested, Mapping):
        return dict(nested)
    return dict(raw) if decision_layer else {}


def _configured_value(
    raw: Mapping[str, Any],
    config: Mapping[str, Any],
    key: str,
    *,
    allow_legacy: bool,
) -> tuple[Any, str | None]:
    if key in config and config.get(key) is not None:
        return config.get(key), "method"
    if not allow_legacy:
        return None, None
    legacy_key = {"nu": "poisson_ratio", "p_range_kpa": "e_range_kPa"}.get(key, key)
    if legacy_key in raw and raw.get(legacy_key) is not None:
        return raw.get(legacy_key), "legacy"
    return None, None


def _approval_from(config: Mapping[str, Any]) -> dict[str, Any]:
    approval = config.get("approval")
    result = dict(approval) if isinstance(approval, Mapping) else {}
    aliases = {
        "approval_status": "status",
        "author": "author",
        "timestamp_utc": "timestamp_utc",
        "reason": "reason",
    }
    for source_key, target_key in aliases.items():
        if config.get(source_key) is not None:
            result[target_key] = config.get(source_key)
    return result


def _approval_complete(approval: Mapping[str, Any]) -> bool:
    status = str(approval.get("status") or "").casefold()
    return (
        status in {"approved", "confirmed", "accepted"}
        and all(str(approval.get(key) or "").strip() for key in ("author", "reason"))
        and _valid_timestamp(approval.get("timestamp_utc"))
    )


def _accepted_pcr_from(config: Mapping[str, Any]) -> dict[str, Any]:
    accepted = config.get("accepted_pcr")
    return dict(accepted) if isinstance(accepted, Mapping) else {}


def _accepted_pcr_complete(record: Mapping[str, Any]) -> bool:
    try:
        value = float(record.get("value_kPa"))
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(value)
        and all(
            str(record.get(key) or "").strip()
            for key in ("accepted_by", "reason")
        )
        and _valid_timestamp(record.get("accepted_at"))
    )


def _valid_timestamp(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def resolve_modulus_method(
    metadata: dict[str, Any] | None,
    test_id: str,
    *,
    overrides: ModulusOverrides | Mapping[str, Any] | None = None,
    manual_confirmation: ModulusOverrides | Mapping[str, Any] | None = None,
    pcr_result: Any | None = None,
    available_p_range: tuple[float, float] | None = None,
) -> ModulusResolution:
    """Resolve one test with precedence global < test < manual < CLI.

    A missing or unapproved range produces a numerical diagnostic contract;
    it never authorises the observed full curve as a primary result.
    """

    raw_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    tests = raw_metadata.get("tests")
    test_raw = (
        dict(tests.get(str(test_id)) or {})
        if isinstance(tests, Mapping) and isinstance(tests.get(str(test_id)), Mapping)
        else {}
    )
    global_raw = {key: value for key, value in raw_metadata.items() if key != "tests"}
    manual_raw = _mapping(manual_confirmation)
    override_raw = _mapping(overrides)
    layers: list[tuple[str, dict[str, Any], bool]] = [
        ("global_metadata", global_raw, False),
        (f"metadata.tests.{test_id}", test_raw, False),
        ("manual_confirmation", manual_raw, True),
        ("cli_override", override_raw, True),
    ]

    profile_id = "diagnostic_unapproved_v1"
    profile_source = "registry_default"
    profile_decision_level = -1
    for level, (source, raw, decision_layer) in enumerate(layers):
        config = _method_config(raw, decision_layer=decision_layer)
        candidate = config.get("profile_id", raw.get("method_profile"))
        if candidate is not None:
            candidate_id = str(candidate)
            if candidate_id != profile_id:
                profile_decision_level = level
            profile_id = candidate_id
            profile_source = f"{source}.profile_id"
    profile = get_modulus_profile(profile_id)
    allow_legacy = profile_id in {"custom_v1", "diagnostic_unapproved_v1"}

    nu = profile.nu
    shape_factor = profile.shape_factor
    nu_source = f"profile:{profile_id}" if nu is not None else "missing"
    shape_source = f"profile:{profile_id}" if shape_factor is not None else "missing"
    p_range: tuple[float, float] | None = None
    range_origin = "missing"
    range_source_value: str | None = None
    approval: dict[str, Any] = {}
    approval_level = -1
    accepted_pcr_record: dict[str, Any] = {}
    accepted_pcr_level = -1
    decision_level = profile_decision_level

    for level, (source, raw, decision_layer) in enumerate(layers):
        config = _method_config(raw, decision_layer=decision_layer)
        value, kind = _configured_value(raw, config, "nu", allow_legacy=allow_legacy)
        if value is not None:
            try:
                candidate_nu = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Некорректный nu в {source}.") from exc
            if nu is None or not math.isclose(candidate_nu, float(nu), abs_tol=1e-12):
                decision_level = max(decision_level, level)
            nu = candidate_nu
            suffix = "modulus_method.nu" if kind == "method" else "poisson_ratio"
            nu_source = f"{source}.{suffix}"
        value, kind = _configured_value(
            raw, config, "shape_factor", allow_legacy=allow_legacy
        )
        if value is not None:
            try:
                candidate_shape = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Некорректный shape_factor в {source}.") from exc
            if shape_factor is None or not math.isclose(
                candidate_shape, float(shape_factor), abs_tol=1e-12
            ):
                decision_level = max(decision_level, level)
            shape_factor = candidate_shape
            suffix = "modulus_method.shape_factor" if kind == "method" else "shape_factor"
            shape_source = f"{source}.{suffix}"
        value, kind = _configured_value(
            raw, config, "p_range_kpa", allow_legacy=allow_legacy
        )
        if value is None and config.get("p_range_kPa") is not None:
            value, kind = config.get("p_range_kPa"), "method"
        if value is not None:
            candidate_range = parse_pressure_range(value)
            if p_range != candidate_range:
                decision_level = max(decision_level, level)
            p_range = candidate_range
            suffix = "modulus_method.p_range_kPa" if kind == "method" else "e_range_kPa"
            range_origin = f"{source}.{suffix}"
        if config.get("p_range_source") is not None:
            candidate_source = str(config.get("p_range_source"))
            if candidate_source != range_source_value:
                decision_level = max(decision_level, level)
            range_source_value = candidate_source
        layer_approval = _approval_from(config)
        if layer_approval:
            # An approval record is atomic.  Fields from an older decision
            # must not silently complete a newer, partial confirmation.
            approval = layer_approval
            approval_level = level
        layer_accepted = _accepted_pcr_from(config)
        if not layer_accepted and isinstance(raw.get("accepted_pcr"), Mapping):
            layer_accepted = dict(raw["accepted_pcr"])
        if layer_accepted:
            accepted_pcr_record = layer_accepted
            accepted_pcr_level = level

    if nu is None or not math.isfinite(float(nu)) or not 0 <= float(nu) < 0.5:
        raise ValueError("Для методики E требуется явный nu в диапазоне [0; 0,5).")
    if shape_factor is None or not math.isfinite(float(shape_factor)) or shape_factor <= 0:
        raise ValueError("Для методики E требуется положительный shape_factor.")

    issues: list[str] = []
    supplied_range = p_range is not None
    if range_source_value is not None and range_source_value not in RANGE_SOURCES:
        raise ValueError(
            "p_range_source должен быть explicit, accepted_pcr или project_profile."
        )
    if supplied_range and range_source_value is None:
        range_source_value = "explicit"
    if not supplied_range:
        if available_p_range is not None:
            p_range = parse_pressure_range(available_p_range)
        range_source_value = "diagnostic_full_curve"
        range_origin = "observed_data"
        issues.append("подтверждённый диапазон давления не задан")

    range_covered = True
    if supplied_range and p_range is not None and available_p_range is not None:
        available_lower, available_upper = parse_pressure_range(available_p_range)
        tolerance = max(abs(available_lower), abs(available_upper), 1.0) * 1e-12
        range_covered = bool(
            p_range[0] >= available_lower - tolerance
            and p_range[1] <= available_upper + tolerance
        )
        if not range_covered:
            issues.append("запрошенный диапазон выходит за наблюдённые устойчивые точки")

    accepted_value: float | None = None
    accepted_complete = False
    acceptance_level = -1
    if _accepted_pcr_complete(accepted_pcr_record):
        accepted_value = float(accepted_pcr_record["value_kPa"])
        accepted_complete = True
        acceptance_level = accepted_pcr_level
    if (
        pcr_result is not None
        and getattr(pcr_result, "pcr_manual", None) is not None
        and acceptance_level <= 2
    ):
        manual_reason = str(getattr(pcr_result, "manual_reason", None) or "").strip()
        manual_author = str(getattr(pcr_result, "manual_author", None) or "").strip()
        manual_time = str(
            getattr(pcr_result, "manual_confirmed_at_utc", None) or ""
        ).strip()
        if manual_reason:
            accepted_value = float(pcr_result.pcr_manual)
            accepted_complete = bool(manual_author and _valid_timestamp(manual_time))
            acceptance_level = 2

    approval_valid_for_decision = bool(
        _approval_complete(approval) and approval_level >= decision_level
    )
    range_approved = False
    if range_source_value == "explicit":
        range_approved = supplied_range and range_covered and approval_valid_for_decision
        if not range_approved:
            issues.append("явный диапазон не имеет полного подтверждения author/time/reason")
    elif range_source_value == "project_profile":
        metadata_origin = range_origin.startswith("global_metadata") or range_origin.startswith(
            "metadata.tests."
        )
        range_approved = (
            supplied_range
            and range_covered
            and metadata_origin
            and approval_valid_for_decision
        )
        if not range_approved:
            issues.append("проектный диапазон не задан и не утверждён полностью в metadata")
    elif range_source_value == "accepted_pcr":
        if accepted_value is None:
            issues.append("accepted_pcr отсутствует; автоматический pcr не принят")
        elif p_range is None or not math.isclose(
            float(p_range[1]), accepted_value, rel_tol=1e-9, abs_tol=1e-9
        ):
            issues.append("верхняя граница диапазона не совпадает с подтверждённым pcr")
        elif not accepted_complete:
            issues.append("подтверждение pcr не содержит автора и времени")
        else:
            range_approved = (
                supplied_range and range_covered and acceptance_level >= decision_level
            )
            if not range_approved:
                issues.append("подтверждение pcr относится к более раннему контракту диапазона")

    shape = str(test_raw.get("stamp_shape", global_raw.get("stamp_shape", ""))).casefold()
    if profile.stamp_shape:
        if not shape:
            issues.append("форма штампа не подтверждена для профиля")
        elif shape not in {"circle", "round", "круг", "круглый"}:
            issues.append("форма штампа не соответствует профилю")
    if profile.nu is not None and not math.isclose(float(nu), profile.nu, abs_tol=1e-12):
        issues.append("nu отличается от фиксированного значения профиля")
    if profile.shape_factor is not None and not math.isclose(
        float(shape_factor), profile.shape_factor, abs_tol=1e-12
    ):
        issues.append("shape_factor отличается от фиксированного значения профиля")
    if profile.profile_id == "diagnostic_unapproved_v1":
        issues.append("диагностический профиль не утверждён для основного результата")
    profile_approved = profile.approval_status == "approved_for_conditional_calculation"
    if profile.profile_id == "custom_v1":
        profile_approved = approval_valid_for_decision
        if not profile_approved:
            issues.append("custom-профиль не имеет полного проектного утверждения")

    is_primary = bool(profile_approved and range_approved and not issues)
    review_status = APPROVED if is_primary else REVIEW_REQUIRED
    if is_primary:
        note = (
            "Условный штамповый модуль рассчитан по утверждённому профилю и "
            "подтверждённому диапазону; это не заявление о нормативном модуле."
        )
    else:
        unique_issues = list(dict.fromkeys(issues))
        note = "Диагностический результат; требуется инженерная проверка: " + "; ".join(
            unique_issues or ["методический контракт неполон"]
        )
    return ModulusResolution(
        profile=profile,
        nu=float(nu),
        shape_factor=float(shape_factor),
        p_min_kpa=float(p_range[0]) if p_range is not None else None,
        p_max_kpa=float(p_range[1]) if p_range is not None else None,
        p_range_source=str(range_source_value),
        nu_source=nu_source,
        shape_factor_source=shape_source,
        profile_source=profile_source,
        p_range_origin=range_origin,
        is_primary=is_primary,
        review_status=review_status,
        methodology_note=note,
        approval_author=str(approval.get("author") or "").strip() or None,
        approval_timestamp_utc=str(approval.get("timestamp_utc") or "").strip() or None,
        approval_reason=str(approval.get("reason") or "").strip() or None,
        accepted_pcr_kpa=accepted_value,
    )


def legacy_modulus_resolution(
    *,
    p_min_kpa: float | None,
    p_max_kpa: float | None,
    nu: float | None,
    shape_factor: float | None,
    available_p_range: tuple[float, float],
) -> ModulusResolution:
    """Preserve legacy numbers while labelling them unapproved diagnostics."""

    if (p_min_kpa is None) != (p_max_kpa is None):
        raise ValueError("Legacy-диапазон требует одновременно p_min_kpa и p_max_kpa.")
    explicit = p_min_kpa is not None
    selected_range = (
        parse_pressure_range((p_min_kpa, p_max_kpa))
        if explicit
        else parse_pressure_range(available_p_range)
    )
    profile = get_modulus_profile("diagnostic_unapproved_v1")
    resolved_nu = profile.nu if nu is None else float(nu)
    resolved_shape = profile.shape_factor if shape_factor is None else float(shape_factor)
    if resolved_nu is None or not 0 <= resolved_nu < 0.5:
        raise ValueError("Коэффициент Пуассона должен быть в диапазоне [0; 0,5).")
    if resolved_shape is None or resolved_shape <= 0:
        raise ValueError("Коэффициент формы должен быть положительным.")
    return ModulusResolution(
        profile=profile,
        nu=resolved_nu,
        shape_factor=resolved_shape,
        p_min_kpa=selected_range[0],
        p_max_kpa=selected_range[1],
        p_range_source="explicit" if explicit else "diagnostic_full_curve",
        nu_source="legacy_argument" if nu is not None else f"profile:{profile.profile_id}",
        shape_factor_source=(
            "legacy_argument" if shape_factor is not None else f"profile:{profile.profile_id}"
        ),
        profile_source="legacy_api",
        p_range_origin="legacy_argument" if explicit else "observed_data",
        is_primary=False,
        review_status=REVIEW_REQUIRED,
        methodology_note=(
            "Legacy API: численное значение сохранено только как diagnostic/unapproved; "
            "для основного результата используйте resolve_modulus_method()."
        ),
    )
