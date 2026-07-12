"""Versioned, lossless primary-data models for manual plate-load drafts."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4


MANUAL_DRAFT_SCHEMA_V1_0 = "manual-entry-draft/1.0"
MANUAL_DRAFT_SCHEMA_V1_1 = "manual-entry-draft/1.1"
MANUAL_DRAFT_SCHEMA_VERSION = "manual-entry-draft/1.2"
MAX_MANUAL_DRAFT_BYTES = 16 * 1024 * 1024
MAX_MANUAL_DRAFT_ROWS = 100_000

MANUAL_TEST_SCOPES = ("laboratory", "field")
MANUAL_PROTOCOL_TYPES = ("static_step", "load_unload", "cyclic", "custom")
MANUAL_BRANCHES = ("loading", "hold", "unloading", "reloading", "cyclic")
MANUAL_ROW_STATUSES = (
    "measurement",
    "failure",
    "instrument_limit",
    "stopped_without_failure",
    "invalid",
)
MANUAL_INDICATOR_CHANNELS = (
    "indicator_1",
    "indicator_2",
    "indicator_3",
    "indicator_4",
    "reference_indicator",
)
MANUAL_VERTICAL_INDICATOR_CHANNELS = MANUAL_INDICATOR_CHANNELS[:4]
MANUAL_SETTLEMENT_AGGREGATIONS = (
    "all_channels_mean",
    "selected_channels_mean",
    "plane_center",
    "primary_channel",
    "no_aggregation",
)
MANUAL_SETTLEMENT_MISSING_CHANNEL_POLICIES = ("block", "allow_if_solvable")
MANUAL_METROLOGY_STATUSES = (
    "draft",
    "migration_review_required",
    "confirmed",
)
MANUAL_EDITOR_COLUMNS = (
    "sequence_no",
    "stage_no",
    "branch",
    "elapsed_time_s",
    "timestamp",
    "load_raw",
    "indicator_1_raw",
    "indicator_2_raw",
    "indicator_3_raw",
    "indicator_4_raw",
    "row_status",
    "comment",
)

_V1_1_COMMON_INDICATOR_FIELDS = (
    "dial_mode",
    "dial_range_mm",
    "dial_resolution_mm",
    "dial_correction_factor",
    "dial_initial_reading",
    "dial_zero_correction_mm",
    "dial_max_increment_mm",
    "dial_reverse_tolerance_mm",
    "dial_travel_range_mm",
    "indicator_type",
    "indicator_serial_numbers",
    "verification_date",
    "verification_valid_until",
)
_V1_2_PASSPORT_FIELDS = (
    "indicator_passports",
    "legacy_common_indicator_passport",
    "settlement_aggregation",
    "settlement_aggregation_channels",
    "settlement_primary_channel",
    "settlement_missing_channel_policy",
    "metrology_status",
)


def empty_manual_indicator_passports() -> dict[str, None]:
    """Return the exact channel registry for a new unassigned draft."""

    return {channel: None for channel in MANUAL_INDICATOR_CHANNELS}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_uuid() -> str:
    return str(uuid4())


def _require_uuid(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} должен быть непустым UUID.")
    try:
        UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} должен быть корректным UUID.") from exc
    return value


def _require_aware_timestamp(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} должен быть непустой временной меткой ISO 8601.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} должен быть временной меткой ISO 8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} должен содержать часовой пояс.")
    return value


def _require_exact_keys(
    payload: dict[str, Any], expected: set[str], *, object_name: str
) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"отсутствуют поля {missing!r}")
        if extra:
            details.append(f"неизвестные поля {extra!r}")
        raise ValueError(f"{object_name}: " + "; ".join(details) + ".")


def _require_text_or_none(value: Any, *, field_name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field_name} должен быть строкой или null.")
    return value


def _require_json_value(value: Any, *, field_name: str) -> None:
    """Reject values that cannot round-trip through strict RFC-style JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} содержит NaN или Infinity.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} содержит нестроковый ключ JSON.")
            _require_json_value(item, field_name=f"{field_name}.{key}")
        return
    raise ValueError(
        f"{field_name} содержит значение типа {type(value).__name__}, несовместимое с JSON."
    )


def migrate_manual_draft_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a lossless copy upgraded through every manual-draft schema.

    Version 1.0 did not have an explicit ``pair_id``.  Version 1.1 stored one
    common indicator passport.  That common object is retained verbatim as
    legacy evidence in 1.2, but it is never assigned to channels implicitly.
    Automatic migration does not append audit events because it has no human
    author or engineering justification.
    """

    if not isinstance(payload, dict):
        raise ValueError("Черновик должен быть JSON-объектом.")
    migrated = deepcopy(payload)
    version = migrated.get("schema_version")
    while version != MANUAL_DRAFT_SCHEMA_VERSION:
        if version not in {MANUAL_DRAFT_SCHEMA_V1_0, MANUAL_DRAFT_SCHEMA_V1_1}:
            raise ValueError(
                f"Неподдерживаемая версия черновика {version!r}; "
                f"ожидается {MANUAL_DRAFT_SCHEMA_VERSION}."
            )
        passport = migrated.get("passport")
        if not isinstance(passport, dict):
            raise ValueError("Поле passport должно быть JSON-объектом.")

        if version == MANUAL_DRAFT_SCHEMA_V1_0:
            if "pair_id" in passport:
                raise ValueError(
                    "passport.pair_id не входил в схему manual-entry-draft/1.0; "
                    "скрытая замена при миграции запрещена."
                )
            passport["pair_id"] = None
            migrated["schema_version"] = MANUAL_DRAFT_SCHEMA_V1_1
            version = MANUAL_DRAFT_SCHEMA_V1_1
            continue

        if version == MANUAL_DRAFT_SCHEMA_V1_1:
            collisions = sorted(set(_V1_2_PASSPORT_FIELDS) & set(passport))
            if collisions:
                raise ValueError(
                    "Паспорт manual-entry-draft/1.1 содержит поля схемы 1.2 "
                    f"{collisions!r}; скрытая перезапись при миграции запрещена."
                )
            missing_legacy = sorted(
                set(_V1_1_COMMON_INDICATOR_FIELDS) - set(passport)
            )
            if missing_legacy:
                raise ValueError(
                    "Паспорт manual-entry-draft/1.1 неполон; отсутствуют общие "
                    f"метрологические поля {missing_legacy!r}."
                )
            passport["legacy_common_indicator_passport"] = {
                name: passport.pop(name)
                for name in _V1_1_COMMON_INDICATOR_FIELDS
            }
            passport["indicator_passports"] = empty_manual_indicator_passports()
            passport["settlement_aggregation"] = "no_aggregation"
            passport["settlement_aggregation_channels"] = []
            passport["settlement_primary_channel"] = None
            passport["settlement_missing_channel_policy"] = "block"
            passport["metrology_status"] = "migration_review_required"
            migrated["schema_version"] = MANUAL_DRAFT_SCHEMA_VERSION
            version = MANUAL_DRAFT_SCHEMA_VERSION
            continue

    return migrated


@dataclass(slots=True)
class ManualReinforcement:
    material: str = ""
    number_of_layers: str | None = None
    depth_mm: str | None = None
    spacing_mm: str | None = None
    length_mm: str | None = None
    width_mm: str | None = None
    bar_diameter_or_aperture_mm: str | None = None
    orientation: str = ""
    custom_parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ManualReinforcement":
        if not isinstance(payload, dict):
            raise ValueError("passport.reinforcement должен быть JSON-объектом.")
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            object_name="passport.reinforcement",
        )
        for name in ("material", "orientation"):
            if not isinstance(payload[name], str):
                raise ValueError(f"passport.reinforcement.{name} должен быть строкой.")
        for name in (
            "number_of_layers",
            "depth_mm",
            "spacing_mm",
            "length_mm",
            "width_mm",
            "bar_diameter_or_aperture_mm",
        ):
            _require_text_or_none(
                payload[name], field_name=f"passport.reinforcement.{name}"
            )
        if not isinstance(payload["custom_parameters"], dict):
            raise ValueError(
                "passport.reinforcement.custom_parameters должен быть JSON-объектом."
            )
        _require_json_value(
            payload["custom_parameters"],
            field_name="passport.reinforcement.custom_parameters",
        )
        return cls(
            material=payload["material"],
            number_of_layers=payload["number_of_layers"],
            depth_mm=payload["depth_mm"],
            spacing_mm=payload["spacing_mm"],
            length_mm=payload["length_mm"],
            width_mm=payload["width_mm"],
            bar_diameter_or_aperture_mm=payload[
                "bar_diameter_or_aperture_mm"
            ],
            orientation=payload["orientation"],
            custom_parameters=dict(payload["custom_parameters"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ManualLegacyIndicatorCommon:
    """Verbatim version-1.1 common passport retained only for review."""

    dial_mode: str = ""
    dial_range_mm: str | None = None
    dial_resolution_mm: str | None = None
    dial_correction_factor: str | None = None
    dial_initial_reading: str | None = None
    dial_zero_correction_mm: str | None = None
    dial_max_increment_mm: str | None = None
    dial_reverse_tolerance_mm: str | None = None
    dial_travel_range_mm: str | None = None
    indicator_type: str = ""
    indicator_serial_numbers: list[str] = field(default_factory=list)
    verification_date: str = ""
    verification_valid_until: str = ""

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any] | None
    ) -> "ManualLegacyIndicatorCommon":
        if not isinstance(payload, dict):
            raise ValueError(
                "passport.legacy_common_indicator_passport должен быть JSON-объектом."
            )
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            object_name="passport.legacy_common_indicator_passport",
        )
        for name in (
            "dial_mode",
            "indicator_type",
            "verification_date",
            "verification_valid_until",
        ):
            if not isinstance(payload[name], str):
                raise ValueError(
                    f"passport.legacy_common_indicator_passport.{name} должен быть строкой."
                )
        for name in (
            "dial_range_mm",
            "dial_resolution_mm",
            "dial_correction_factor",
            "dial_initial_reading",
            "dial_zero_correction_mm",
            "dial_max_increment_mm",
            "dial_reverse_tolerance_mm",
            "dial_travel_range_mm",
        ):
            _require_text_or_none(
                payload[name],
                field_name=f"passport.legacy_common_indicator_passport.{name}",
            )
        serials = payload["indicator_serial_numbers"]
        if not isinstance(serials, list) or any(
            not isinstance(value, str) for value in serials
        ):
            raise ValueError(
                "passport.legacy_common_indicator_passport.indicator_serial_numbers "
                "должен быть массивом строк."
            )
        return cls(
            **{
                name: payload[name]
                for name in cls.__dataclass_fields__
                if name != "indicator_serial_numbers"
            },
            indicator_serial_numbers=list(serials),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ManualIndicatorPassport:
    """Raw, channel-specific metrological passport stored in a manual draft."""

    type: str = ""
    serial_number: str = ""
    instrument_id: str | None = None
    range_mm: str | None = None
    division_mm: str | None = None
    correction_factor: str | None = None
    mode: str = ""
    initial_reading: str | None = None
    initial_turn: int | None = None
    zero_correction_mm: str | None = None
    max_increment_mm: str | None = None
    reverse_tolerance_mm: str | None = None
    travel_range_mm: str | None = None
    verification_date: str = ""
    verification_valid_until: str = ""
    x_mm: str | None = None
    y_mm: str | None = None
    cumulative_sign: str | None = None
    assignment_status: str = "draft"

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any] | None, *, channel: str = "indicator"
    ) -> "ManualIndicatorPassport":
        if not isinstance(payload, dict):
            raise ValueError(f"passport.indicator_passports.{channel} должен быть JSON-объектом.")
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            object_name=f"passport.indicator_passports.{channel}",
        )
        for name in (
            "type",
            "serial_number",
            "mode",
            "verification_date",
            "verification_valid_until",
            "assignment_status",
        ):
            if not isinstance(payload[name], str):
                raise ValueError(
                    f"passport.indicator_passports.{channel}.{name} должен быть строкой."
                )
        for name in (
            "instrument_id",
            "range_mm",
            "division_mm",
            "correction_factor",
            "initial_reading",
            "zero_correction_mm",
            "max_increment_mm",
            "reverse_tolerance_mm",
            "travel_range_mm",
            "x_mm",
            "y_mm",
            "cumulative_sign",
        ):
            _require_text_or_none(
                payload[name],
                field_name=f"passport.indicator_passports.{channel}.{name}",
            )
        initial_turn = payload["initial_turn"]
        if initial_turn is not None and (
            isinstance(initial_turn, bool) or not isinstance(initial_turn, int)
        ):
            raise ValueError(
                f"passport.indicator_passports.{channel}.initial_turn "
                "должен быть целым числом или null."
            )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ManualPassport:
    project_name: str = ""
    series_name: str = ""
    test_name: str = ""
    archive_number: str = ""
    test_date: str = ""
    operator: str = ""
    laboratory_or_site: str = ""
    test_scope: str = "laboratory"
    protocol_type: str = "static_step"
    group_name: str = ""
    is_reinforced: bool = False
    baseline_group: str = ""
    pair_id: str | None = None
    soil_type: str = ""
    soil_batch: str = ""
    reinforcement_type: str = "none"
    stamp_shape: str = "circle"
    stamp_diameter_mm: str | None = None
    stamp_area_m2: str | None = None
    load_kind: str = "force"
    load_unit: str = "kN"
    load_factor: str | None = None
    load_zero: str | None = None
    lever_ratio: str | None = None
    settlement_unit: str = "mm"
    number_of_indicators: int | None = None
    indicator_passports: dict[str, ManualIndicatorPassport | None] = field(
        default_factory=empty_manual_indicator_passports
    )
    legacy_common_indicator_passport: ManualLegacyIndicatorCommon | None = None
    settlement_aggregation: str = "no_aggregation"
    settlement_aggregation_channels: list[str] = field(default_factory=list)
    settlement_primary_channel: str | None = None
    settlement_missing_channel_policy: str = "block"
    metrology_status: str = "draft"
    comment: str = ""
    reinforcement: ManualReinforcement = field(default_factory=ManualReinforcement)

    @property
    def test_id(self) -> str:
        return (self.archive_number or self.test_name).strip()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ManualPassport":
        if not isinstance(payload, dict):
            raise ValueError("Поле passport должно быть JSON-объектом.")
        _require_exact_keys(
            payload, set(cls.__dataclass_fields__), object_name="passport"
        )
        text_fields = (
            "project_name",
            "series_name",
            "test_name",
            "archive_number",
            "test_date",
            "operator",
            "laboratory_or_site",
            "test_scope",
            "protocol_type",
            "group_name",
            "baseline_group",
            "soil_type",
            "soil_batch",
            "reinforcement_type",
            "stamp_shape",
            "load_kind",
            "load_unit",
            "settlement_unit",
            "settlement_aggregation",
            "settlement_missing_channel_policy",
            "metrology_status",
            "comment",
        )
        for name in text_fields:
            if not isinstance(payload[name], str):
                raise ValueError(f"passport.{name} должен быть строкой.")
        optional_text_fields = (
            "pair_id",
            "stamp_diameter_mm",
            "stamp_area_m2",
            "load_factor",
            "load_zero",
            "lever_ratio",
            "settlement_primary_channel",
        )
        for name in optional_text_fields:
            _require_text_or_none(payload[name], field_name=f"passport.{name}")
        if not isinstance(payload["is_reinforced"], bool):
            raise ValueError("passport.is_reinforced должен быть boolean.")
        number = payload["number_of_indicators"]
        if number is not None and (
            isinstance(number, bool) or not isinstance(number, int)
        ):
            raise ValueError(
                "passport.number_of_indicators должен быть целым числом или null."
            )
        if payload["settlement_aggregation"] not in MANUAL_SETTLEMENT_AGGREGATIONS:
            raise ValueError(
                "passport.settlement_aggregation должен быть одним из: "
                + ", ".join(MANUAL_SETTLEMENT_AGGREGATIONS)
                + "."
            )
        if (
            payload["settlement_missing_channel_policy"]
            not in MANUAL_SETTLEMENT_MISSING_CHANNEL_POLICIES
        ):
            raise ValueError(
                "passport.settlement_missing_channel_policy должен быть одним из: "
                + ", ".join(MANUAL_SETTLEMENT_MISSING_CHANNEL_POLICIES)
                + "."
            )
        if payload["metrology_status"] not in MANUAL_METROLOGY_STATUSES:
            raise ValueError(
                "passport.metrology_status должен быть одним из: "
                + ", ".join(MANUAL_METROLOGY_STATUSES)
                + "."
            )

        aggregation_channels = payload["settlement_aggregation_channels"]
        if not isinstance(aggregation_channels, list) or any(
            not isinstance(value, str) for value in aggregation_channels
        ):
            raise ValueError(
                "passport.settlement_aggregation_channels должен быть массивом строк."
            )
        if any(
            value not in MANUAL_VERTICAL_INDICATOR_CHANNELS
            for value in aggregation_channels
        ):
            raise ValueError(
                "passport.settlement_aggregation_channels содержит неизвестный "
                "вертикальный канал."
            )
        if len(set(aggregation_channels)) != len(aggregation_channels):
            raise ValueError(
                "passport.settlement_aggregation_channels не должен содержать повторы."
            )
        primary = payload["settlement_primary_channel"]
        if primary is not None and primary not in MANUAL_VERTICAL_INDICATOR_CHANNELS:
            raise ValueError(
                "passport.settlement_primary_channel должен быть вертикальным "
                "каналом indicator_1..4 или null."
            )

        indicator_payload = payload["indicator_passports"]
        if not isinstance(indicator_payload, dict):
            raise ValueError("passport.indicator_passports должен быть JSON-объектом.")
        _require_exact_keys(
            indicator_payload,
            set(MANUAL_INDICATOR_CHANNELS),
            object_name="passport.indicator_passports",
        )
        indicator_passports = {
            channel: (
                None
                if indicator_payload[channel] is None
                else ManualIndicatorPassport.from_dict(
                    indicator_payload[channel], channel=channel
                )
            )
            for channel in MANUAL_INDICATOR_CHANNELS
        }
        legacy_payload = payload["legacy_common_indicator_passport"]
        legacy = (
            None
            if legacy_payload is None
            else ManualLegacyIndicatorCommon.from_dict(legacy_payload)
        )
        return cls(
            **{name: payload[name] for name in text_fields},
            is_reinforced=payload["is_reinforced"],
            **{name: payload[name] for name in optional_text_fields},
            number_of_indicators=number,
            indicator_passports=indicator_passports,
            legacy_common_indicator_passport=legacy,
            settlement_aggregation_channels=list(aggregation_channels),
            reinforcement=ManualReinforcement.from_dict(payload["reinforcement"]),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reinforcement"] = self.reinforcement.to_dict()
        payload["legacy_common_indicator_passport"] = (
            self.legacy_common_indicator_passport.to_dict()
            if self.legacy_common_indicator_passport is not None
            else None
        )
        payload["indicator_passports"] = {
            channel: (
                passport.to_dict() if passport is not None else None
            )
            for channel, passport in self.indicator_passports.items()
        }
        return payload


@dataclass(slots=True)
class ManualPoint:
    manual_row_uuid: str
    sequence_no: int
    stage_no: str | None = None
    branch: str = "loading"
    elapsed_time_s: str | None = None
    timestamp: str | None = None
    load_raw: str | None = None
    indicator_1_raw: str | None = None
    indicator_2_raw: str | None = None
    indicator_3_raw: str | None = None
    indicator_4_raw: str | None = None
    row_status: str = "measurement"
    comment: str = ""
    source_type: str = "manual"
    source_row: None = None
    created_by: str = "local-user"
    created_at: str = field(default_factory=utc_now_iso)
    modified_by: str = "local-user"
    modified_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def create(cls, sequence_no: int, *, author: str = "local-user") -> "ManualPoint":
        now = utc_now_iso()
        return cls(
            manual_row_uuid=new_uuid(),
            sequence_no=int(sequence_no),
            created_by=author,
            created_at=now,
            modified_by=author,
            modified_at=now,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManualPoint":
        if not isinstance(payload, dict):
            raise ValueError("Каждая строка ручного черновика должна быть JSON-объектом.")
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            object_name="Строка ручного черновика",
        )
        if payload.get("source_type") != "manual":
            raise ValueError("source_type каждой ручной строки должен быть равен 'manual'.")
        if payload.get("source_row") is not None:
            raise ValueError("source_row ручной строки должен быть null; скрытая замена запрещена.")
        sequence_no = payload.get("sequence_no")
        if isinstance(sequence_no, bool) or not isinstance(sequence_no, int):
            raise ValueError("sequence_no сохранённой строки должен быть целым числом.")
        optional_raw_fields = (
            "stage_no",
            "elapsed_time_s",
            "timestamp",
            "load_raw",
            "indicator_1_raw",
            "indicator_2_raw",
            "indicator_3_raw",
            "indicator_4_raw",
        )
        for name in optional_raw_fields:
            _require_text_or_none(payload[name], field_name=f"row.{name}")
        for name in (
            "branch",
            "row_status",
            "comment",
            "created_by",
            "created_at",
            "modified_by",
            "modified_at",
        ):
            if not isinstance(payload[name], str):
                raise ValueError(f"row.{name} должен быть строкой.")
        return cls(
            manual_row_uuid=_require_uuid(
                payload.get("manual_row_uuid"), field_name="manual_row_uuid"
            ),
            sequence_no=sequence_no,
            stage_no=payload["stage_no"],
            branch=payload["branch"],
            elapsed_time_s=payload["elapsed_time_s"],
            timestamp=payload["timestamp"],
            load_raw=payload["load_raw"],
            indicator_1_raw=payload["indicator_1_raw"],
            indicator_2_raw=payload["indicator_2_raw"],
            indicator_3_raw=payload["indicator_3_raw"],
            indicator_4_raw=payload["indicator_4_raw"],
            row_status=payload["row_status"],
            comment=payload["comment"],
            source_type=payload["source_type"],
            source_row=None,
            created_by=payload["created_by"],
            created_at=payload["created_at"],
            modified_by=payload["modified_by"],
            modified_at=payload["modified_at"],
        )

    def indicator_raw(self, index: int) -> str | None:
        if index not in range(1, 5):
            raise ValueError("Номер индикатора должен быть от 1 до 4.")
        return getattr(self, f"indicator_{index}_raw")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ManualAuditEvent:
    event_id: str
    author: str
    timestamp: str
    action: str
    entity_id: str
    field: str | None
    old_value: Any
    new_value: Any
    reason: str

    @classmethod
    def create(
        cls,
        *,
        author: str,
        action: str,
        entity_id: str,
        field: str | None,
        old_value: Any,
        new_value: Any,
        reason: str = "manual_edit",
    ) -> "ManualAuditEvent":
        _require_json_value(old_value, field_name="audit.old_value")
        _require_json_value(new_value, field_name="audit.new_value")
        return cls(
            event_id=new_uuid(),
            author=author,
            timestamp=utc_now_iso(),
            action=action,
            entity_id=entity_id,
            field=field,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManualAuditEvent":
        if not isinstance(payload, dict):
            raise ValueError("Каждое событие audit_events должно быть JSON-объектом.")
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            object_name="Событие audit_events",
        )
        event_id = _require_uuid(payload.get("event_id"), field_name="audit.event_id")
        timestamp = _require_aware_timestamp(
            payload.get("timestamp"), field_name="audit.timestamp"
        )
        required_text: dict[str, str] = {}
        for name in ("author", "action", "entity_id", "reason"):
            value = payload.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"audit.{name} должен быть непустой строкой.")
            required_text[name] = value
        field_value = payload.get("field")
        if field_value is not None and not isinstance(field_value, str):
            raise ValueError("audit.field должен быть строкой или null.")
        _require_json_value(payload["old_value"], field_name="audit.old_value")
        _require_json_value(payload["new_value"], field_name="audit.new_value")
        return cls(
            event_id=event_id,
            author=required_text["author"],
            timestamp=timestamp,
            action=required_text["action"],
            entity_id=required_text["entity_id"],
            field=field_value,
            old_value=payload.get("old_value"),
            new_value=payload.get("new_value"),
            reason=required_text["reason"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ManualDraft:
    draft_id: str
    passport: ManualPassport
    rows: list[ManualPoint]
    audit_events: list[ManualAuditEvent]
    created_by: str
    created_at: str
    updated_at: str
    status: str = "draft"
    schema_version: str = MANUAL_DRAFT_SCHEMA_VERSION

    @classmethod
    def create(
        cls, *, author: str = "local-user", initial_rows: int = 2
    ) -> "ManualDraft":
        if initial_rows < 0 or initial_rows > MAX_MANUAL_DRAFT_ROWS:
            raise ValueError("Недопустимое число начальных строк.")
        now = utc_now_iso()
        return cls(
            draft_id=new_uuid(),
            passport=ManualPassport(),
            rows=[ManualPoint.create(index + 1, author=author) for index in range(initial_rows)],
            audit_events=[],
            created_by=author,
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManualDraft":
        if not isinstance(payload, dict):
            raise ValueError("Черновик должен быть JSON-объектом.")
        payload = migrate_manual_draft_payload(payload)
        version = str(payload.get("schema_version") or "")
        if version != MANUAL_DRAFT_SCHEMA_VERSION:
            raise ValueError(
                f"Неподдерживаемая версия черновика {version!r}; ожидается {MANUAL_DRAFT_SCHEMA_VERSION}."
            )
        rows_payload = payload.get("rows")
        audit_payload = payload.get("audit_events")
        passport_payload = payload.get("passport")
        if not isinstance(rows_payload, list) or not isinstance(audit_payload, list):
            raise ValueError("Поля rows и audit_events должны быть JSON-массивами.")
        if not isinstance(passport_payload, dict):
            raise ValueError("Поле passport должно быть JSON-объектом.")
        _require_exact_keys(
            payload,
            set(cls.__dataclass_fields__),
            object_name="Черновик",
        )
        if len(rows_payload) > MAX_MANUAL_DRAFT_ROWS:
            raise ValueError("Черновик превышает допустимое число строк.")
        rows = [ManualPoint.from_dict(item) for item in rows_payload]
        uuids = [row.manual_row_uuid for row in rows]
        if any(not value for value in uuids) or len(set(uuids)) != len(uuids):
            raise ValueError("manual_row_uuid должны быть непустыми и уникальными.")
        sequences = [row.sequence_no for row in rows]
        if len(set(sequences)) != len(sequences):
            raise ValueError("sequence_no в сохранённом черновике должны быть уникальными.")
        audit_events = [ManualAuditEvent.from_dict(item) for item in audit_payload]
        event_ids = [event.event_id for event in audit_events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("audit.event_id должны быть уникальными.")
        draft_id = _require_uuid(payload.get("draft_id"), field_name="draft_id")
        created_by = payload.get("created_by")
        if not isinstance(created_by, str) or not created_by.strip():
            raise ValueError("created_by черновика должен быть непустой строкой.")
        created_at = _require_aware_timestamp(
            payload.get("created_at"), field_name="created_at"
        )
        updated_at = _require_aware_timestamp(
            payload.get("updated_at"), field_name="updated_at"
        )
        status = payload.get("status")
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status черновика должен быть непустой строкой.")
        return cls(
            draft_id=draft_id,
            passport=ManualPassport.from_dict(passport_payload),
            rows=rows,
            audit_events=audit_events,
            created_by=created_by,
            created_at=created_at,
            updated_at=updated_at,
            status=status,
            schema_version=version,
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> "ManualDraft":
        raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        if len(raw) > MAX_MANUAL_DRAFT_BYTES:
            raise ValueError("Файл черновика превышает допустимый размер.")
        def reject_constant(value: str) -> None:
            raise ValueError(f"Недопустимая JSON-константа {value}.")

        try:
            data = json.loads(
                raw.decode("utf-8-sig"), parse_constant=reject_constant
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Черновик JSON повреждён: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "draft_id": self.draft_id,
            "status": self.status,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "passport": self.passport.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "audit_events": [event.to_dict() for event in self.audit_events],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
            separators=(",", ":") if indent is None else None,
        )

    @property
    def sha256(self) -> str:
        canonical = self.to_json(indent=None).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
