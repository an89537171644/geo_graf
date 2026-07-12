"""Mutation and clipboard services for lossless manual-entry drafts.

The service deliberately keeps raw measurement values as text.  Parsing and
scientific validation belong to the manual-entry adapter/validation layers;
this module only provides auditable editor operations.
"""

from __future__ import annotations

import copy
import csv
import io
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .manual_entry_models import (
    MANUAL_EDITOR_COLUMNS,
    MANUAL_INDICATOR_CHANNELS,
    MAX_MANUAL_DRAFT_ROWS,
    ManualAuditEvent,
    ManualDraft,
    ManualPassport,
    ManualPoint,
    new_uuid,
    utc_now_iso,
)


EDITOR_UUID_COLUMN = "manual_row_uuid"
EDITOR_COLUMNS = (EDITOR_UUID_COLUMN, *MANUAL_EDITOR_COLUMNS)
_TEXT_FIELDS = {"branch", "row_status", "comment"}
_OPTIONAL_TEXT_FIELDS = set(MANUAL_EDITOR_COLUMNS) - _TEXT_FIELDS - {"sequence_no"}
_PROTECTED_EDITOR_FIELDS = {EDITOR_UUID_COLUMN}
_DEFAULT_PASSPORT_UPDATE_REASON = "manual_edit"
_METROLOGY_SENSITIVE_PASSPORT_FIELDS = frozenset(
    {
        "test_date",
        "number_of_indicators",
        "indicator_passports",
        "legacy_common_indicator_passport",
        "settlement_aggregation",
        "settlement_aggregation_channels",
        "settlement_primary_channel",
        "settlement_missing_channel_policy",
    }
)


class ManualEntryServiceError(ValueError):
    """Base exception raised for rejected editor operations."""


class ConfirmationRequiredError(ManualEntryServiceError):
    """Raised when a destructive operation lacks explicit confirmation."""


class ClipboardShapeError(ManualEntryServiceError):
    """Raised for a non-rectangular or out-of-bounds clipboard block."""


class EditorConflictError(ManualEntryServiceError):
    """Raised when an editor frame cannot be reconciled with the draft."""


class HistoryEmptyError(ManualEntryServiceError):
    """Raised when undo or redo has no command to apply."""


@dataclass(slots=True)
class _ContentSnapshot:
    passport: dict[str, Any]
    rows: list[dict[str, Any]]
    status: str
    audit_len: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "passport": copy.deepcopy(self.passport),
            "rows": copy.deepcopy(self.rows),
            "status": self.status,
        }


@dataclass(slots=True)
class _HistoryCommand:
    command_id: str
    action: str
    before: _ContentSnapshot
    after: _ContentSnapshot
    effects: tuple["_HistoryEffect", ...]


@dataclass(slots=True)
class _HistoryEffect:
    action: str
    entity_id: str
    field: str | None
    old_value: Any
    new_value: Any


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _editor_value(field: str, value: Any) -> Any:
    """Normalize a widget scalar without parsing decimal text."""

    if field == "sequence_no":
        if _is_missing(value) or value == "":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise EditorConflictError("sequence_no должен быть целым числом.") from exc
        if not numeric.is_integer():
            raise EditorConflictError("sequence_no должен быть целым числом.")
        return int(numeric)
    if _is_missing(value):
        return "" if field in _TEXT_FIELDS else None
    if field in _TEXT_FIELDS or field in _OPTIONAL_TEXT_FIELDS:
        return str(value)
    return value


def _clipboard_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def parse_rectangular_tsv(payload: str) -> list[list[str | None]]:
    """Parse an Excel-compatible TSV block and require a rectangle.

    Empty clipboard cells become ``None``.  All non-empty values, including
    decimal-comma numbers, remain exact strings.
    """

    if not isinstance(payload, str):
        raise ClipboardShapeError("Буфер обмена должен быть текстом TSV.")
    try:
        rows = list(csv.reader(io.StringIO(payload), delimiter="\t"))
    except csv.Error as exc:
        raise ClipboardShapeError(f"Некорректный TSV-блок: {exc}") from exc
    if not rows:
        raise ClipboardShapeError("Буфер обмена пуст.")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ClipboardShapeError("Вставляемый TSV-блок должен быть прямоугольным.")
    if next(iter(widths)) == 0:
        raise ClipboardShapeError("Буфер обмена не содержит ячеек.")
    return [[value if value != "" else None for value in row] for row in rows]


def editor_frame(
    draft: ManualDraft, n_indicators: int | None = None
) -> pd.DataFrame:
    """Return an object-typed, lossless editor view of draft rows.

    ``manual_row_uuid`` is intentionally included as a hidden identity column.
    Limiting visible indicator columns never deletes values from the draft when
    the frame is later applied.
    """

    if n_indicators is None:
        n_indicators = draft.passport.number_of_indicators
    if n_indicators is None:
        n_indicators = 4
    try:
        n_indicators = int(n_indicators)
    except (TypeError, ValueError) as exc:
        raise ManualEntryServiceError("Число индикаторов должно быть целым.") from exc
    if n_indicators < 0 or n_indicators > 4:
        raise ManualEntryServiceError("Поддерживается от 0 до 4 индикаторов.")

    columns = [
        EDITOR_UUID_COLUMN,
        *(
            column
            for column in MANUAL_EDITOR_COLUMNS
            if not column.startswith("indicator_")
            or int(column.split("_")[1]) <= n_indicators
        ),
    ]
    records = []
    for row in draft.rows:
        payload = row.to_dict()
        records.append({column: payload.get(column) for column in columns})
    return pd.DataFrame(records, columns=columns, dtype=object)


class ManualEntryService:
    """Stateful, append-only command service for one manual draft."""

    def __init__(
        self, draft: ManualDraft | None = None, *, author: str = "local-user"
    ) -> None:
        self.draft = draft or ManualDraft.create(author=author)
        self.author = author
        self._undo_stack: list[_HistoryCommand] = []
        self._redo_stack: list[_HistoryCommand] = []

    @classmethod
    def from_json(
        cls, payload: str | bytes, *, author: str = "local-user"
    ) -> "ManualEntryService":
        """Open a versioned draft without normalizing its raw values."""

        return cls(ManualDraft.from_json(payload), author=author)

    def to_json(self, *, indent: int | None = 2) -> str:
        return self.draft.to_json(indent=indent)

    def editor_frame(self, n_indicators: int | None = None) -> pd.DataFrame:
        return editor_frame(self.draft, n_indicators=n_indicators)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def _author(self, author: str | None) -> str:
        return author or self.author

    def _snapshot(self) -> _ContentSnapshot:
        return _ContentSnapshot(
            passport=copy.deepcopy(self.draft.passport.to_dict()),
            rows=[copy.deepcopy(row.to_dict()) for row in self.draft.rows],
            status=self.draft.status,
            audit_len=len(self.draft.audit_events),
        )

    def _restore(self, snapshot: _ContentSnapshot) -> None:
        self.draft.passport = ManualPassport.from_dict(
            copy.deepcopy(snapshot.passport)
        )
        self.draft.rows = [
            ManualPoint.from_dict(copy.deepcopy(payload))
            for payload in snapshot.rows
        ]
        self.draft.status = snapshot.status
        self.draft.updated_at = utc_now_iso()

    def _event(
        self,
        *,
        author: str,
        action: str,
        entity_id: str,
        field: str | None,
        old_value: Any,
        new_value: Any,
        reason: str,
    ) -> None:
        self.draft.audit_events.append(
            ManualAuditEvent.create(
                author=author,
                action=action,
                entity_id=entity_id,
                field=field,
                old_value=copy.deepcopy(old_value),
                new_value=copy.deepcopy(new_value),
                reason=reason,
            )
        )

    def _finish(
        self, *, action: str, before: _ContentSnapshot
    ) -> bool:
        after = self._snapshot()
        if before.to_dict() == after.to_dict():
            return False
        self.draft.updated_at = utc_now_iso()
        self._undo_stack.append(
            _HistoryCommand(
                command_id=new_uuid(),
                action=action,
                before=before,
                after=after,
                effects=tuple(
                    _HistoryEffect(
                        action=event.action,
                        entity_id=event.entity_id,
                        field=event.field,
                        old_value=copy.deepcopy(event.old_value),
                        new_value=copy.deepcopy(event.new_value),
                    )
                    for event in self.draft.audit_events[before.audit_len :]
                ),
            )
        )
        self._redo_stack.clear()
        return True

    def _row_index(self, row: int | str, *, allow_end: bool = False) -> int:
        if isinstance(row, int):
            maximum = len(self.draft.rows) if allow_end else len(self.draft.rows) - 1
            if row < 0 or row > maximum:
                raise ManualEntryServiceError("Индекс строки вне таблицы.")
            return row
        for index, point in enumerate(self.draft.rows):
            if point.manual_row_uuid == row:
                return index
        raise ManualEntryServiceError(f"Строка {row!r} не найдена.")

    @staticmethod
    def _column_index(column: int | str) -> int:
        if isinstance(column, int):
            if column < 0 or column >= len(MANUAL_EDITOR_COLUMNS):
                raise ManualEntryServiceError("Индекс столбца вне таблицы.")
            return column
        try:
            return MANUAL_EDITOR_COLUMNS.index(column)
        except ValueError as exc:
            raise ManualEntryServiceError(f"Столбец {column!r} не редактируется.") from exc

    @staticmethod
    def _mark_modified(point: ManualPoint, author: str) -> None:
        point.modified_by = author
        point.modified_at = utc_now_iso()

    def _renumber_internal(
        self, *, author: str, reason: str, action: str = "renumber"
    ) -> int:
        changes = 0
        for sequence, point in enumerate(self.draft.rows, start=1):
            if point.sequence_no == sequence:
                continue
            old = point.sequence_no
            point.sequence_no = sequence
            self._mark_modified(point, author)
            self._event(
                author=author,
                action=action,
                entity_id=point.manual_row_uuid,
                field="sequence_no",
                old_value=old,
                new_value=sequence,
                reason=reason,
            )
            changes += 1
        return changes

    def update_passport(
        self,
        changes: Mapping[str, Any],
        *,
        author: str | None = None,
        reason: str = "manual_edit",
    ) -> bool:
        """Update passport or ``reinforcement.<field>`` values atomically."""

        if not isinstance(changes, Mapping):
            raise ManualEntryServiceError("Изменения паспорта должны быть словарём.")
        actor = self._author(author)
        before = self._snapshot()
        passport_payload = self.draft.passport.to_dict()
        valid_passport = set(passport_payload) - {"reinforcement"}
        reinforcement_payload = dict(passport_payload["reinforcement"])
        valid_reinforcement = set(reinforcement_payload)
        requested: list[tuple[str, Any, Any]] = []

        for field, value in changes.items():
            if field.startswith("reinforcement."):
                nested = field.split(".", 1)[1]
                if nested not in valid_reinforcement:
                    raise ManualEntryServiceError(f"Неизвестное поле паспорта {field!r}.")
                old = reinforcement_payload[nested]
                reinforcement_payload[nested] = copy.deepcopy(value)
                requested.append((field, old, value))
            else:
                if field not in valid_passport:
                    raise ManualEntryServiceError(f"Неизвестное поле паспорта {field!r}.")
                old = passport_payload[field]
                normalized_value = copy.deepcopy(value)
                if field == "number_of_indicators" and value is not None:
                    if isinstance(value, bool):
                        raise ManualEntryServiceError(
                            "number_of_indicators должен быть целым числом."
                        )
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ManualEntryServiceError(
                            "number_of_indicators должен быть целым числом."
                        ) from exc
                    if not numeric.is_integer():
                        raise ManualEntryServiceError(
                            "number_of_indicators должен быть целым числом."
                        )
                    normalized_value = int(numeric)
                passport_payload[field] = normalized_value
                requested.append((field, old, value))

        sensitive_changed = any(
            before.passport.get(field) != passport_payload.get(field)
            for field in _METROLOGY_SENSITIVE_PASSPORT_FIELDS
        )
        final_status_requested = passport_payload.get("metrology_status")
        normalized_reason = reason.strip() if isinstance(reason, str) else ""
        explicit_confirmation = bool(
            final_status_requested == "confirmed"
            and "metrology_status" in changes
            and changes["metrology_status"] == "confirmed"
            and normalized_reason
            and normalized_reason != _DEFAULT_PASSPORT_UPDATE_REASON
        )
        confirmation_transition = bool(
            before.passport.get("metrology_status") != "confirmed"
            and final_status_requested == "confirmed"
        )
        if confirmation_transition and not explicit_confirmation:
            raise ManualEntryServiceError(
                "Для подтверждения метрологии требуется явное инженерное "
                "обоснование reason, отличное от manual_edit."
            )
        explicit_reconfirmation = bool(
            sensitive_changed and explicit_confirmation
        )
        auto_invalidated = bool(
            sensitive_changed
            and final_status_requested == "confirmed"
            and not explicit_confirmation
        )
        if auto_invalidated:
            passport_payload["metrology_status"] = "draft"

        passport_payload["reinforcement"] = reinforcement_payload
        replacement = ManualPassport.from_dict(passport_payload)
        normalized = replacement.to_dict()

        prepared_events: list[ManualAuditEvent] = []
        for field, old, _ in requested:
            if field.startswith("reinforcement."):
                new = normalized["reinforcement"][field.split(".", 1)[1]]
            else:
                new = normalized[field]
            if old == new:
                continue
            if auto_invalidated and field == "metrology_status":
                continue
            prepared_events.append(
                ManualAuditEvent.create(
                    author=actor,
                    action="update_passport",
                    entity_id=f"{self.draft.draft_id}:passport",
                    field=field,
                    old_value=old,
                    new_value=new,
                    reason=reason,
                )
            )
        if auto_invalidated:
            previous_status = before.passport["metrology_status"]
            prepared_events.append(
                ManualAuditEvent.create(
                    author=actor,
                    action=(
                        "invalidate_metrology_confirmation"
                        if previous_status == "confirmed"
                        else "reject_metrology_confirmation"
                    ),
                    entity_id=f"{self.draft.draft_id}:passport",
                    field="metrology_status",
                    old_value=previous_status,
                    new_value="draft",
                    reason=reason,
                )
            )
        elif explicit_reconfirmation:
            prepared_events.append(
                ManualAuditEvent.create(
                    author=actor,
                    action="reconfirm_metrology_after_update",
                    entity_id=f"{self.draft.draft_id}:passport",
                    field="metrology_status",
                    old_value=before.passport["metrology_status"],
                    new_value="confirmed",
                    reason=reason,
                )
            )

        if before.passport == normalized and not prepared_events:
            return False

        candidate_draft_payload = self.draft.to_dict()
        candidate_draft_payload["passport"] = normalized
        candidate_draft_payload["audit_events"] = [
            *candidate_draft_payload["audit_events"],
            *(event.to_dict() for event in prepared_events),
        ]
        ManualDraft.from_dict(candidate_draft_payload)

        original_passport = self.draft.passport
        original_audit_events = self.draft.audit_events
        original_updated_at = self.draft.updated_at
        original_undo_stack = list(self._undo_stack)
        original_redo_stack = list(self._redo_stack)
        try:
            self.draft.passport = replacement
            self.draft.audit_events = [
                *original_audit_events,
                *prepared_events,
            ]
            return self._finish(action="update_passport", before=before)
        except Exception:
            self.draft.passport = original_passport
            self.draft.audit_events = original_audit_events
            self.draft.updated_at = original_updated_at
            self._undo_stack = original_undo_stack
            self._redo_stack = original_redo_stack
            raise

    def copy_indicator_passport(
        self,
        source_channel: str,
        target_channels: Sequence[str],
        *,
        author: str | None = None,
        reason: str,
    ) -> bool:
        """Explicitly copy one stored channel passport into other channels.

        Every changed target receives an independent deep copy marked for a
        fresh engineering review and its own audit event.  A previously
        confirmed project is explicitly invalidated.  The complete candidate
        draft and all events are validated before the live draft is mutated.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ManualEntryServiceError(
                "Для копирования паспорта требуется непустое обоснование reason."
            )
        if source_channel not in MANUAL_INDICATOR_CHANNELS:
            raise ManualEntryServiceError(
                f"Неизвестный исходный канал индикатора {source_channel!r}."
            )
        source = self.draft.passport.indicator_passports.get(source_channel)
        if source is None:
            raise ManualEntryServiceError(
                f"Для исходного канала {source_channel!r} паспорт не задан."
            )
        if isinstance(target_channels, (str, bytes)):
            raise ManualEntryServiceError(
                "target_channels должен быть последовательностью имён каналов."
            )
        targets = list(target_channels)
        if not targets:
            raise ManualEntryServiceError("Укажите хотя бы один целевой канал.")
        if any(not isinstance(channel, str) for channel in targets):
            raise ManualEntryServiceError("Все target_channels должны быть строками.")
        unknown = sorted(set(targets) - set(MANUAL_INDICATOR_CHANNELS))
        if unknown:
            raise ManualEntryServiceError(
                f"Неизвестные целевые каналы индикаторов: {unknown!r}."
            )
        if len(set(targets)) != len(targets):
            raise ManualEntryServiceError("target_channels не должен содержать повторы.")
        if source_channel in targets:
            raise ManualEntryServiceError(
                "Исходный канал не может одновременно быть целевым."
            )

        before = self._snapshot()
        candidate_passport_payload = copy.deepcopy(before.passport)
        source_payload = copy.deepcopy(
            candidate_passport_payload["indicator_passports"][source_channel]
        )
        source_payload["assignment_status"] = "review_required"
        changed_targets: list[tuple[str, Any, dict[str, Any]]] = []
        for target in targets:
            previous = copy.deepcopy(
                candidate_passport_payload["indicator_passports"][target]
            )
            replacement_payload = copy.deepcopy(source_payload)
            if previous == replacement_payload:
                continue
            candidate_passport_payload["indicator_passports"][target] = (
                replacement_payload
            )
            changed_targets.append((target, previous, replacement_payload))

        # Validate the complete passport even for a semantic no-op.  A corrupt
        # in-memory source (for example NaN injected by an integration) must not
        # be accepted merely because a target contains the same corrupt value.
        candidate_passport = ManualPassport.from_dict(candidate_passport_payload)
        if not changed_targets:
            return False

        previous_metrology_status = candidate_passport.metrology_status
        invalidates_confirmation = previous_metrology_status == "confirmed"
        if invalidates_confirmation:
            candidate_passport_payload["metrology_status"] = "draft"
            candidate_passport = ManualPassport.from_dict(candidate_passport_payload)

        actor = self._author(author)
        normalized_reason = reason.strip()
        prepared_events = [
            ManualAuditEvent.create(
                author=actor,
                action="copy_indicator_passport",
                entity_id=f"{self.draft.draft_id}:passport:{target}",
                field=f"indicator_passports.{target}",
                old_value=old_value,
                new_value=new_value,
                reason=normalized_reason,
            )
            for target, old_value, new_value in changed_targets
        ]
        if invalidates_confirmation:
            prepared_events.append(
                ManualAuditEvent.create(
                    author=actor,
                    action="invalidate_metrology_confirmation",
                    entity_id=f"{self.draft.draft_id}:passport",
                    field="metrology_status",
                    old_value=previous_metrology_status,
                    new_value="draft",
                    reason=normalized_reason,
                )
            )

        # A full runtime round-trip validates existing rows/audit data together
        # with the staged passport and new events before any live-state change.
        candidate_draft_payload = self.draft.to_dict()
        candidate_draft_payload["passport"] = candidate_passport.to_dict()
        candidate_draft_payload["audit_events"] = [
            *candidate_draft_payload["audit_events"],
            *(event.to_dict() for event in prepared_events),
        ]
        ManualDraft.from_dict(candidate_draft_payload)

        original_passport = self.draft.passport
        original_audit_events = self.draft.audit_events
        original_updated_at = self.draft.updated_at
        original_undo_stack = list(self._undo_stack)
        original_redo_stack = list(self._redo_stack)
        try:
            self.draft.passport = candidate_passport
            self.draft.audit_events = [
                *original_audit_events,
                *prepared_events,
            ]
            return self._finish(action="copy_indicator_passport", before=before)
        except Exception:
            self.draft.passport = original_passport
            self.draft.audit_events = original_audit_events
            self.draft.updated_at = original_updated_at
            self._undo_stack = original_undo_stack
            self._redo_stack = original_redo_stack
            raise

    def _new_point(
        self, *, author: str, values: Mapping[str, Any] | None = None
    ) -> ManualPoint:
        if len(self.draft.rows) >= MAX_MANUAL_DRAFT_ROWS:
            raise ManualEntryServiceError("Достигнут предел числа строк черновика.")
        point = ManualPoint.create(len(self.draft.rows) + 1, author=author)
        for field, value in (values or {}).items():
            if field in _PROTECTED_EDITOR_FIELDS or field not in MANUAL_EDITOR_COLUMNS:
                raise ManualEntryServiceError(f"Поле {field!r} нельзя задать для новой строки.")
            if field == "sequence_no":
                continue
            setattr(point, field, _editor_value(field, value))
        return point

    def add_row(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        author: str | None = None,
        reason: str = "manual_edit",
    ) -> str:
        actor = self._author(author)
        before = self._snapshot()
        point = self._new_point(author=actor, values=values)
        self.draft.rows.append(point)
        self._event(
            author=actor,
            action="add_row",
            entity_id=point.manual_row_uuid,
            field=None,
            old_value=None,
            new_value=point.to_dict(),
            reason=reason,
        )
        self._finish(action="add_row", before=before)
        return point.manual_row_uuid

    def insert_row(
        self,
        target: int | str,
        *,
        position: str = "before",
        values: Mapping[str, Any] | None = None,
        author: str | None = None,
        reason: str = "manual_edit",
    ) -> str:
        if position not in {"before", "after"}:
            raise ManualEntryServiceError("position должен быть 'before' или 'after'.")
        actor = self._author(author)
        before = self._snapshot()
        target_index = self._row_index(target)
        insert_index = target_index + (position == "after")
        point = self._new_point(author=actor, values=values)
        self.draft.rows.insert(insert_index, point)
        self._event(
            author=actor,
            action=f"insert_{position}",
            entity_id=point.manual_row_uuid,
            field=None,
            old_value=None,
            new_value=point.to_dict(),
            reason=reason,
        )
        self._renumber_internal(author=actor, reason=reason)
        self._finish(action=f"insert_{position}", before=before)
        return point.manual_row_uuid

    def delete_row(
        self,
        target: int | str,
        *,
        confirmed: bool = False,
        author: str | None = None,
        reason: str = "manual_edit",
    ) -> ManualPoint:
        if not confirmed:
            raise ConfirmationRequiredError("Удаление строки требует подтверждения.")
        actor = self._author(author)
        before = self._snapshot()
        index = self._row_index(target)
        point = self.draft.rows.pop(index)
        self._event(
            author=actor,
            action="delete_row",
            entity_id=point.manual_row_uuid,
            field=None,
            old_value=point.to_dict(),
            new_value=None,
            reason=reason,
        )
        self._renumber_internal(author=actor, reason=reason)
        self._finish(action="delete_row", before=before)
        return point

    def duplicate_row(
        self,
        target: int | str,
        *,
        position: str = "after",
        author: str | None = None,
        reason: str = "manual_edit",
    ) -> str:
        if position not in {"before", "after"}:
            raise ManualEntryServiceError("position должен быть 'before' или 'after'.")
        actor = self._author(author)
        before = self._snapshot()
        index = self._row_index(target)
        source = self.draft.rows[index]
        values = {
            field: getattr(source, field)
            for field in MANUAL_EDITOR_COLUMNS
            if field != "sequence_no"
        }
        duplicate = self._new_point(author=actor, values=values)
        insert_index = index + (position == "after")
        self.draft.rows.insert(insert_index, duplicate)
        self._event(
            author=actor,
            action="duplicate_row",
            entity_id=duplicate.manual_row_uuid,
            field=None,
            old_value={"source_uuid": source.manual_row_uuid},
            new_value=duplicate.to_dict(),
            reason=reason,
        )
        self._renumber_internal(author=actor, reason=reason)
        self._finish(action="duplicate_row", before=before)
        return duplicate.manual_row_uuid

    def renumber(
        self,
        *,
        author: str | None = None,
        reason: str = "manual_edit",
    ) -> int:
        actor = self._author(author)
        before = self._snapshot()
        changed = self._renumber_internal(author=actor, reason=reason)
        self._finish(action="renumber", before=before)
        return changed

    def fill_stages(
        self,
        *,
        start: int = 1,
        step: int = 1,
        rows: Sequence[int | str] | None = None,
        author: str | None = None,
        reason: str = "manual_edit",
    ) -> int:
        if step == 0:
            raise ManualEntryServiceError("Шаг номера ступени не может быть нулём.")
        actor = self._author(author)
        indices = (
            list(range(len(self.draft.rows)))
            if rows is None
            else [self._row_index(row) for row in rows]
        )
        before = self._snapshot()
        changed = 0
        for offset, index in enumerate(indices):
            point = self.draft.rows[index]
            new_value = str(start + offset * step)
            if point.stage_no == new_value:
                continue
            old = point.stage_no
            point.stage_no = new_value
            self._mark_modified(point, actor)
            self._event(
                author=actor,
                action="fill_stages",
                entity_id=point.manual_row_uuid,
                field="stage_no",
                old_value=old,
                new_value=new_value,
                reason=reason,
            )
            changed += 1
        self._finish(action="fill_stages", before=before)
        return changed

    def paste_block(
        self,
        start_row: int | str,
        start_column: int | str,
        payload: str,
        *,
        expand_rows: bool = True,
        author: str | None = None,
        reason: str = "clipboard_paste",
    ) -> int:
        block = parse_rectangular_tsv(payload)
        row_index = self._row_index(start_row, allow_end=expand_rows)
        column_index = self._column_index(start_column)
        if column_index + len(block[0]) > len(MANUAL_EDITOR_COLUMNS):
            raise ClipboardShapeError("TSV-блок выходит за правую границу таблицы.")
        required = row_index + len(block)
        if required > MAX_MANUAL_DRAFT_ROWS:
            raise ClipboardShapeError("TSV-блок превышает предел числа строк.")
        if required > len(self.draft.rows) and not expand_rows:
            raise ClipboardShapeError("TSV-блок выходит за нижнюю границу таблицы.")

        # Validate and normalize the complete block before changing the draft.
        # This keeps a rejected paste atomic and prevents partial audit events.
        normalized_block: list[list[Any]] = []
        for values in block:
            normalized_row: list[Any] = []
            for column_offset, raw_value in enumerate(values):
                field = MANUAL_EDITOR_COLUMNS[column_index + column_offset]
                normalized_row.append(_editor_value(field, raw_value))
            normalized_block.append(normalized_row)

        actor = self._author(author)
        before = self._snapshot()
        while len(self.draft.rows) < required:
            point = self._new_point(author=actor)
            self.draft.rows.append(point)
            self._event(
                author=actor,
                action="paste_add_row",
                entity_id=point.manual_row_uuid,
                field=None,
                old_value=None,
                new_value=point.to_dict(),
                reason=reason,
            )

        changed = 0
        for row_offset, values in enumerate(normalized_block):
            point = self.draft.rows[row_index + row_offset]
            row_changed = False
            for column_offset, new_value in enumerate(values):
                field = MANUAL_EDITOR_COLUMNS[column_index + column_offset]
                if field == "sequence_no":
                    # sequence_no is always derived from current row order.
                    continue
                old_value = getattr(point, field)
                if old_value == new_value:
                    continue
                setattr(point, field, new_value)
                self._event(
                    author=actor,
                    action="paste_cell",
                    entity_id=point.manual_row_uuid,
                    field=field,
                    old_value=old_value,
                    new_value=new_value,
                    reason=reason,
                )
                changed += 1
                row_changed = True
            if row_changed:
                self._mark_modified(point, actor)
        self._renumber_internal(author=actor, reason=reason)
        self._finish(action="paste_block", before=before)
        return changed

    def copy_block(
        self,
        start_row: int | str,
        end_row: int | str,
        start_column: int | str,
        end_column: int | str,
    ) -> str:
        first_row = self._row_index(start_row)
        last_row = self._row_index(end_row)
        first_column = self._column_index(start_column)
        last_column = self._column_index(end_column)
        if last_row < first_row or last_column < first_column:
            raise ClipboardShapeError("Конец копируемого диапазона предшествует началу.")
        output = io.StringIO(newline="")
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for point in self.draft.rows[first_row : last_row + 1]:
            writer.writerow(
                _clipboard_text(getattr(point, field))
                for field in MANUAL_EDITOR_COLUMNS[first_column : last_column + 1]
            )
        return output.getvalue().rstrip("\n")

    def clear_cells(
        self,
        cells: Iterable[tuple[int | str, int | str]],
        *,
        author: str | None = None,
        reason: str = "clear_cells",
    ) -> int:
        actor = self._author(author)
        resolved: list[tuple[int, str]] = []
        seen: set[tuple[int, str]] = set()
        for row, column in cells:
            index = self._row_index(row)
            field = MANUAL_EDITOR_COLUMNS[self._column_index(column)]
            if field == "sequence_no":
                raise ManualEntryServiceError("sequence_no очищать нельзя.")
            key = (index, field)
            if key not in seen:
                seen.add(key)
                resolved.append(key)

        before = self._snapshot()
        changed = 0
        changed_rows: set[int] = set()
        for index, field in resolved:
            point = self.draft.rows[index]
            new_value = "" if field in _TEXT_FIELDS else None
            old_value = getattr(point, field)
            if old_value == new_value:
                continue
            setattr(point, field, new_value)
            self._event(
                author=actor,
                action="clear_cell",
                entity_id=point.manual_row_uuid,
                field=field,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
            )
            changed += 1
            changed_rows.add(index)
        for index in changed_rows:
            self._mark_modified(self.draft.rows[index], actor)
        self._finish(action="clear_cells", before=before)
        return changed

    def apply_editor_frame(
        self,
        frame: pd.DataFrame,
        *,
        author: str | None = None,
        confirm_deletions: bool = False,
        reason: str = "manual_edit",
    ) -> bool:
        """Apply editor diffs by stable UUID and retain unseen raw columns.

        Existing nonblank UUIDs must belong to the draft.  Blank UUIDs denote
        newly inserted GUI rows.  Removing an existing UUID requires explicit
        ``confirm_deletions=True``.
        """

        if not isinstance(frame, pd.DataFrame):
            raise EditorConflictError("Редактор должен передать pandas.DataFrame.")
        if EDITOR_UUID_COLUMN not in frame.columns:
            raise EditorConflictError("В таблице отсутствует manual_row_uuid.")
        unexpected = set(frame.columns) - set(EDITOR_COLUMNS)
        if unexpected:
            raise EditorConflictError(
                f"Неизвестные столбцы редактора: {sorted(unexpected)!r}."
            )
        if len(frame) > MAX_MANUAL_DRAFT_ROWS:
            raise EditorConflictError("Таблица превышает допустимое число строк.")

        existing = {point.manual_row_uuid: point for point in self.draft.rows}
        identities: list[str | None] = []
        seen: set[str] = set()
        for value in frame[EDITOR_UUID_COLUMN].tolist():
            identity = None if _is_missing(value) or value == "" else str(value)
            if identity is not None:
                if identity in seen:
                    raise EditorConflictError(
                        f"manual_row_uuid {identity!r} повторяется в таблице."
                    )
                if identity not in existing:
                    raise EditorConflictError(
                        f"Неизвестный manual_row_uuid {identity!r}."
                    )
                seen.add(identity)
            identities.append(identity)
        deleted = set(existing) - seen
        if deleted and not confirm_deletions:
            raise ConfirmationRequiredError(
                "Удаление строк из редактора требует подтверждения."
            )

        visible_fields = [
            column for column in frame.columns if column in MANUAL_EDITOR_COLUMNS
        ]
        normalized_rows: list[dict[str, Any]] = []
        for _, editor_row in frame.iterrows():
            normalized_rows.append(
                {
                    field: _editor_value(field, editor_row[field])
                    for field in visible_fields
                }
            )

        actor = self._author(author)
        before = self._snapshot()
        new_rows: list[ManualPoint] = []
        for position, _ in enumerate(frame.iterrows(), start=1):
            identity = identities[position - 1]
            if identity is None:
                # The final frame length was checked above.  Creating directly
                # also permits replacing rows when a draft is already at the
                # row limit, without transiently exceeding that limit.
                point = ManualPoint.create(position, author=actor)
                self._event(
                    author=actor,
                    action="editor_add_row",
                    entity_id=point.manual_row_uuid,
                    field=None,
                    old_value=None,
                    new_value=point.to_dict(),
                    reason=reason,
                )
            else:
                point = existing[identity]
            row_changed = False
            for field in visible_fields:
                if field == "sequence_no":
                    continue
                new_value = normalized_rows[position - 1][field]
                old_value = getattr(point, field)
                if old_value == new_value:
                    continue
                setattr(point, field, new_value)
                self._event(
                    author=actor,
                    action="editor_cell_edit",
                    entity_id=point.manual_row_uuid,
                    field=field,
                    old_value=old_value,
                    new_value=new_value,
                    reason=reason,
                )
                row_changed = True
            if row_changed:
                self._mark_modified(point, actor)
            new_rows.append(point)

        for identity in deleted:
            point = existing[identity]
            self._event(
                author=actor,
                action="editor_delete_row",
                entity_id=identity,
                field=None,
                old_value=point.to_dict(),
                new_value=None,
                reason=reason,
            )
        if [row.manual_row_uuid for row in new_rows] != [
            row.manual_row_uuid for row in self.draft.rows if row.manual_row_uuid not in deleted
        ]:
            self._event(
                author=actor,
                action="editor_reorder_rows",
                entity_id=self.draft.draft_id,
                field="row_order",
                old_value=[row.manual_row_uuid for row in self.draft.rows],
                new_value=[row.manual_row_uuid for row in new_rows],
                reason=reason,
            )
        self.draft.rows = new_rows
        self._renumber_internal(author=actor, reason=reason)
        return self._finish(action="apply_editor_frame", before=before)

    # Compatibility spelling for callers that use the task text terminology.
    apply_editor_diffs = apply_editor_frame

    def undo(
        self,
        *,
        author: str | None = None,
        reason: str = "undo",
    ) -> str:
        if not self._undo_stack:
            raise HistoryEmptyError("Нет изменения для отмены.")
        actor = self._author(author)
        command = self._undo_stack.pop()
        self._restore(command.before)
        if command.effects:
            for effect in reversed(command.effects):
                self._event(
                    author=actor,
                    action="undo",
                    entity_id=effect.entity_id,
                    field=effect.field,
                    old_value=effect.new_value,
                    new_value=effect.old_value,
                    reason=f"{reason}:{command.command_id}:{effect.action}",
                )
        else:
            self._event(
                author=actor,
                action="undo",
                entity_id=self.draft.draft_id,
                field=None,
                old_value={"command_id": command.command_id, "action": command.action},
                new_value=None,
                reason=reason,
            )
        self._redo_stack.append(command)
        return command.action

    def redo(
        self,
        *,
        author: str | None = None,
        reason: str = "redo",
    ) -> str:
        if not self._redo_stack:
            raise HistoryEmptyError("Нет изменения для повтора.")
        actor = self._author(author)
        command = self._redo_stack.pop()
        self._restore(command.after)
        if command.effects:
            for effect in command.effects:
                self._event(
                    author=actor,
                    action="redo",
                    entity_id=effect.entity_id,
                    field=effect.field,
                    old_value=effect.old_value,
                    new_value=effect.new_value,
                    reason=f"{reason}:{command.command_id}:{effect.action}",
                )
        else:
            self._event(
                author=actor,
                action="redo",
                entity_id=self.draft.draft_id,
                field=None,
                old_value=None,
                new_value={"command_id": command.command_id, "action": command.action},
                reason=reason,
            )
        self._undo_stack.append(command)
        return command.action


__all__ = [
    "ClipboardShapeError",
    "ConfirmationRequiredError",
    "EDITOR_COLUMNS",
    "EDITOR_UUID_COLUMN",
    "EditorConflictError",
    "HistoryEmptyError",
    "ManualEntryService",
    "ManualEntryServiceError",
    "editor_frame",
    "parse_rectangular_tsv",
]
