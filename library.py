"""
AI-OSINT — Библиотека результатов
====================================

Журнал симуляций: сохранение, просмотр, удаление и экспорт
результатов экспериментов.

Архитектура хранения
--------------------
Библиотека использует двухуровневую систему:

    1. st.session_state — оперативное хранилище текущей сессии.
       Все записи доступны мгновенно, без I/O.

    2. JSON-файл (опционально) — персистентное хранилище.
       Позволяет сохранить журнал между перезапусками Streamlit.
       На Streamlit Cloud файловая система эфемерна, поэтому
       JSON используется как fallback (через volume mount или
       внешнее хранилище в будущих версиях).

Каждая запись (SimulationRecord) содержит:
    - Метаданные: дата/время, сценарий, параметры
    - Результаты ABM: траектория, таймлайн активации
    - Результаты Монте-Карло: P(VIRAL), ДИ, распределения
    - Индикаторы: все 6 значений + Индекс угрозы
    - Рекомендации

Спецификация (раздел 5, вкладка 6):
    - Журнал симуляций по датам
    - Параметры + результаты каждого эксперимента
    - Удаление записей (чтобы не засорялась)
    - Выбор записей для генерации отчёта

Зависимости:
    json, datetime, hashlib, config (внутренний),
    indicators (внутренний)

Авторы:
    Абсаттаров Г.Р., Саятбек С., Кайролла А.Б.
    КазУМОиМЯ им. Абылай хана | 2026
"""

from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import (
    COLORS,
    SCENARIOS,
    MARKOV_STATE_ORDER,
    render_tooltip,
)
from indicators import (
    DetectionResult,
    detection_to_dict,
    detection_from_dict,
)


# ═══════════════════════════════════════════════════════════════════
# 1. ЗАПИСЬ СИМУЛЯЦИИ
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SimulationRecord:
    """
    Единица хранения — результат одного эксперимента.

    Attributes
    ----------
    record_id : str
        Уникальный идентификатор (SHA-256[:12] от timestamp + params).
    timestamp : str
        ISO-8601 дата/время создания записи.
    display_time : str
        Отформатированное время для UI (ДД.ММ.ГГГГ ЧЧ:ММ).

    # Параметры эксперимента
    scenario_key : str
        Ключ сценария (organic / amplified / coordinated / hybrid).
    scenario_label : str
        Русское название сценария.
    n_agents : int
        Размер популяции.
    n_steps : int
        Число шагов (суток).
    seed : int
        Seed для воспроизводимости.

    # Результаты единичного прогона (ABM + Markov)
    max_state : int
        Максимальное достигнутое состояние Маркова (0–4).
    max_state_label : str
        Русское название максимального состояния.
    final_state : int
        Финальное состояние Маркова.
    final_state_label : str
        Русское название финального состояния.
    final_active_ratio : float
        Доля активных агентов в конце симуляции.
    trajectory : List[int]
        Полная траектория Маркова (для визуализации).

    # Результаты Монте-Карло (если был запуск)
    mc_n_simulations : Optional[int]
        Число итераций МК.
    mc_viral_probability : Optional[float]
        P(VIRAL).
    mc_viral_ci_lower : Optional[float]
        Нижняя граница 95% ДИ.
    mc_viral_ci_upper : Optional[float]
        Верхняя граница 95% ДИ.
    mc_mean_steps_to_viral : Optional[float]
        Среднее шагов до вирусной фазы.
    mc_peak_distribution : Optional[List[float]]
        Распределение пиковых состояний (5 элементов).
    mc_final_distribution : Optional[List[float]]
        Распределение финальных состояний.

    # Индикаторы
    detection_data : Optional[Dict]
        Сериализованный DetectionResult (через detection_to_dict).
    threat_index : float
        Агрегированный Индекс угрозы (0–100).
    threat_label : str
        Словесный уровень угрозы.

    # Примечание пользователя
    note : str
        Произвольная заметка (по желанию).
    """

    record_id: str = ""
    timestamp: str = ""
    display_time: str = ""

    scenario_key: str = "organic"
    scenario_label: str = ""
    n_agents: int = 200
    n_steps: int = 60
    seed: int = 42

    max_state: int = 0
    max_state_label: str = ""
    final_state: int = 0
    final_state_label: str = ""
    final_active_ratio: float = 0.0
    trajectory: List[int] = field(default_factory=list)

    mc_n_simulations: Optional[int] = None
    mc_viral_probability: Optional[float] = None
    mc_viral_ci_lower: Optional[float] = None
    mc_viral_ci_upper: Optional[float] = None
    mc_mean_steps_to_viral: Optional[float] = None
    mc_peak_distribution: Optional[List[float]] = None
    mc_final_distribution: Optional[List[float]] = None

    detection_data: Optional[Dict] = None
    threat_index: float = 0.0
    threat_label: str = "—"

    note: str = ""


# ═══════════════════════════════════════════════════════════════════
# 2. ФАБРИКА ЗАПИСЕЙ
# ═══════════════════════════════════════════════════════════════════

def _state_label(state_idx: int) -> str:
    """Русское название состояния Маркова по индексу."""
    from config import MARKOV_STATES
    key = MARKOV_STATE_ORDER[state_idx] if state_idx < len(MARKOV_STATE_ORDER) else "latent"
    state_obj = MARKOV_STATES.get(key)
    if state_obj is None:
        return "—"
    # Поддержка и dataclass, и dict
    if hasattr(state_obj, "label_ru"):
        return state_obj.label_ru
    if isinstance(state_obj, dict):
        return state_obj.get("label_ru", state_obj.get("name_ru", "—"))
    return "—"


def _scenario_label(scenario_key: str) -> str:
    """Русское название сценария."""
    scn = SCENARIOS.get(scenario_key)
    if scn is None:
        return scenario_key
    if hasattr(scn, "label_ru"):
        return scn.label_ru
    if isinstance(scn, dict):
        return scn.get("label_ru", scn.get("name_ru", scenario_key))
    return scenario_key


def _generate_id(timestamp_str: str, params: Dict) -> str:
    """SHA-256[:12] от конкатенации timestamp и параметров."""
    raw = f"{timestamp_str}|{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def create_record(
    sim_result: Optional[Dict] = None,
    mc_result: Optional[Any] = None,
    detection: Optional[DetectionResult] = None,
    scenario_key: str = "organic",
    n_agents: int = 200,
    n_steps: int = 60,
    seed: int = 42,
    note: str = "",
) -> SimulationRecord:
    """
    Фабричная функция: создаёт SimulationRecord из результатов.

    Parameters
    ----------
    sim_result : dict, optional
        Результат run_full_simulation() из abm_engine.py.
    mc_result : MonteCarloResult, optional
        Результат run_monte_carlo().
    detection : DetectionResult, optional
        Результат IndicatorEngine.compute_all().
    scenario_key : str
        Ключ сценария.
    n_agents, n_steps, seed : int
        Параметры эксперимента.
    note : str
        Примечание.

    Returns
    -------
    SimulationRecord
    """
    now = datetime.now(timezone.utc)
    ts_iso = now.isoformat()
    ts_display = now.strftime("%d.%m.%Y %H:%M")

    params = {
        "scenario": scenario_key,
        "n_agents": n_agents,
        "n_steps": n_steps,
        "seed": seed,
    }
    record_id = _generate_id(ts_iso, params)

    # ── ABM + Markov ──
    max_state = 0
    final_state = 0
    final_active = 0.0
    trajectory: List[int] = []

    if sim_result:
        max_state = sim_result.get("max_state", 0)
        final_state = sim_result.get("final_state", 0)
        trajectory = sim_result.get("trajectory", [])

        abm = sim_result.get("abm")
        if abm is not None:
            final_active = abm.get_active_ratio()

    # ── Монте-Карло ──
    mc_n = None
    mc_pviral = None
    mc_ci_lo = None
    mc_ci_hi = None
    mc_steps = None
    mc_peak: Optional[List[float]] = None
    mc_final: Optional[List[float]] = None

    if mc_result is not None:
        mc_n = mc_result.n_simulations
        mc_pviral = mc_result.viral_probability
        mc_ci_lo = mc_result.viral_ci_lower
        mc_ci_hi = mc_result.viral_ci_upper
        mc_steps = mc_result.mean_steps_to_viral

        if hasattr(mc_result, "peak_distribution"):
            pd = mc_result.peak_distribution
            mc_peak = pd.tolist() if isinstance(pd, np.ndarray) else list(pd)

        if hasattr(mc_result, "final_state_distribution"):
            fd = mc_result.final_state_distribution
            mc_final = fd.tolist() if isinstance(fd, np.ndarray) else list(fd)

    # ── Индикаторы ──
    det_data = None
    threat_idx = 0.0
    threat_lbl = "—"

    if detection is not None:
        det_data = detection_to_dict(detection)
        threat_idx = detection.threat_index
        threat_lbl = detection.threat_label

    return SimulationRecord(
        record_id=record_id,
        timestamp=ts_iso,
        display_time=ts_display,
        scenario_key=scenario_key,
        scenario_label=_scenario_label(scenario_key),
        n_agents=n_agents,
        n_steps=n_steps,
        seed=seed,
        max_state=max_state,
        max_state_label=_state_label(max_state),
        final_state=final_state,
        final_state_label=_state_label(final_state),
        final_active_ratio=round(final_active, 4),
        trajectory=trajectory,
        mc_n_simulations=mc_n,
        mc_viral_probability=mc_pviral,
        mc_viral_ci_lower=mc_ci_lo,
        mc_viral_ci_upper=mc_ci_hi,
        mc_mean_steps_to_viral=mc_steps,
        mc_peak_distribution=mc_peak,
        mc_final_distribution=mc_final,
        detection_data=det_data,
        threat_index=round(threat_idx, 1),
        threat_label=threat_lbl,
        note=note,
    )


# ═══════════════════════════════════════════════════════════════════
# 3. СЕРИАЛИЗАЦИЯ ЗАПИСЕЙ
# ═══════════════════════════════════════════════════════════════════

def record_to_dict(record: SimulationRecord) -> Dict:
    """Преобразует SimulationRecord в JSON-совместимый словарь."""
    data = {
        "record_id": record.record_id,
        "timestamp": record.timestamp,
        "display_time": record.display_time,
        "scenario_key": record.scenario_key,
        "scenario_label": record.scenario_label,
        "n_agents": record.n_agents,
        "n_steps": record.n_steps,
        "seed": record.seed,
        "max_state": record.max_state,
        "max_state_label": record.max_state_label,
        "final_state": record.final_state,
        "final_state_label": record.final_state_label,
        "final_active_ratio": record.final_active_ratio,
        "trajectory": record.trajectory,
        "mc_n_simulations": record.mc_n_simulations,
        "mc_viral_probability": record.mc_viral_probability,
        "mc_viral_ci_lower": record.mc_viral_ci_lower,
        "mc_viral_ci_upper": record.mc_viral_ci_upper,
        "mc_mean_steps_to_viral": record.mc_mean_steps_to_viral,
        "mc_peak_distribution": record.mc_peak_distribution,
        "mc_final_distribution": record.mc_final_distribution,
        "detection_data": record.detection_data,
        "threat_index": record.threat_index,
        "threat_label": record.threat_label,
        "note": record.note,
    }
    return data


def record_from_dict(data: Dict) -> SimulationRecord:
    """Восстанавливает SimulationRecord из словаря."""
    return SimulationRecord(
        record_id=data.get("record_id", ""),
        timestamp=data.get("timestamp", ""),
        display_time=data.get("display_time", ""),
        scenario_key=data.get("scenario_key", "organic"),
        scenario_label=data.get("scenario_label", ""),
        n_agents=data.get("n_agents", 200),
        n_steps=data.get("n_steps", 60),
        seed=data.get("seed", 42),
        max_state=data.get("max_state", 0),
        max_state_label=data.get("max_state_label", ""),
        final_state=data.get("final_state", 0),
        final_state_label=data.get("final_state_label", ""),
        final_active_ratio=data.get("final_active_ratio", 0.0),
        trajectory=data.get("trajectory", []),
        mc_n_simulations=data.get("mc_n_simulations"),
        mc_viral_probability=data.get("mc_viral_probability"),
        mc_viral_ci_lower=data.get("mc_viral_ci_lower"),
        mc_viral_ci_upper=data.get("mc_viral_ci_upper"),
        mc_mean_steps_to_viral=data.get("mc_mean_steps_to_viral"),
        mc_peak_distribution=data.get("mc_peak_distribution"),
        mc_final_distribution=data.get("mc_final_distribution"),
        detection_data=data.get("detection_data"),
        threat_index=data.get("threat_index", 0.0),
        threat_label=data.get("threat_label", "—"),
        note=data.get("note", ""),
    )


# ═══════════════════════════════════════════════════════════════════
# 4. БИБЛИОТЕКА РЕЗУЛЬТАТОВ
# ═══════════════════════════════════════════════════════════════════

# Ключ в st.session_state
_SESSION_KEY = "aio_library_records"

# Путь к JSON-хранилищу (fallback)
_JSON_PATH = "library_data.json"

# Максимум записей (защита от переполнения)
MAX_RECORDS = 100


class ResultsLibrary:
    """
    Библиотека результатов симуляций.

    CRUD-операции над журналом экспериментов. Хранит данные
    в st.session_state (оперативно) с опциональной записью
    в JSON (персистентность).

    Пример использования::

        lib = ResultsLibrary()
        lib.add(record)
        records = lib.get_all()
        lib.delete(record_id)
        selected = lib.get_selected([id1, id2])
    """

    def __init__(self, use_json: bool = False) -> None:
        """
        Parameters
        ----------
        use_json : bool
            Если True — дублирует записи в JSON-файл.
            На Streamlit Cloud файловая система эфемерна,
            поэтому по умолчанию отключено.
        """
        self.use_json = use_json
        self._ensure_session()

        if use_json:
            self._load_from_json()

    # ──────── инициализация хранилища ────────

    @staticmethod
    def _ensure_session() -> None:
        """Гарантирует существование ключа в session_state."""
        try:
            import streamlit as st
            if _SESSION_KEY not in st.session_state:
                st.session_state[_SESSION_KEY] = []
        except ImportError:
            # Вне Streamlit — используем fallback
            pass

    @property
    def _records(self) -> List[Dict]:
        """Ссылка на хранилище записей."""
        try:
            import streamlit as st
            self._ensure_session()
            return st.session_state[_SESSION_KEY]
        except ImportError:
            # Fallback: атрибут экземпляра
            if not hasattr(self, "_fallback_records"):
                self._fallback_records: List[Dict] = []
            return self._fallback_records

    # ──────── CRUD ────────

    def add(self, record: SimulationRecord) -> str:
        """
        Добавляет запись в библиотеку.

        Parameters
        ----------
        record : SimulationRecord
            Запись симуляции.

        Returns
        -------
        str
            record_id добавленной записи.
        """
        data = record_to_dict(record)

        # Проверка дубликата
        existing_ids = {r.get("record_id") for r in self._records}
        if data["record_id"] in existing_ids:
            return data["record_id"]

        # Вставляем в начало (новые сверху)
        self._records.insert(0, data)

        # Ограничение размера
        while len(self._records) > MAX_RECORDS:
            self._records.pop()

        if self.use_json:
            self._save_to_json()

        return data["record_id"]

    def get_all(self) -> List[SimulationRecord]:
        """
        Возвращает все записи (новые первые).

        Returns
        -------
        list[SimulationRecord]
        """
        return [record_from_dict(r) for r in self._records]

    def get_by_id(self, record_id: str) -> Optional[SimulationRecord]:
        """
        Возвращает запись по ID.

        Parameters
        ----------
        record_id : str

        Returns
        -------
        SimulationRecord | None
        """
        for r in self._records:
            if r.get("record_id") == record_id:
                return record_from_dict(r)
        return None

    def get_selected(
        self,
        record_ids: List[str],
    ) -> List[SimulationRecord]:
        """
        Возвращает записи по списку ID (для генерации отчёта).

        Parameters
        ----------
        record_ids : list[str]
            Список record_id для выборки.

        Returns
        -------
        list[SimulationRecord]
        """
        id_set = set(record_ids)
        return [
            record_from_dict(r)
            for r in self._records
            if r.get("record_id") in id_set
        ]

    def delete(self, record_id: str) -> bool:
        """
        Удаляет запись по ID.

        Parameters
        ----------
        record_id : str

        Returns
        -------
        bool
            True если запись найдена и удалена.
        """
        initial_len = len(self._records)
        self._records[:] = [
            r for r in self._records
            if r.get("record_id") != record_id
        ]
        deleted = len(self._records) < initial_len

        if deleted and self.use_json:
            self._save_to_json()

        return deleted

    def clear_all(self) -> int:
        """
        Очищает всю библиотеку.

        Returns
        -------
        int
            Количество удалённых записей.
        """
        count = len(self._records)
        self._records.clear()

        if self.use_json:
            self._save_to_json()

        return count

    def update_note(self, record_id: str, note: str) -> bool:
        """
        Обновляет примечание к записи.

        Parameters
        ----------
        record_id : str
        note : str

        Returns
        -------
        bool
            True если запись найдена и обновлена.
        """
        for r in self._records:
            if r.get("record_id") == record_id:
                r["note"] = note
                if self.use_json:
                    self._save_to_json()
                return True
        return False

    @property
    def count(self) -> int:
        """Количество записей в библиотеке."""
        return len(self._records)

    @property
    def is_empty(self) -> bool:
        """Библиотека пуста?"""
        return len(self._records) == 0

    # ──────── JSON persistence ────────

    def _save_to_json(self) -> None:
        """Сохраняет текущее состояние в JSON-файл."""
        try:
            with open(_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(self._records, f, ensure_ascii=False, indent=2)
        except (OSError, IOError):
            pass  # Молча игнорируем ошибки записи (read-only FS)

    def _load_from_json(self) -> None:
        """Загружает записи из JSON-файла (при наличии)."""
        if not os.path.exists(_JSON_PATH):
            return
        try:
            with open(_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # Мерджим: JSON-данные + текущие session_state
                existing_ids = {r.get("record_id") for r in self._records}
                for item in data:
                    if item.get("record_id") not in existing_ids:
                        self._records.append(item)
        except (OSError, IOError, json.JSONDecodeError):
            pass

    # ──────── экспорт ────────

    def export_json(self, record_ids: Optional[List[str]] = None) -> str:
        """
        Экспорт записей в JSON-строку.

        Parameters
        ----------
        record_ids : list[str], optional
            Если указан — экспортируются только выбранные.
            Если None — все записи.

        Returns
        -------
        str
            JSON-строка (pretty-printed).
        """
        if record_ids:
            id_set = set(record_ids)
            data = [
                r for r in self._records
                if r.get("record_id") in id_set
            ]
        else:
            data = self._records

        return json.dumps(data, ensure_ascii=False, indent=2)

    def export_summary_table(
        self,
        record_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Экспорт сводной таблицы для отображения в Streamlit.

        Каждая строка — словарь с ключами, соответствующими
        колонкам таблицы (для st.dataframe / st.table).

        Parameters
        ----------
        record_ids : list[str], optional
            Фильтр по ID.

        Returns
        -------
        list[dict]
            Список строк таблицы.
        """
        records = (
            self.get_selected(record_ids)
            if record_ids
            else self.get_all()
        )

        table = []
        for rec in records:
            row = {
                "Дата": rec.display_time,
                "Сценарий": rec.scenario_label,
                "Агентов": rec.n_agents,
                "Шагов": rec.n_steps,
                "Макс. фаза": rec.max_state_label,
                "Индекс угрозы": rec.threat_index,
                "Уровень": rec.threat_label,
            }

            # Монте-Карло (если есть)
            if rec.mc_viral_probability is not None:
                row["P(вирусная)"] = f"{rec.mc_viral_probability:.1%}"
                row["МК итераций"] = rec.mc_n_simulations
            else:
                row["P(вирусная)"] = "—"
                row["МК итераций"] = "—"

            if rec.note:
                row["Примечание"] = rec.note

            table.append(row)

        return table


# ═══════════════════════════════════════════════════════════════════
# 5. UI-ПОМОЩНИКИ (для app.py)
# ═══════════════════════════════════════════════════════════════════

def render_library_card(record: SimulationRecord) -> str:
    """
    HTML-карточка записи для вкладки «Библиотека результатов».

    Отображает: дату, сценарий, ключевые метрики, Индекс угрозы.
    """
    # Цвет border по уровню угрозы
    if record.threat_index <= 30:
        border_color = COLORS.green_deep
    elif record.threat_index <= 60:
        border_color = COLORS.yellow_kerria
    else:
        border_color = COLORS.red_crimson

    # Монте-Карло строка
    mc_html = ""
    if record.mc_viral_probability is not None:
        mc_html = f"""
        <div style="font-size: 0.82rem; color: {COLORS.text_secondary};
             margin-top: 0.3rem;">
            Монте-Карло ({record.mc_n_simulations} итераций):
            P(вирусная) = {record.mc_viral_probability:.1%}
            [{record.mc_viral_ci_lower:.1%}–{record.mc_viral_ci_upper:.1%}]
        </div>
        """

    # Примечание
    note_html = ""
    if record.note:
        note_html = f"""
        <div style="font-size: 0.8rem; color: {COLORS.text_secondary};
             margin-top: 0.4rem; font-style: italic;">
            📝 {record.note}
        </div>
        """

    return f"""
    <div class="aio-card"
         style="border-left: 4px solid {border_color};">
        <div style="display: flex; justify-content: space-between;
             align-items: flex-start;">
            <div>
                <h3 style="margin-bottom: 0.2rem;">
                    {record.scenario_label}
                </h3>
                <div style="font-size: 0.78rem;
                     color: {COLORS.text_secondary};">
                    {record.display_time} &nbsp;|&nbsp;
                    {record.n_agents} агентов &nbsp;|&nbsp;
                    {record.n_steps} суток &nbsp;|&nbsp;
                    seed={record.seed}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.4rem; font-weight: 700;">
                    {record.threat_index:.1f}
                </div>
                <div style="font-size: 0.75rem;
                     color: {COLORS.text_secondary};">
                    {record.threat_label}
                </div>
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem;">
            Макс. фаза: <strong>{record.max_state_label}</strong>
            &nbsp;|&nbsp;
            Финал: <strong>{record.final_state_label}</strong>
            &nbsp;|&nbsp;
            Активных: <strong>{record.final_active_ratio:.1%}</strong>
        </div>
        {mc_html}
        {note_html}
    </div>
    """


def render_library_empty() -> str:
    """HTML-заглушка для пустой библиотеки."""
    return f"""
    <div class="aio-card" style="text-align: center;
         padding: 2.5rem 1.5rem;">
        <div style="font-size: 2rem; margin-bottom: 0.8rem;">📭</div>
        <div style="font-size: 1rem; font-weight: 600;
             color: {COLORS.text_primary}; margin-bottom: 0.4rem;">
            Библиотека пуста
        </div>
        <div style="font-size: 0.85rem;
             color: {COLORS.text_secondary};">
            Запустите симуляцию на вкладке «АОМ-симуляция» или
            «Монте-Карло», затем сохраните результаты в библиотеку.
        </div>
    </div>
    """


def render_library_stats(library: ResultsLibrary) -> str:
    """HTML-блок статистики библиотеки (количество записей)."""
    records = library.get_all()
    if not records:
        return ""

    # Распределение по сценариям
    scenarios: Dict[str, int] = {}
    for r in records:
        lbl = r.scenario_label or r.scenario_key
        scenarios[lbl] = scenarios.get(lbl, 0) + 1

    scenario_parts = [
        f"{lbl}: {cnt}" for lbl, cnt in scenarios.items()
    ]
    scenario_str = " &nbsp;|&nbsp; ".join(scenario_parts)

    # Средний индекс угрозы
    avg_threat = np.mean([r.threat_index for r in records])

    return f"""
    <div class="aio-card" style="padding: 0.8rem 1.2rem;">
        <div style="display: flex; justify-content: space-between;
             align-items: center;">
            <div>
                <span style="font-weight: 600;
                      color: {COLORS.blue_navy};">
                    Записей: {library.count}
                </span>
                <span style="font-size: 0.8rem;
                      color: {COLORS.text_secondary};
                      margin-left: 1rem;">
                    {scenario_str}
                </span>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.8rem;
                      color: {COLORS.text_secondary};">
                    Средний индекс угрозы:
                </span>
                <span style="font-weight: 600;">
                    {avg_threat:.1f}
                </span>
            </div>
        </div>
    </div>
    """
