"""
AI-OSINT — Индикаторы обнаружения информационных операций
==========================================================

Шесть индикаторов + агрегированный Индекс угрозы + экспертная
система рекомендаций.

Индикаторы разделены на два класса по источнику данных:

    GDELT-производные (вычисляются из данных data_engine.py):
        1. Индекс аномальности        — Z-оценка объёма публикаций
        2. Коэффициент синхронности    — концентрация во времени
        3. Текстовая гомогенность      — косинусная близость TF-IDF
        4. Тональный сдвиг             — |Δsentiment| за 24 ч

    ABM-производные (вычисляются из результатов abm_engine.py):
        5. Скорость распространения    — шаги до порога 50%
        6. Бот-активность              — композитный индикатор

Когда GDELT-данные недоступны (демо-режим, офлайн), модуль
вычисляет прокси-оценки на основе состояния агентной сети,
обеспечивая работоспособность MVP при любых условиях.

Агрегированный Индекс угрозы (0–100):
    T = Σ(wᵢ × norm(Iᵢ)) × 100

    где wᵢ — вес индикатора, norm(Iᵢ) — нормализованное
    значение в [0, 1] через пороги из config.INDICATORS.

Экспертная система рекомендаций — набор продукционных правил
(if-then) по комбинациям значений индикаторов. Не ML, а
rule-based engine (Buchanan, Shortliffe, 1984).

Теоретические основания:
    - Nimmo B. (2019) — «The Breakout Scale» (DFRLab)
    - Gleicher N. (2020) — Meta CIB taxonomy
    - Varol O. et al. (2017) — Botometer / BotOrNot
    - Broder A. (1997) — near-duplicate detection
    - GDELT Technical Documentation — Average Tone metric

Зависимости:
    numpy, config (внутренний модуль), abm_engine (внутренний)

Авторы:
    Абсаттаров Г.Р., Саятбек С., Кайролла А.Б.
    КазУМОиМЯ им. Абылай хана | 2026
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import (
    INDICATORS,
    THREAT_INDEX_THRESHOLDS,
    COLORS,
    traffic_light,
    traffic_emoji,
    threat_index_level,
    render_tooltip,
    MARKOV_STATE_ORDER,
)


# ═══════════════════════════════════════════════════════════════════
# 1. РЕЗУЛЬТАТ ОДНОГО ИНДИКАТОРА
# ═══════════════════════════════════════════════════════════════════

@dataclass
class IndicatorResult:
    """
    Результат вычисления одного индикатора.

    Содержит всё необходимое для отображения в UI:
    числовое значение, светофорный статус, CSS-класс, эмодзи,
    а также метаданные из config.IndicatorConfig для всплывающих
    пояснений (формула, пороги, методология, пример).

    Attributes
    ----------
    key : str
        Системный ключ индикатора (напр. «anomaly_index»).
    value : float
        Вычисленное значение.
    label_ru : str
        Название на русском для UI.
    formula : str
        Формула (из IndicatorConfig).
    emoji : str
        Светофорный эмодзи (🟢 / 🟡 / 🔴).
    css_class : str
        CSS-класс бейджа (badge-green / badge-yellow / badge-red).
    status_ru : str
        Словесный статус: «Норма» / «Повышенное внимание» / «Угроза».
    normalized : float
        Значение, нормализованное в [0, 1] для расчёта Индекса угрозы.
    description : str
        Описание индикатора.
    methodology : str
        Обоснование порогов.
    example_url : str
        Ссылка на реальный пример.
    example_text : str
        Описание примера.
    source : str
        Источник данных: «gdelt» или «abm».
    is_proxy : bool
        True, если значение вычислено через ABM-прокси (нет GDELT).
    """

    key: str
    value: float
    label_ru: str
    formula: str
    emoji: str
    css_class: str
    status_ru: str
    normalized: float
    description: str
    methodology: str
    example_url: str
    example_text: str
    source: str = "gdelt"
    is_proxy: bool = False


# ═══════════════════════════════════════════════════════════════════
# 2. РЕЗУЛЬТАТ ВСЕХ ИНДИКАТОРОВ
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """
    Агрегированный результат вычисления всех 6 индикаторов.

    Attributes
    ----------
    indicators : Dict[str, IndicatorResult]
        Все 6 индикаторов по ключам.
    threat_index : float
        Агрегированный Индекс угрозы (0–100).
    threat_emoji : str
        Светофорный эмодзи для Индекса угрозы.
    threat_css : str
        CSS-класс бейджа.
    threat_label : str
        Словесный уровень угрозы.
    recommendations : List[str]
        Список рекомендаций от экспертной системы.
    has_proxy : bool
        True, если хотя бы один индикатор вычислен через прокси.
    """

    indicators: Dict[str, IndicatorResult]
    threat_index: float
    threat_emoji: str
    threat_css: str
    threat_label: str
    recommendations: List[str] = field(default_factory=list)
    has_proxy: bool = False


# ═══════════════════════════════════════════════════════════════════
# 3. ВЕСА ИНДИКАТОРОВ ДЛЯ АГРЕГАЦИИ
# ═══════════════════════════════════════════════════════════════════

# Веса отражают диагностическую значимость каждого индикатора
# для обнаружения координированных информационных операций.
#
# Обоснование:
#   - Синхронность и текстовая гомогенность — наиболее надёжные
#     маркеры координации (Nimmo, 2019; Gleicher, 2020)
#   - Бот-активность — прямой маркер, но подвержен false positives
#   - Аномальность и тональный сдвиг — косвенные, требуют контекста
#   - Скорость распространения — ABM-прокси, менее надёжен
#
# Σ весов = 1.0

INDICATOR_WEIGHTS: Dict[str, float] = {
    "anomaly_index":    0.15,
    "synchrony_coeff":  0.22,
    "text_homogeneity": 0.22,
    "tonal_shift":      0.12,
    "spread_speed":     0.10,
    "bot_activity":     0.19,
}


# ═══════════════════════════════════════════════════════════════════
# 4. ВЫЧИСЛЕНИЕ ОТДЕЛЬНЫХ ИНДИКАТОРОВ
# ═══════════════════════════════════════════════════════════════════

def compute_anomaly_index(
    volumes_30d: np.ndarray,
    current_volume: float,
) -> float:
    """
    Индикатор 1: Индекс аномальности.

    Z-оценка текущего объёма публикаций относительно 30-дневного
    скользящего среднего и стандартного отклонения.

        Z = (x − μ₃₀) / σ₃₀

    Статистическая интерпретация:
        Z < 1.5  — в пределах нормальной флуктуации
        Z ∈ [1.5, 2.5) — умеренное отклонение (p < 0.05)
        Z ∈ [2.5, 3.5) — значимое отклонение (p < 0.01)
        Z ≥ 3.5 — экстремальное (p < 0.001)

    Parameters
    ----------
    volumes_30d : numpy.ndarray
        Массив суточных объёмов публикаций за 30 дней.
    current_volume : float
        Объём публикаций за текущие сутки.

    Returns
    -------
    float
        Z-оценка (неотрицательная).
    """
    if len(volumes_30d) < 2:
        return 0.0

    mu = np.mean(volumes_30d)
    sigma = np.std(volumes_30d, ddof=1)

    if sigma < 1e-9:
        return 0.0 if abs(current_volume - mu) < 1e-9 else 5.0

    z = (current_volume - mu) / sigma
    return max(0.0, z)


def compute_synchrony_coeff(
    timestamps_hours: np.ndarray,
    window_hours: float = 2.0,
) -> float:
    """
    Индикатор 2: Коэффициент синхронности.

    Доля публикаций, вышедших в наиболее плотном временном окне
    заданной ширины, от общего числа публикаций за сутки.

        S = max_posts_in_window / total_posts

    Метод: скользящее окно по 24-часовой шкале.

    Обоснование порогов (Stanford IO, Nimmo 2019):
        S < 0.15 — органическое распределение (энтропия ≈ max)
        S ∈ [0.15, 0.35) — возможно PR / таргетированный постинг
        S ∈ [0.35, 0.55) — признаки координации
        S ≥ 0.55 — высокая синхронность (бот-ферма, CIB)

    Parameters
    ----------
    timestamps_hours : numpy.ndarray
        Массив временных меток публикаций в часах (0.0–24.0).
    window_hours : float
        Ширина окна в часах (по умолчанию 2).

    Returns
    -------
    float
        Коэффициент синхронности [0, 1].
    """
    n = len(timestamps_hours)
    if n < 2:
        return 0.0

    sorted_ts = np.sort(timestamps_hours)
    max_count = 0

    for i in range(n):
        window_end = sorted_ts[i] + window_hours
        count = np.searchsorted(sorted_ts, window_end, side="right") - i
        if count > max_count:
            max_count = count

    return max_count / n


def compute_text_homogeneity(
    tfidf_similarity_matrix: Optional[np.ndarray] = None,
    mean_similarity: Optional[float] = None,
) -> float:
    """
    Индикатор 3: Текстовая гомогенность.

    Средняя косинусная близость TF-IDF-векторов публикаций
    одного кластера.

        H = mean(cosine_similarity(TF-IDF_vectors))

    Принимает либо матрицу попарных сходств, либо предвычисленное
    среднее значение (data_engine.py может передать любой формат).

    Обоснование порогов (Broder, 1997; IRA Troll Dataset):
        H < 0.4  — разнородный контент (органика)
        H ∈ [0.4, 0.7) — умеренная однородность (PR / шаблоны)
        H ∈ [0.7, 0.85) — высокая однородность (явные шаблоны)
        H ≥ 0.85 — near-duplicate (ботнет, копипаст)

    Parameters
    ----------
    tfidf_similarity_matrix : numpy.ndarray, optional
        Квадратная матрица попарных косинусных сходств.
    mean_similarity : float, optional
        Предвычисленное среднее значение (если матрица уже обработана).

    Returns
    -------
    float
        Средняя косинусная близость [0, 1].
    """
    if mean_similarity is not None:
        return np.clip(mean_similarity, 0.0, 1.0)

    if tfidf_similarity_matrix is None or tfidf_similarity_matrix.size < 4:
        return 0.0

    n = tfidf_similarity_matrix.shape[0]
    # Извлекаем верхний треугольник (без диагонали)
    upper_indices = np.triu_indices(n, k=1)
    pairwise = tfidf_similarity_matrix[upper_indices]

    if len(pairwise) == 0:
        return 0.0

    return float(np.clip(np.mean(pairwise), 0.0, 1.0))


def compute_tonal_shift(
    tone_today: float,
    tone_yesterday: float,
) -> float:
    """
    Индикатор 4: Тональный сдвиг.

    Абсолютное изменение средней тональности публикаций за 24 часа.

        |Δsentiment| = |tone_today − tone_yesterday|

    Шкала GDELT Average Tone: −10 … +10.

    Обоснование порогов (GDELT Technical Documentation):
        |Δ| < 0.2  — нормальная суточная волатильность
        |Δ| ∈ [0.2, 0.4) — повышенная, но возможна при реальном событии
        |Δ| ∈ [0.4, 0.7) — значительный сдвиг, требует проверки
        |Δ| ≥ 0.7  — экстремальный, вероятна координированная атака

    Parameters
    ----------
    tone_today : float
        Средняя тональность за текущие сутки (−10 … +10).
    tone_yesterday : float
        Средняя тональность за предыдущие сутки.

    Returns
    -------
    float
        Абсолютное значение тонального сдвига.
    """
    return abs(tone_today - tone_yesterday)


def compute_spread_speed(
    history: List[Dict],
    threshold: float = 0.50,
) -> float:
    """
    Индикатор 5: Скорость распространения.

    Количество шагов (суток) АОМ-симуляции до достижения порога
    активации 50% агентов. Вычисляется из истории ABM.

        V = min{t : active_ratio(t) ≥ 0.50}

    Если порог не достигнут — возвращается общее число шагов + 1
    (= «не достиг»).

    Обоснование порогов (Axelrod, 1997; Epstein & Axtell, 1996):
        V > 20  — органический темп (нарратив набирает аудиторию
                  естественным путём, недели-месяцы)
        V ∈ (10, 20] — умеренный темп (PR-усиление)
        V ∈ (5, 10]  — быстрый (координированная кампания)
        V ≤ 5  — взрывной (CIB, бот-атака)

    NB: Обратная логика порогов — меньше = хуже.

    Parameters
    ----------
    history : list[dict]
        История ABM — список снимков от AgentBasedModel.step().
        Каждый снимок содержит ключ «active_ratio».
    threshold : float
        Порог активации (доля активных агентов, по умолчанию 0.50).

    Returns
    -------
    float
        Число шагов до порога (или len(history) + 1).
    """
    if not history:
        return 61.0  # Не достигнуто, максимум

    for snapshot in history:
        ratio = snapshot.get("active_ratio", 0.0)
        if ratio >= threshold:
            return float(snapshot.get("tick", len(history)))

    return float(len(history) + 1)


def compute_bot_activity(
    agents: Optional[List[Any]] = None,
    cv_timing: Optional[float] = None,
    ttr: Optional[float] = None,
    repost_ratio: Optional[float] = None,
) -> float:
    """
    Индикатор 6: Бот-активность.

    Композитный индикатор на основе трёх маркеров ботоподобного
    поведения:
        - CV(timing)    — коэффициент вариации интервалов между
                          постами (боты: CV < 0.1, люди: CV > 0.3)
        - TTR           — type-token ratio (лексическое разнообразие;
                          боты: TTR < 0.3, люди: TTR > 0.5)
        - repost_ratio  — доля репостов (боты: > 90%)

    Формула:
        B = 0.35 × f(CV) + 0.35 × f(TTR) + 0.30 × f(repost)

        где f(x) — нормализация к [0, 1]:
            f(CV)       = max(0, 1 − CV / 0.3)
            f(TTR)      = max(0, 1 − TTR / 0.6)
            f(repost)   = max(0, (repost − 0.5) / 0.5)

    Источники:
        Varol O. et al. (2017) — Botometer / BotOrNot
        DFRLab / Atlantic Council — методология анализа бот-ферм

    Parameters
    ----------
    agents : list[Agent], optional
        Агенты ABM (для вычисления прокси через bot_probability
        и repost_rate при отсутствии GDELT-данных).
    cv_timing : float, optional
        Коэффициент вариации интервалов (из GDELT / data_engine).
    ttr : float, optional
        Type-Token Ratio (из NLP-анализа текстов).
    repost_ratio : float, optional
        Доля репостов (из данных).

    Returns
    -------
    float
        Композитный индикатор бот-активности [0, 1].
    """
    # Если есть прямые метрики — используем их
    if all(v is not None for v in (cv_timing, ttr, repost_ratio)):
        f_cv = max(0.0, 1.0 - cv_timing / 0.3)
        f_ttr = max(0.0, 1.0 - ttr / 0.6)
        f_repost = max(0.0, (repost_ratio - 0.5) / 0.5)
        return 0.35 * f_cv + 0.35 * f_ttr + 0.30 * f_repost

    # ABM-прокси: агрегируем bot_probability и repost_rate усилителей
    if agents is not None:
        amplifiers = [
            a for a in agents
            if a.agent_type == "amplifier" and a.active
        ]
        if not amplifiers:
            return 0.0

        mean_bot_prob = np.mean([a.bot_probability for a in amplifiers])
        mean_repost = np.mean([a.repost_rate for a in amplifiers])

        # Нормализация: bot_probability уже в [0,1],
        # repost_rate тоже в [0,1]
        f_bot = mean_bot_prob
        f_repost = max(0.0, (mean_repost - 0.5) / 0.5)

        # CV-прокси: высокая bot_probability → низкий CV → высокий f_cv
        f_cv_proxy = mean_bot_prob * 0.8

        return 0.35 * f_cv_proxy + 0.35 * f_bot + 0.30 * f_repost

    return 0.0


# ═══════════════════════════════════════════════════════════════════
# 5. ABM-ПРОКСИ ДЛЯ GDELT-ИНДИКАТОРОВ
# ═══════════════════════════════════════════════════════════════════

def _proxy_anomaly_from_abm(history: List[Dict]) -> float:
    """
    Прокси Индекса аномальности из истории ABM.

    Логика: резкий рост числа активных агентов = аналог аномального
    всплеска публикаций. Вычисляем Z-оценку последнего шага
    относительно первых 70% истории (условный «30-дневный фон»).
    """
    if len(history) < 5:
        return 0.0

    ratios = [s["active_ratio"] for s in history]
    baseline_end = max(3, int(len(ratios) * 0.7))
    baseline = np.array(ratios[:baseline_end])
    current = ratios[-1]

    mu = np.mean(baseline)
    sigma = np.std(baseline, ddof=1)

    if sigma < 1e-9:
        return 0.0 if abs(current - mu) < 1e-9 else 4.0

    return max(0.0, (current - mu) / sigma)


def _proxy_synchrony_from_abm(history: List[Dict]) -> float:
    """
    Прокси Коэффициента синхронности из истории ABM.

    Логика: если большая часть новых активаций приходится на узкий
    интервал шагов (< 20% от общей длины), это аналог синхронного
    постинга.
    """
    if len(history) < 3:
        return 0.0

    newly = [s.get("newly_activated", 0) for s in history]
    total_new = sum(newly)
    if total_new == 0:
        return 0.0

    # Ширина окна = 20% от длины истории
    window = max(1, int(len(newly) * 0.2))
    max_in_window = 0
    for i in range(len(newly) - window + 1):
        s = sum(newly[i:i + window])
        if s > max_in_window:
            max_in_window = s

    return min(1.0, max_in_window / total_new)


def _proxy_homogeneity_from_abm(
    agents: List[Any],
    scenario_amp: float,
) -> float:
    """
    Прокси Текстовой гомогенности из ABM.

    Логика: доля активных усилителей × коэффициент усиления сценария
    коррелирует с однородностью контента (больше ботов → больше
    копипаста).
    """
    if not agents:
        return 0.0

    active_amps = sum(
        1 for a in agents
        if a.agent_type == "amplifier" and a.active
    )
    total_amps = sum(1 for a in agents if a.agent_type == "amplifier")

    if total_amps == 0:
        return 0.0

    amp_ratio = active_amps / total_amps

    # Масштабирование: scenario_amp=0.6 + amp_ratio=1.0 → H≈0.9
    return min(1.0, amp_ratio * (0.5 + scenario_amp * 0.8))


def _proxy_tonal_shift_from_markov(trajectory: List[int]) -> float:
    """
    Прокси Тонального сдвига из траектории Маркова.

    Логика: переход между состояниями = изменение «тональности»
    медийного пространства. Каждое состояние имеет условную
    тональность.
    """
    # Условная шкала тональности по состояниям:
    # LATENT=0.0, EMERGING=-0.3, GROWING=-0.6, VIRAL=-1.0, DECLINING=-0.2
    tone_map = np.array([0.0, -0.3, -0.6, -1.0, -0.2])

    if len(trajectory) < 2:
        return 0.0

    # Сдвиг за последний шаг, масштабированный к шкале GDELT (×3)
    current_tone = tone_map[trajectory[-1]]
    prev_tone = tone_map[trajectory[-2]]

    return abs(current_tone - prev_tone) * 3.0


# ═══════════════════════════════════════════════════════════════════
# 6. НОРМАЛИЗАЦИЯ ЗНАЧЕНИЙ
# ═══════════════════════════════════════════════════════════════════

def _normalize_indicator(
    value: float,
    thresholds: Tuple[float, float, float],
    inverse: bool = False,
) -> float:
    """
    Нормализация значения индикатора в [0, 1] для агрегации.

    Линейная интерполяция по трём порогам:
        [0, green) → [0.0, 0.33)
        [green, yellow) → [0.33, 0.66)
        [yellow, red+) → [0.66, 1.0]

    Parameters
    ----------
    value : float
        Сырое значение индикатора.
    thresholds : tuple
        (зелёный_порог, жёлтый_порог, красный_порог).
    inverse : bool
        True для обратной логики (spread_speed: больше = лучше).

    Returns
    -------
    float
        Нормализованное значение [0.0, 1.0].
    """
    g, y, r = thresholds

    if inverse:
        # Обратная: value > g → зелёный, value ≤ r → красный
        if value >= g:
            return 0.0
        elif value >= y:
            return 0.33 * (1.0 - (value - y) / max(g - y, 1e-9))
        elif value > r:
            return 0.33 + 0.33 * (1.0 - (value - r) / max(y - r, 1e-9))
        else:
            return min(1.0, 0.66 + 0.34 * (1.0 - value / max(r, 1e-9)))
    else:
        # Прямая: value < g → зелёный, value ≥ r → красный
        if value < g:
            return 0.33 * (value / max(g, 1e-9))
        elif value < y:
            return 0.33 + 0.33 * ((value - g) / max(y - g, 1e-9))
        elif value < r:
            return 0.66 + 0.34 * ((value - y) / max(r - y, 1e-9))
        else:
            return 1.0


def _status_label(css_class: str) -> str:
    """Словесный статус по CSS-классу."""
    return {
        "badge-green": "Норма",
        "badge-yellow": "Повышенное внимание",
        "badge-red": "Угроза",
    }.get(css_class, "—")


# ═══════════════════════════════════════════════════════════════════
# 7. ГЛАВНЫЙ ДВИЖОК ИНДИКАТОРОВ
# ═══════════════════════════════════════════════════════════════════

class IndicatorEngine:
    """
    Движок вычисления 6 индикаторов обнаружения.

    Принимает данные из двух источников:
        1. GDELT (data_engine.py) — словарь gdelt_data
        2. ABM (abm_engine.py)    — результат run_full_simulation()

    Если GDELT-данные отсутствуют (None или пустой словарь),
    автоматически переключается на ABM-прокси для индикаторов 1–4.

    Пример использования::

        engine = IndicatorEngine()
        result = engine.compute_all(
            gdelt_data=gdelt_data,    # dict от data_engine, или None
            sim_result=sim_result,    # dict от run_full_simulation()
        )

        # result.threat_index → 0–100
        # result.indicators["anomaly_index"].value → Z-score
        # result.recommendations → ["Рекомендация 1", ...]
    """

    def compute_all(
        self,
        gdelt_data: Optional[Dict] = None,
        sim_result: Optional[Dict] = None,
        mc_result: Optional[Any] = None,
    ) -> DetectionResult:
        """
        Вычисляет все 6 индикаторов + Индекс угрозы + рекомендации.

        Parameters
        ----------
        gdelt_data : dict, optional
            Данные от data_engine.py. Ожидаемая структура:
                - «volumes_30d»    : np.ndarray — суточные объёмы за 30 дней
                - «current_volume» : float — объём за текущие сутки
                - «timestamps_hours» : np.ndarray — метки времени (0–24 ч)
                - «tfidf_similarity» : np.ndarray — матрица сходств, или
                  «mean_tfidf_similarity» : float — предвычисленное среднее
                - «tone_today»     : float — средняя тональность сегодня
                - «tone_yesterday» : float — средняя тональность вчера
                - «cv_timing»      : float — CV интервалов (для бот-индикатора)
                - «ttr»            : float — Type-Token Ratio
                - «repost_ratio»   : float — доля репостов

        sim_result : dict, optional
            Результат run_full_simulation() из abm_engine.py:
                - «abm»     : AgentBasedModel
                - «markov»  : MarkovNarrative
                - «history» : list[dict]

        mc_result : MonteCarloResult, optional
            Результат run_monte_carlo() (для расширенной аналитики).

        Returns
        -------
        DetectionResult
            Полный набор: 6 индикаторов + индекс + рекомендации.
        """
        gdelt = gdelt_data or {}
        has_gdelt = bool(gdelt)

        # Извлечение ABM-объектов
        abm = sim_result.get("abm") if sim_result else None
        markov = sim_result.get("markov") if sim_result else None
        history = sim_result.get("history", []) if sim_result else []
        agents = abm.agents if abm else []
        trajectory = markov.trajectory if markov else []

        # Коэффициент усиления сценария (для прокси)
        scenario_amp = 0.05
        if abm and hasattr(abm, "scenario"):
            scn = abm.scenario
            # Поддержка как dict, так и dataclass
            if isinstance(scn, dict):
                scenario_amp = scn.get("amp_ratio", scn.get("amplification", 0.05))
            elif hasattr(scn, "amplification"):
                scenario_amp = scn.amplification
            elif hasattr(scn, "amp_ratio"):
                scenario_amp = scn.amp_ratio

        indicators: Dict[str, IndicatorResult] = {}
        has_proxy = False

        # ── Индикатор 1: Индекс аномальности ──
        if "volumes_30d" in gdelt and "current_volume" in gdelt:
            val = compute_anomaly_index(
                np.asarray(gdelt["volumes_30d"]),
                gdelt["current_volume"],
            )
            proxy = False
        elif history:
            val = _proxy_anomaly_from_abm(history)
            proxy = True
            has_proxy = True
        else:
            val = 0.0
            proxy = True
            has_proxy = True

        indicators["anomaly_index"] = self._build_result(
            "anomaly_index", val, proxy=proxy, source="gdelt",
        )

        # ── Индикатор 2: Коэффициент синхронности ──
        if "timestamps_hours" in gdelt:
            val = compute_synchrony_coeff(
                np.asarray(gdelt["timestamps_hours"]),
            )
            proxy = False
        elif history:
            val = _proxy_synchrony_from_abm(history)
            proxy = True
            has_proxy = True
        else:
            val = 0.0
            proxy = True
            has_proxy = True

        indicators["synchrony_coeff"] = self._build_result(
            "synchrony_coeff", val, proxy=proxy, source="gdelt",
        )

        # ── Индикатор 3: Текстовая гомогенность ──
        if "mean_tfidf_similarity" in gdelt:
            val = compute_text_homogeneity(
                mean_similarity=gdelt["mean_tfidf_similarity"],
            )
            proxy = False
        elif "tfidf_similarity" in gdelt:
            val = compute_text_homogeneity(
                tfidf_similarity_matrix=np.asarray(gdelt["tfidf_similarity"]),
            )
            proxy = False
        elif agents:
            val = _proxy_homogeneity_from_abm(agents, scenario_amp)
            proxy = True
            has_proxy = True
        else:
            val = 0.0
            proxy = True
            has_proxy = True

        indicators["text_homogeneity"] = self._build_result(
            "text_homogeneity", val, proxy=proxy, source="gdelt",
        )

        # ── Индикатор 4: Тональный сдвиг ──
        if "tone_today" in gdelt and "tone_yesterday" in gdelt:
            val = compute_tonal_shift(
                gdelt["tone_today"], gdelt["tone_yesterday"],
            )
            proxy = False
        elif trajectory:
            val = _proxy_tonal_shift_from_markov(trajectory)
            proxy = True
            has_proxy = True
        else:
            val = 0.0
            proxy = True
            has_proxy = True

        indicators["tonal_shift"] = self._build_result(
            "tonal_shift", val, proxy=proxy, source="gdelt",
        )

        # ── Индикатор 5: Скорость распространения ──
        val = compute_spread_speed(history)
        indicators["spread_speed"] = self._build_result(
            "spread_speed", val, proxy=False, source="abm",
        )

        # ── Индикатор 6: Бот-активность ──
        cv_t = gdelt.get("cv_timing")
        ttr_v = gdelt.get("ttr")
        rr = gdelt.get("repost_ratio")

        if all(v is not None for v in (cv_t, ttr_v, rr)):
            val = compute_bot_activity(
                cv_timing=cv_t, ttr=ttr_v, repost_ratio=rr,
            )
            proxy = False
        elif agents:
            val = compute_bot_activity(agents=agents)
            proxy = True
            has_proxy = True
        else:
            val = 0.0
            proxy = True
            has_proxy = True

        indicators["bot_activity"] = self._build_result(
            "bot_activity", val, proxy=proxy, source="abm",
        )

        # ── Агрегированный Индекс угрозы ──
        threat_index = self._compute_threat_index(indicators)
        t_emoji, t_css, t_label = threat_index_level(threat_index)

        # ── Рекомендации ──
        recommendations = self._generate_recommendations(
            indicators, threat_index,
        )

        return DetectionResult(
            indicators=indicators,
            threat_index=threat_index,
            threat_emoji=t_emoji,
            threat_css=t_css,
            threat_label=t_label,
            recommendations=recommendations,
            has_proxy=has_proxy,
        )

    # ──────── построение результата индикатора ────────

    @staticmethod
    def _build_result(
        key: str,
        value: float,
        proxy: bool = False,
        source: str = "gdelt",
    ) -> IndicatorResult:
        """Формирует IndicatorResult по ключу и значению."""
        cfg = INDICATORS[key]

        # Определение inverse (для spread_speed)
        inverse = (key == "spread_speed")

        emoji = traffic_emoji(value, cfg.thresholds, inverse=inverse)
        css = traffic_light(value, cfg.thresholds, inverse=inverse)
        status = _status_label(css)

        normalized = _normalize_indicator(
            value, cfg.thresholds, inverse=inverse,
        )

        return IndicatorResult(
            key=key,
            value=round(value, 4),
            label_ru=cfg.label_ru,
            formula=cfg.formula,
            emoji=emoji,
            css_class=css,
            status_ru=status,
            normalized=round(normalized, 4),
            description=cfg.description,
            methodology=cfg.methodology,
            example_url=cfg.example_url,
            example_text=cfg.example_text,
            source=source,
            is_proxy=proxy,
        )

    # ──────── Индекс угрозы ────────

    @staticmethod
    def _compute_threat_index(
        indicators: Dict[str, IndicatorResult],
    ) -> float:
        """
        Агрегированный Индекс угрозы (0–100).

            T = Σ(wᵢ × norm(Iᵢ)) × 100

        Веса определены в INDICATOR_WEIGHTS.
        """
        weighted_sum = 0.0
        for key, weight in INDICATOR_WEIGHTS.items():
            if key in indicators:
                weighted_sum += weight * indicators[key].normalized

        return round(np.clip(weighted_sum * 100.0, 0.0, 100.0), 1)

    # ──────── экспертная система рекомендаций ────────

    @staticmethod
    def _generate_recommendations(
        indicators: Dict[str, IndicatorResult],
        threat_index: float,
    ) -> List[str]:
        """
        Экспертная система рекомендаций на основе продукционных
        правил (rule-based, Buchanan & Shortliffe, 1984).

        Каждое правило:
            ЕСЛИ <условие на комбинацию индикаторов>
            ТО <рекомендация>

        Порядок: от общих к частным, от критических к превентивным.
        """
        recs: List[str] = []

        def _val(key: str) -> float:
            return indicators[key].value if key in indicators else 0.0

        def _css(key: str) -> str:
            return indicators[key].css_class if key in indicators else ""

        # ── Правило 1: Критический уровень ──
        if threat_index >= 70:
            recs.append(
                "⚠️ КРИТИЧЕСКИЙ УРОВЕНЬ УГРОЗЫ. "
                "Зафиксированы множественные признаки координированной "
                "информационной операции. Рекомендуется немедленное "
                "информирование ответственных структур и активация "
                "протокола противодействия."
            )

        # ── Правило 2: Высокая синхронность + гомогенность ──
        if (
            _css("synchrony_coeff") == "badge-red"
            and _css("text_homogeneity") == "badge-red"
        ):
            recs.append(
                "Обнаружена высокая синхронность публикаций "
                "в сочетании с текстовой однородностью — классический "
                "паттерн координированного недостоверного поведения "
                "(CIB, Gleicher 2020). Рекомендуется: "
                "верификация источников, поиск общей инфраструктуры "
                "(домены, IP-адреса, дата регистрации аккаунтов)."
            )

        # ── Правило 3: Бот-активность + скорость ──
        if (
            _css("bot_activity") in ("badge-red", "badge-yellow")
            and _css("spread_speed") == "badge-red"
        ):
            recs.append(
                "Высокая бот-активность в сочетании со взрывной "
                "скоростью распространения указывает на использование "
                "автоматизированных инструментов (бот-фермы, "
                "координированные репосты). Рекомендуется: "
                "анализ CV интервалов постинга и TTR подозрительных "
                "аккаунтов."
            )

        # ── Правило 4: Аномальный объём ──
        if _css("anomaly_index") == "badge-red":
            recs.append(
                "Зафиксирован экстремальный всплеск упоминаний "
                f"(Z-оценка: {_val('anomaly_index'):.2f}). "
                "Необходимо проверить, соответствует ли объём "
                "публикаций реальному информационному поводу. "
                "При отсутствии повода — вероятна искусственная "
                "накрутка."
            )

        # ── Правило 5: Тональный сдвиг без повода ──
        if _css("tonal_shift") in ("badge-red", "badge-yellow"):
            recs.append(
                "Обнаружен значительный тональный сдвиг "
                f"(|Δ| = {_val('tonal_shift'):.2f}). "
                "Рекомендуется: сопоставить с реальными событиями "
                "за последние 24 часа. Если объективный повод "
                "отсутствует — вероятна целенаправленная негативизация "
                "или позитивизация медиаполя."
            )

        # ── Правило 6: Текстовая гомогенность ──
        if _css("text_homogeneity") == "badge-red":
            recs.append(
                "Текстовая однородность публикаций превышает порог "
                "near-duplicate detection (H ≥ 0.85). Вероятно "
                "использование шаблонов или автоматической генерации "
                "текста. Рекомендуется: выборочная проверка "
                "формулировок на совпадения."
            )

        # ── Правило 7: Синхронность ──
        if (
            _css("synchrony_coeff") in ("badge-red", "badge-yellow")
            and _css("text_homogeneity") != "badge-red"
        ):
            recs.append(
                "Повышенная временная синхронность публикаций "
                "при умеренной текстовой вариативности может "
                "указывать на координированную, но адаптированную "
                "кампанию (гибридный тип). Рекомендуется: "
                "мониторинг в динамике, отслеживание нарастания "
                "однородности."
            )

        # ── Правило 8: Норма ──
        if threat_index <= 30:
            recs.append(
                "Индикаторы в пределах нормы. Признаков "
                "координированного информационного воздействия "
                "не обнаружено. Рекомендуется: продолжить "
                "фоновый мониторинг."
            )

        # ── Правило 9: Прокси-режим ──
        has_proxy = any(
            ind.is_proxy for ind in indicators.values()
        )
        if has_proxy:
            recs.append(
                "⚙️ Часть индикаторов вычислена через модельные "
                "прокси-оценки (данные GDELT недоступны). "
                "Точность агрегированного индекса снижена. "
                "Для полной диагностики подключите источник "
                "живых данных."
            )

        return recs


# ═══════════════════════════════════════════════════════════════════
# 8. UI-ПОМОЩНИКИ (для app.py)
# ═══════════════════════════════════════════════════════════════════

def render_indicator_card(result: IndicatorResult) -> str:
    """
    HTML-карточка индикатора для Streamlit (st.markdown).

    Отображает: эмодзи + название + значение + статус-бейдж.
    Используется на вкладке «Индикаторы обнаружения».
    """
    proxy_tag = (
        f' <span style="color: {COLORS.gray_nezumi}; '
        f'font-size: 0.7rem;">(модельная оценка)</span>'
        if result.is_proxy else ""
    )

    return f"""
    <div class="aio-card" style="border-left: 4px solid
        {'%s' % (COLORS.green_deep if result.css_class == 'badge-green'
                 else COLORS.yellow_kerria if result.css_class == 'badge-yellow'
                 else COLORS.red_crimson)};">
        <h3>{result.emoji} {result.label_ru}{proxy_tag}</h3>
        <div class="value">{result.value:.2f}</div>
        <div style="margin-top: 0.4rem;">
            <span class="{result.css_class}">{result.status_ru}</span>
        </div>
    </div>
    """


def render_indicator_detail(result: IndicatorResult) -> str:
    """
    HTML-блок развёрнутого пояснения индикатора.

    Отображает: формулу, пороги, методологию, живой пример.
    Используется во всплывающем окне при клике.
    """
    cfg = INDICATORS[result.key]
    g, y, r = cfg.thresholds

    # Для spread_speed — обратная логика
    if result.key == "spread_speed":
        thresholds_html = (
            f"🟢 > {g} суток &nbsp;|&nbsp; "
            f"🟡 > {y} суток &nbsp;|&nbsp; "
            f"🔴 ≤ {r} суток"
        )
    else:
        thresholds_html = (
            f"🟢 < {g} &nbsp;|&nbsp; "
            f"🟡 < {y} &nbsp;|&nbsp; "
            f"🔴 ≥ {r}"
        )

    return f"""
    <div class="aio-tooltip">
        <div class="tooltip-title">{result.label_ru}</div>
        <div style="margin-bottom: 0.6rem;">{cfg.description}</div>

        <div style="margin-bottom: 0.5rem;">
            <strong style="color: {COLORS.yellow_kerria};">Формула:</strong>
            <code style="background: rgba(255,255,255,0.08);
                         padding: 2px 6px; border-radius: 4px;">
                {cfg.formula}
            </code>
        </div>

        <div style="margin-bottom: 0.5rem;">
            <strong style="color: {COLORS.yellow_kerria};">Пороги:</strong>
            {thresholds_html}
        </div>

        <div style="margin-bottom: 0.5rem;">
            <strong style="color: {COLORS.yellow_kerria};">Обоснование:</strong>
            <span style="font-size: 0.82rem;">{cfg.methodology}</span>
        </div>

        <div style="margin-bottom: 0.3rem;">
            <strong style="color: {COLORS.yellow_kerria};">Живой пример:</strong>
            <span style="font-size: 0.82rem;">{cfg.example_text}</span>
        </div>
        <div>
            <a href="{cfg.example_url}" target="_blank"
               style="color: {COLORS.blue_indigo}; font-size: 0.8rem;">
                🔗 Источник примера
            </a>
        </div>
    </div>
    """


def render_threat_gauge(detection: DetectionResult) -> str:
    """
    HTML-блок агрегированного Индекса угрозы.

    Большой числовой дисплей + светофорный бейдж + визуальная полоса.
    """
    idx = detection.threat_index
    pct = min(idx, 100.0)

    # Цвет полосы по уровню
    if idx <= THREAT_INDEX_THRESHOLDS[0]:
        bar_color = COLORS.green_deep
    elif idx <= THREAT_INDEX_THRESHOLDS[1]:
        bar_color = COLORS.yellow_kerria
    else:
        bar_color = COLORS.red_crimson

    return f"""
    <div class="aio-card" style="text-align: center;
         border-top: 4px solid {bar_color};">
        <h3>ИНДЕКС УГРОЗЫ</h3>
        <div class="value" style="font-size: 3rem;">
            {detection.threat_emoji} {idx:.1f}
        </div>
        <div style="margin: 0.8rem 0;">
            <span class="{detection.threat_css}" style="font-size: 1rem;
                padding: 6px 16px;">
                {detection.threat_label}
            </span>
        </div>
        <div style="background: {COLORS.bg_main};
             border-radius: 8px; height: 12px; overflow: hidden;
             border: 1px solid rgba(0,0,0,0.08);">
            <div style="background: {bar_color};
                 width: {pct}%; height: 100%;
                 border-radius: 8px;
                 transition: width 0.5s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between;
             font-size: 0.7rem; color: {COLORS.text_secondary};
             margin-top: 0.3rem;">
            <span>0 — Норма</span>
            <span>100 — Критическая угроза</span>
        </div>
    </div>
    """


def render_recommendations(recommendations: List[str]) -> str:
    """HTML-блок рекомендаций."""
    if not recommendations:
        return ""

    items_html = ""
    for i, rec in enumerate(recommendations, 1):
        items_html += f"""
        <div style="padding: 0.7rem 0; border-bottom: 1px solid
             rgba(0,0,0,0.05); font-size: 0.88rem; line-height: 1.6;">
            <span style="color: {COLORS.blue_indigo};
                  font-weight: 600;">{i}.</span> {rec}
        </div>
        """

    return f"""
    <div class="aio-card" style="border-left: 4px solid {COLORS.blue_indigo};">
        <h3>РЕКОМЕНДАЦИИ ЭКСПЕРТНОЙ СИСТЕМЫ</h3>
        {items_html}
    </div>
    """


# ═══════════════════════════════════════════════════════════════════
# 9. СЕРИАЛИЗАЦИЯ (для library.py и report_gen.py)
# ═══════════════════════════════════════════════════════════════════

def detection_to_dict(result: DetectionResult) -> Dict:
    """
    Сериализация DetectionResult в JSON-совместимый словарь.

    Используется библиотекой результатов (library.py) для сохранения
    и генератором отчётов (report_gen.py) для экспорта.
    """
    return {
        "threat_index": result.threat_index,
        "threat_label": result.threat_label,
        "threat_emoji": result.threat_emoji,
        "has_proxy": result.has_proxy,
        "indicators": {
            key: {
                "key": ind.key,
                "value": ind.value,
                "label_ru": ind.label_ru,
                "formula": ind.formula,
                "emoji": ind.emoji,
                "css_class": ind.css_class,
                "status_ru": ind.status_ru,
                "normalized": ind.normalized,
                "source": ind.source,
                "is_proxy": ind.is_proxy,
            }
            for key, ind in result.indicators.items()
        },
        "recommendations": result.recommendations,
    }


def detection_from_dict(data: Dict) -> DetectionResult:
    """
    Десериализация DetectionResult из словаря.

    Восстанавливает полный объект из сохранённых данных
    (library.py → отображение на вкладке «Библиотека»).
    """
    indicators = {}
    for key, ind_data in data.get("indicators", {}).items():
        cfg = INDICATORS.get(key)
        indicators[key] = IndicatorResult(
            key=ind_data["key"],
            value=ind_data["value"],
            label_ru=ind_data["label_ru"],
            formula=ind_data["formula"],
            emoji=ind_data["emoji"],
            css_class=ind_data["css_class"],
            status_ru=ind_data["status_ru"],
            normalized=ind_data["normalized"],
            description=cfg.description if cfg else "",
            methodology=cfg.methodology if cfg else "",
            example_url=cfg.example_url if cfg else "",
            example_text=cfg.example_text if cfg else "",
            source=ind_data.get("source", "gdelt"),
            is_proxy=ind_data.get("is_proxy", False),
        )

    t_emoji, t_css, t_label = threat_index_level(
        data.get("threat_index", 0),
    )

    return DetectionResult(
        indicators=indicators,
        threat_index=data.get("threat_index", 0.0),
        threat_emoji=t_emoji,
        threat_css=t_css,
        threat_label=t_label,
        recommendations=data.get("recommendations", []),
        has_proxy=data.get("has_proxy", False),
    )
