"""
AI-OSINT — Движок данных
==============================
Подключение к живому GDELT API, сбор и обработка данных
по информационному полю Казахстана в 7 языках.

Источники:
  - GDELT DOC 2.0 API — поиск статей, тональность, объём
  - GDELT Timeline API — временные ряды
  - Кэширование через Streamlit @st.cache_data (TTL)
  - Rate limiting: 1 запрос / 5 сек (требование GDELT)

Языки: EN, RU, ZH, FR, ES, AR, KZ (6 ООН + казахский)
"""

import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import requests
import numpy as np
import pandas as pd
import streamlit as st

from config import GDELT, COLORS, SOURCE_RELIABILITY_LEVELS

# ═══════════════════════════════════════════════════════════════════
# Логирование
# ═══════════════════════════════════════════════════════════════════

logger = logging.getLogger("ai_osint.data_engine")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════
# 1. RATE LIMITER
# ═══════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Глобальный ограничитель частоты запросов к GDELT API.
    GDELT требует ≥5 секунд между запросами.
    Реализация через простое ожидание (sleep).
    """

    def __init__(self, min_interval: float = GDELT.rate_limit_seconds):
        self._min_interval = min_interval
        self._last_request: float = 0.0

    def wait(self) -> None:
        """Блокирует выполнение до истечения минимального интервала."""
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._min_interval:
            wait_time = self._min_interval - elapsed
            logger.debug(f"Rate limit: ожидание {wait_time:.1f} сек")
            time.sleep(wait_time)
        self._last_request = time.time()


# Глобальный экземпляр — один на весь процесс Streamlit
_rate_limiter = RateLimiter()


# ═══════════════════════════════════════════════════════════════════
# 2. БАЗОВЫЙ HTTP-КЛИЕНТ
# ═══════════════════════════════════════════════════════════════════

def _gdelt_request(url: str, params: Dict[str, Any]) -> Optional[Dict]:
    """
    Выполняет HTTP GET к GDELT API с rate limiting.

    Возвращает:
        JSON-ответ как dict, или None при ошибке.
    """
    _rate_limiter.wait()

    try:
        resp = requests.get(
            url,
            params=params,
            timeout=GDELT.timeout_seconds,
            headers={"User-Agent": "AI-OSINT (Academic Research)"},
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        logger.warning(f"Таймаут запроса: {url}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP ошибка {e.response.status_code}: {url}")
        return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"Ошибка соединения: {url}")
        return None
    except ValueError:
        # JSON decode error — GDELT иногда возвращает HTML при ошибках
        logger.warning(f"Невалидный JSON: {url}")
        return None


def _gdelt_request_text(url: str, params: Dict[str, Any]) -> Optional[str]:
    """
    Выполняет HTTP GET, возвращает сырой текст ответа.
    Для Timeline API, который возвращает CSV.
    """
    _rate_limiter.wait()

    try:
        resp = requests.get(
            url,
            params=params,
            timeout=GDELT.timeout_seconds,
            headers={"User-Agent": "AI-OSINT (Academic Research)"},
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f"Ошибка запроса: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# 3. GDELT DOC API — ПОИСК СТАТЕЙ
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GDELTArticle:
    """Структура одной статьи из GDELT."""
    url: str
    title: str
    source_domain: str
    source_country: str
    language: str
    tone: float                      # Average Tone (−10…+10)
    date: datetime
    image_url: Optional[str] = None
    reliability: str = "unknown"     # Ключ из SOURCE_RELIABILITY_LEVELS


@st.cache_data(ttl=GDELT.cache_ttl_seconds, show_spinner=False)
def fetch_articles(
    query: str = GDELT.query_term,
    days_back: int = GDELT.default_days_back,
    lang: Optional[str] = None,
    max_records: int = GDELT.max_records,
    sort: str = "DateDesc",
) -> List[Dict[str, Any]]:
    """
    Получает статьи из GDELT DOC 2.0 API.

    Параметры:
        query: поисковый запрос (по умолчанию "Kazakhstan")
        days_back: глубина поиска в днях
        lang: код языка GDELT (eng, rus, zho, ...) или None для всех
        max_records: максимум записей
        sort: сортировка (DateDesc, ToneAsc, ToneDesc, ...)

    Возвращает:
        Список словарей с данными статей.
    """
    start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")
    end_date = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": "ArtList",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "maxrecords": max_records,
        "sort": sort,
        "format": "json",
    }

    if lang:
        params["sourcelang"] = lang

    data = _gdelt_request(GDELT.doc_api_base, params)
    if not data:
        return []

    articles = data.get("articles", [])

    result = []
    for art in articles:
        result.append({
            "url": art.get("url", ""),
            "title": art.get("title", ""),
            "source_domain": art.get("domain", ""),
            "source_country": art.get("sourcecountry", ""),
            "language": art.get("language", ""),
            "tone": float(art.get("tone", 0.0)),
            "date": art.get("seendate", ""),
            "image_url": art.get("socialimage", ""),
        })

    return result


@st.cache_data(ttl=GDELT.cache_ttl_seconds, show_spinner=False)
def fetch_articles_all_languages(
    query: str = GDELT.query_term,
    days_back: int = GDELT.default_days_back,
    max_per_lang: int = 50,
) -> pd.DataFrame:
    """
    Собирает статьи по всем 7 языкам и возвращает объединённый DataFrame.

    Возвращает DataFrame с колонками:
        url, title, source_domain, source_country, language,
        tone, date, image_url, reliability, lang_label
    """
    all_articles = []

    for lang_key, lang_code in GDELT.lang_codes.items():
        articles = fetch_articles(
            query=query,
            days_back=days_back,
            lang=lang_code,
            max_records=max_per_lang,
        )
        lang_label = GDELT.languages[lang_key]
        for art in articles:
            art["lang_key"] = lang_key
            art["lang_label"] = lang_label
            art["reliability"] = classify_source_reliability(art["source_domain"])
        all_articles.extend(articles)

    if not all_articles:
        return _empty_articles_df()

    df = pd.DataFrame(all_articles)

    # Парсинг даты (GDELT формат: "YYYYMMDDTHHMMSS" или подобный)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df


def _empty_articles_df() -> pd.DataFrame:
    """Возвращает пустой DataFrame с правильной структурой."""
    return pd.DataFrame(columns=[
        "url", "title", "source_domain", "source_country",
        "language", "tone", "date", "image_url",
        "lang_key", "lang_label", "reliability",
    ])


# ═══════════════════════════════════════════════════════════════════
# 4. GDELT TIMELINE API — ВРЕМЕННЫЕ РЯДЫ
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=GDELT.cache_ttl_seconds, show_spinner=False)
def fetch_volume_timeline(
    query: str = GDELT.query_term,
    days_back: int = GDELT.default_days_back,
    resolution: str = "day",
) -> pd.DataFrame:
    """
    Частота упоминаний по дням (Timeline Vol).

    Возвращает DataFrame:
        date (datetime), volume (int)
    """
    start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")
    end_date = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": "TimelineVol",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "format": "json",
    }

    data = _gdelt_request(GDELT.doc_api_base, params)
    if not data:
        return _empty_timeline_df()

    timeline = data.get("timeline", [])
    if not timeline:
        return _empty_timeline_df()

    # GDELT Timeline возвращает список серий; берём первую
    series = timeline[0].get("data", []) if timeline else []

    rows = []
    for point in series:
        dt_str = point.get("date", "")
        val = point.get("value", 0)
        rows.append({"date": dt_str, "volume": int(val)})

    if not rows:
        return _empty_timeline_df()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


@st.cache_data(ttl=GDELT.cache_ttl_seconds, show_spinner=False)
def fetch_tone_timeline(
    query: str = GDELT.query_term,
    days_back: int = GDELT.default_days_back,
) -> pd.DataFrame:
    """
    Средняя тональность по дням (Timeline Tone).

    Возвращает DataFrame:
        date (datetime), tone (float)
    """
    start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")
    end_date = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": "TimelineTone",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "format": "json",
    }

    data = _gdelt_request(GDELT.doc_api_base, params)
    if not data:
        return _empty_tone_df()

    timeline = data.get("timeline", [])
    if not timeline:
        return _empty_tone_df()

    series = timeline[0].get("data", []) if timeline else []

    rows = []
    for point in series:
        dt_str = point.get("date", "")
        val = point.get("value", 0.0)
        rows.append({"date": dt_str, "tone": float(val)})

    if not rows:
        return _empty_tone_df()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


@st.cache_data(ttl=GDELT.cache_ttl_seconds, show_spinner=False)
def fetch_source_country_breakdown(
    query: str = GDELT.query_term,
    days_back: int = GDELT.default_days_back,
) -> pd.DataFrame:
    """
    Распределение публикаций по странам-источникам.

    Возвращает DataFrame:
        country (str), count (int), share (float)
    """
    start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")
    end_date = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    params = {
        "query": query,
        "mode": "TimelineSourceCountry",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "format": "json",
    }

    data = _gdelt_request(GDELT.doc_api_base, params)
    if not data:
        return pd.DataFrame(columns=["country", "count", "share"])

    timeline = data.get("timeline", [])

    # Агрегируем по странам из всех временных точек
    country_totals: Dict[str, int] = {}
    for series_item in timeline:
        country = series_item.get("series", "Unknown")
        data_points = series_item.get("data", [])
        total = sum(int(p.get("value", 0)) for p in data_points)
        country_totals[country] = country_totals.get(country, 0) + total

    if not country_totals:
        return pd.DataFrame(columns=["country", "count", "share"])

    df = pd.DataFrame([
        {"country": k, "count": v}
        for k, v in sorted(country_totals.items(), key=lambda x: -x[1])
    ])
    total = df["count"].sum()
    df["share"] = (df["count"] / total * 100).round(1) if total > 0 else 0.0

    return df


@st.cache_data(ttl=GDELT.cache_ttl_seconds, show_spinner=False)
def fetch_language_breakdown(
    query: str = GDELT.query_term,
    days_back: int = GDELT.default_days_back,
) -> pd.DataFrame:
    """
    Языковое распределение публикаций из 7 целевых языков.

    Возвращает DataFrame:
        language (str), lang_label (str), count (int), share (float)
    """
    language_counts = {}

    for lang_key, lang_code in GDELT.lang_codes.items():
        articles = fetch_articles(
            query=query,
            days_back=days_back,
            lang=lang_code,
            max_records=1,  # Нам нужен только count, не сами статьи
            sort="DateDesc",
        )
        # GDELT не возвращает total count напрямую в ArtList;
        # используем len результата как нижнюю оценку, либо
        # запрашиваем Timeline Vol для каждого языка
        count = _fetch_language_volume(query, days_back, lang_code)
        language_counts[lang_key] = count

    rows = []
    total = sum(language_counts.values())
    for lang_key, count in language_counts.items():
        rows.append({
            "language": lang_key,
            "lang_label": GDELT.languages[lang_key],
            "count": count,
            "share": round(count / total * 100, 1) if total > 0 else 0.0,
        })

    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return df


def _fetch_language_volume(query: str, days_back: int, lang_code: str) -> int:
    """Суммарный объём публикаций на одном языке за период."""
    start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")
    end_date = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    params = {
        "query": f"{query} sourcelang:{lang_code}",
        "mode": "TimelineVol",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "format": "json",
    }

    data = _gdelt_request(GDELT.doc_api_base, params)
    if not data:
        return 0

    timeline = data.get("timeline", [])
    if not timeline:
        return 0

    series = timeline[0].get("data", []) if timeline else []
    return sum(int(p.get("value", 0)) for p in series)


def _empty_timeline_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "volume"])


def _empty_tone_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "tone"])


# ═══════════════════════════════════════════════════════════════════
# 5. МАРКИРОВКА НАДЁЖНОСТИ ИСТОЧНИКОВ
# ═══════════════════════════════════════════════════════════════════

# Базовый классификатор: домен → степень надёжности.
# В production подключить Media Bias/Fact Check API.
# Здесь — ручной маппинг основных доменов.

_VERIFIED_DOMAINS = frozenset({
    # Международные агентства
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "france24.com", "dw.com", "aljazeera.com", "theguardian.com",
    "nytimes.com", "washingtonpost.com", "ft.com", "economist.com",
    "lemonde.fr", "elpais.com", "asahi.com", "scmp.com",
    # Казахстанские проверенные
    "inform.kz", "kazinform.kz", "lsm.kz",
    "vlast.kz", "forbes.kz", "kapital.kz",
})

_STATE_DOMAINS = frozenset({
    # Государственные СМИ (не 'фейк', но требуют перекрёстной проверки)
    "rt.com", "sputniknews.com", "tass.com", "ria.ru",
    "xinhuanet.com", "globaltimes.cn", "cgtn.com",
    "presstv.ir", "trt.net.tr",
    "akorda.kz", "primeminister.kz",
    "government.kz", "strategy2050.kz",
})

_TABLOID_DOMAINS = frozenset({
    "dailymail.co.uk", "thesun.co.uk", "nypost.com",
    "mirror.co.uk", "express.co.uk",
})


def classify_source_reliability(domain: str) -> str:
    """
    Классифицирует домен-источник по степени надёжности.

    НЕ определяет 'фейк/не фейк' (юридически опасно),
    а присваивает степень надёжности для визуальной маркировки.

    Параметры:
        domain: доменное имя источника

    Возвращает:
        Ключ из SOURCE_RELIABILITY_LEVELS: 'verified', 'state', 'tabloid', 'unknown'
    """
    domain_lower = domain.lower().strip()

    # Убираем www.
    if domain_lower.startswith("www."):
        domain_lower = domain_lower[4:]

    if domain_lower in _VERIFIED_DOMAINS:
        return "verified"
    elif domain_lower in _STATE_DOMAINS:
        return "state"
    elif domain_lower in _TABLOID_DOMAINS:
        return "tabloid"
    else:
        return "unknown"


def reliability_badge_html(reliability_key: str) -> str:
    """Возвращает HTML-бейдж надёжности источника."""
    info = SOURCE_RELIABILITY_LEVELS.get(reliability_key)
    if not info:
        info = SOURCE_RELIABILITY_LEVELS["unknown"]

    text_color = "white" if reliability_key != "tabloid" else COLORS.text_primary
    return (
        f'<span style="background:{info.color}; color:{text_color}; '
        f'padding:2px 8px; border-radius:4px; font-size:0.75rem; '
        f'font-weight:600;">{info.label_ru}</span>'
    )


# ═══════════════════════════════════════════════════════════════════
# 6. АГРЕГАЦИИ И РАСЧЁТЫ
# ═══════════════════════════════════════════════════════════════════

def compute_moving_average(series: pd.Series, window: int = 30) -> pd.Series:
    """
    Скользящее среднее (MA).

    Параметры:
        series: числовой ряд
        window: размер окна (по умолчанию 30 — MA30)

    Возвращает:
        pd.Series с MA
    """
    return series.rolling(window=window, min_periods=1).mean()


def compute_z_score(value: float, mean: float, std: float) -> float:
    """
    Z-оценка для Индекса Аномальности.

    Z = (x − μ) / σ

    Если σ = 0, возвращает 0.0 (нет вариации = нет аномалии).
    """
    if std == 0:
        return 0.0
    return (value - mean) / std


def compute_tonal_shift(tone_series: pd.Series) -> float:
    """
    Тональный сдвиг: абсолютное изменение средней тональности за 24 часа.

    Берёт последние 2 значения в ряде (предполагается дневная гранулярность).
    """
    if len(tone_series) < 2:
        return 0.0
    return abs(float(tone_series.iloc[-1] - tone_series.iloc[-2]))


def compute_synchrony_coefficient(
    articles_df: pd.DataFrame,
    window_hours: float = 2.0,
) -> float:
    """
    Коэффициент синхронности: доля статей, вышедших в пиковом окне.

    Логика:
        1. Группируем по часам
        2. Находим пиковое окно (window_hours)
        3. S = articles_in_peak_window / total_articles
    """
    if articles_df.empty or "date" not in articles_df.columns:
        return 0.0

    df = articles_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])

    if len(df) < 2:
        return 0.0

    # Берём последние 24 часа
    cutoff = df["date"].max() - timedelta(hours=24)
    df_24h = df[df["date"] >= cutoff]

    if len(df_24h) < 2:
        return 0.0

    total = len(df_24h)

    # Группируем по часовым интервалам
    df_24h = df_24h.set_index("date")
    hourly = df_24h.resample("1h").size()

    # Скользящее окно = window_hours
    window_size = max(1, int(window_hours))
    rolling_sum = hourly.rolling(window=window_size, min_periods=1).sum()
    peak_count = rolling_sum.max()

    return float(peak_count / total) if total > 0 else 0.0


def compute_volume_anomaly(volume_df: pd.DataFrame, ma_window: int = 30) -> Dict[str, float]:
    """
    Рассчитывает аномальность текущего объёма публикаций.

    Возвращает словарь:
        current_volume: сегодняшний объём
        ma30: скользящее среднее за 30 дней
        std30: стандартное отклонение за 30 дней
        z_score: Z-оценка
    """
    if volume_df.empty or len(volume_df) < 2:
        return {"current_volume": 0, "ma30": 0.0, "std30": 0.0, "z_score": 0.0}

    volumes = volume_df["volume"].values
    current = float(volumes[-1])
    window = volumes[-ma_window:] if len(volumes) >= ma_window else volumes
    mean = float(np.mean(window))
    std = float(np.std(window))
    z = compute_z_score(current, mean, std)

    return {
        "current_volume": current,
        "ma30": round(mean, 1),
        "std30": round(std, 1),
        "z_score": round(z, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# 7. ГЕНЕРАТОР ДАЙДЖЕСТА
# ═══════════════════════════════════════════════════════════════════

def generate_digest(
    articles_df: pd.DataFrame,
    days: int = 3,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Дайджест за N дней: топ-N статей по абсолютной тональности.

    Логика: берём статьи с самой высокой |tone| (и позитив, и негатив),
    потому что они наиболее «окрашенные» и информативные.
    """
    if articles_df.empty:
        return articles_df

    df = articles_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    cutoff = datetime.utcnow().replace(tzinfo=df["date"].dt.tz) - timedelta(days=days)
    df = df[df["date"] >= cutoff]

    if df.empty:
        return df

    df["abs_tone"] = df["tone"].abs()
    df = df.sort_values("abs_tone", ascending=False).head(top_n)
    df = df.drop(columns=["abs_tone"])

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 8. ПРОВЕРКА ДОСТУПНОСТИ API
# ═══════════════════════════════════════════════════════════════════

def check_gdelt_health() -> Tuple[bool, str]:
    """
    Проверяет доступность GDELT API.

    Возвращает:
        (is_ok, message)
    """
    try:
        resp = requests.get(
            GDELT.doc_api_base,
            params={
                "query": "test",
                "mode": "ArtList",
                "maxrecords": 1,
                "format": "json",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return True, "GDELT API доступен"
        else:
            return False, f"GDELT API: HTTP {resp.status_code}"
    except Exception as e:
        return False, f"GDELT API недоступен: {str(e)}"


# ═══════════════════════════════════════════════════════════════════
# 9. FALLBACK — ДЕМО-ДАННЫЕ
# ═══════════════════════════════════════════════════════════════════

def generate_fallback_volume(days: int = 90) -> pd.DataFrame:
    """
    Генерирует демонстрационный временной ряд объёма публикаций.
    Используется ТОЛЬКО если GDELT API недоступен.
    Визуально помечается в UI как «демо-режим».
    """
    np.random.seed(42)
    dates = pd.date_range(
        end=datetime.utcnow(),
        periods=days,
        freq="D",
        tz="UTC",
    )
    # Базовый тренд + шум + 2 всплеска
    base = 120 + np.random.normal(0, 15, days).cumsum() * 0.3
    base = np.maximum(base, 20)

    # Всплеск 1 (середина периода)
    spike1_center = days // 3
    for i in range(max(0, spike1_center - 3), min(days, spike1_center + 4)):
        base[i] *= 2.5 + np.random.uniform(0, 0.5)

    # Всплеск 2 (последняя четверть)
    spike2_center = int(days * 0.8)
    for i in range(max(0, spike2_center - 2), min(days, spike2_center + 3)):
        base[i] *= 1.8 + np.random.uniform(0, 0.3)

    return pd.DataFrame({
        "date": dates,
        "volume": np.round(base).astype(int),
    })


def generate_fallback_tone(days: int = 90) -> pd.DataFrame:
    """Демо-данные тональности."""
    np.random.seed(43)
    dates = pd.date_range(
        end=datetime.utcnow(),
        periods=days,
        freq="D",
        tz="UTC",
    )
    # Тональность: слегка позитивная + шум + негативные скачки
    tone = np.random.normal(0.8, 0.4, days)
    # Негативный скачок в середине (корреляция с volume spike)
    spike_center = days // 3
    for i in range(max(0, spike_center - 3), min(days, spike_center + 4)):
        tone[i] -= np.random.uniform(2.0, 4.0)

    tone = np.clip(tone, -10, 10)

    return pd.DataFrame({
        "date": dates,
        "tone": np.round(tone, 2),
    })


def generate_fallback_countries() -> pd.DataFrame:
    """Демо-данные по странам-источникам."""
    data = [
        {"country": "United States", "count": 3200, "share": 28.5},
        {"country": "United Kingdom", "count": 1800, "share": 16.0},
        {"country": "Russia", "count": 1500, "share": 13.4},
        {"country": "Kazakhstan", "count": 1200, "share": 10.7},
        {"country": "China", "count": 900, "share": 8.0},
        {"country": "Turkey", "count": 650, "share": 5.8},
        {"country": "Germany", "count": 500, "share": 4.5},
        {"country": "France", "count": 450, "share": 4.0},
        {"country": "India", "count": 350, "share": 3.1},
        {"country": "Другие", "count": 670, "share": 6.0},
    ]
    return pd.DataFrame(data)


def generate_fallback_languages() -> pd.DataFrame:
    """Демо-данные языкового распределения."""
    data = [
        {"language": "english", "lang_label": "Английский", "count": 5200, "share": 46.3},
        {"language": "russian", "lang_label": "Русский", "count": 2100, "share": 18.7},
        {"language": "chinese", "lang_label": "Китайский", "count": 1100, "share": 9.8},
        {"language": "french", "lang_label": "Французский", "count": 850, "share": 7.6},
        {"language": "spanish", "lang_label": "Испанский", "count": 700, "share": 6.2},
        {"language": "arabic", "lang_label": "Арабский", "count": 650, "share": 5.8},
        {"language": "kazakh", "lang_label": "Казахский", "count": 630, "share": 5.6},
    ]
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════
# 10. ФАСАДНАЯ ФУНКЦИЯ — ЕДИНАЯ ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DataSnapshot:
    """
    Полный снимок данных за один запрос.
    Используется во всех вкладках.
    """
    volume_df: pd.DataFrame = field(default_factory=_empty_timeline_df)
    tone_df: pd.DataFrame = field(default_factory=_empty_tone_df)
    countries_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    languages_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    articles_df: pd.DataFrame = field(default_factory=_empty_articles_df)
    anomaly: Dict[str, float] = field(default_factory=dict)
    tonal_shift: float = 0.0
    synchrony: float = 0.0
    is_live: bool = False       # True = живые данные, False = демо
    timestamp: str = ""
    error_msg: str = ""


def load_data(
    query: str = GDELT.query_term,
    days_back: int = GDELT.default_days_back,
) -> DataSnapshot:
    """
    Загружает все данные (живые или fallback) и возвращает DataSnapshot.

    Логика:
        1. Проверяем доступность GDELT API
        2. Если доступен — загружаем живые данные
        3. Если нет — генерируем демо-данные, помечаем флагом
        4. Рассчитываем агрегаты (аномальность, тональный сдвиг, синхронность)
    """
    snapshot = DataSnapshot(timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

    # Проверка API
    is_ok, msg = check_gdelt_health()

    if is_ok:
        try:
            snapshot.volume_df = fetch_volume_timeline(query, days_back)
            snapshot.tone_df = fetch_tone_timeline(query, days_back)
            snapshot.countries_df = fetch_source_country_breakdown(query, days_back)
            snapshot.languages_df = fetch_language_breakdown(query, days_back)
            snapshot.articles_df = fetch_articles_all_languages(query, days_back)
            snapshot.is_live = True
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            snapshot.error_msg = str(e)
            _fill_fallback(snapshot, days_back)
    else:
        snapshot.error_msg = msg
        _fill_fallback(snapshot, days_back)

    # Расчёт агрегатов
    snapshot.anomaly = compute_volume_anomaly(snapshot.volume_df)
    if not snapshot.tone_df.empty:
        snapshot.tonal_shift = compute_tonal_shift(snapshot.tone_df["tone"])
    if not snapshot.articles_df.empty:
        snapshot.synchrony = compute_synchrony_coefficient(snapshot.articles_df)

    return snapshot


def _fill_fallback(snapshot: DataSnapshot, days_back: int) -> None:
    """Заполняет snapshot демо-данными."""
    snapshot.volume_df = generate_fallback_volume(days_back)
    snapshot.tone_df = generate_fallback_tone(days_back)
    snapshot.countries_df = generate_fallback_countries()
    snapshot.languages_df = generate_fallback_languages()
    snapshot.is_live = False
