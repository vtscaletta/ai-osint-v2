"""
AI-OSINT — Генерация аналитического PDF-отчёта
=================================================

Формирует PDF-документ на основе записей из библиотеки
результатов (SimulationRecord). Документ содержит:

    1. Титульная страница — метаданные, авторы, университет
    2. Сводная таблица экспериментов
    3. Индикаторы обнаружения — значения, светофоры, нормализация
    4. Агрегированный Индекс угрозы — визуальная шкала
    5. Рекомендации экспертной системы
    6. Параметры экспериментов — детализация каждой записи

Используемая библиотека: fpdf2 >= 2.7.0
    - TTF-шрифты (DejaVu Sans) для кириллицы
    - Программная отрисовка (нет зависимости от ОС)

Палитра: 和色 из config.py (WashokuPalette)

Теоретические основания:
    - ГОСТ Р 7.0.97-2016 — оформление документов
    - ISO/IEC 27001 — структура отчёта по инф. безопасности
    - Buchanan & Shortliffe (1984) — экспертные системы

Зависимости:
    fpdf2, config (внутренний), indicators (внутренний),
    library (внутренний)

Авторы:
    Абсаттаров Г.Р., Саятбек С., Кайролла А.Б.
    КазУМОиМЯ им. Абылай хана | 2026
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from fpdf import FPDF

from config import (
    COLORS,
    INDICATORS,
    THREAT_INDEX_THRESHOLDS,
    AUTHORS,
    UNIVERSITY,
    YEAR,
    COPYRIGHT,
    APP_TITLE,
    APP_SUBTITLE,
    APP_METHOD,
    MARKOV_STATE_ORDER,
    traffic_emoji,
    threat_index_level,
)
from indicators import (
    DetectionResult,
    IndicatorResult,
    detection_from_dict,
)
from library import SimulationRecord


# ═══════════════════════════════════════════════════════════════════
# 1. ЦВЕТА — HEX → RGB
# ═══════════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Конвертация HEX (#RRGGBB) в кортеж (R, G, B)."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# Предвычисленные RGB — профессиональная палитра Navy Blue
_C_BG_MAIN    = (255, 255, 255)
_C_BG_CARD    = (237, 242, 250)                # Светло-голубой фон карточек
_C_BG_DARK    = (15, 25, 50)                   # Глубокий navy
_C_NAVY       = (10, 30, 70)                   # Navy Blue основной
_C_INDIGO     = (25, 65, 155)                  # Royal Blue акцент
_C_ULTRA      = (20, 50, 120)                  # Тёмно-синий
_C_GREEN      = (15, 140, 70)                  # Профессиональный зелёный
_C_YELLOW     = (230, 160, 0)                  # Тёплый жёлтый
_C_RED        = (195, 40, 40)                  # Глубокий красный
_C_GRAY       = (90, 95, 110)                  # Серый текст
_C_SILVER     = (175, 180, 195)
_C_TEXT       = (20, 20, 25)                    # Почти чёрный
_C_TEXT2      = (60, 65, 80)                    # Тёмно-серый
_C_WHITE      = (255, 255, 255)


# ═══════════════════════════════════════════════════════════════════
# 2. ШРИФТЫ — DejaVu Sans (кириллица)
# ═══════════════════════════════════════════════════════════════════

# Стандартные пути к шрифтам на Linux (Streamlit Cloud — Debian)
_FONT_PATHS = {
    "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "bold":    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "italic":  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
}

# Liberation Serif = Times New Roman (метрически совместимый, кириллица)
_SERIF_PATHS = {
    "regular": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "bold":    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "italic":  "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
}


def _download_font(url: str, path: str) -> bool:
    """Скачивает шрифт по URL, если файл не существует."""
    if os.path.exists(path):
        return True
    try:
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
        return os.path.exists(path)
    except Exception:
        return False


# URL-ы DejaVu на GitHub (официальный релиз)
_DEJAVU_BASE = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/"
_DEJAVU_URLS = {
    "regular": _DEJAVU_BASE + "DejaVuSans.ttf",
    "bold":    _DEJAVU_BASE + "DejaVuSans-Bold.ttf",
    "italic":  _DEJAVU_BASE + "DejaVuSans-Oblique.ttf",
}
_DEJAVU_LOCAL = {
    "regular": "/tmp/fonts/DejaVuSans.ttf",
    "bold":    "/tmp/fonts/DejaVuSans-Bold.ttf",
    "italic":  "/tmp/fonts/DejaVuSans-Oblique.ttf",
}


def _register_fonts(pdf: FPDF) -> str:
    """
    Регистрирует шрифты. Приоритет:
    1. Liberation Serif (= Times New Roman, кириллица)
    2. DejaVu Sans (кириллица, fallback)
    3. Helvetica (без кириллицы, крайний fallback)
    """
    # Попытка 1: Liberation Serif (Times New Roman)
    if os.path.exists(_SERIF_PATHS["regular"]):
        pdf.add_font("TimesRU", "", _SERIF_PATHS["regular"])
        pdf.add_font("TimesRU", "B", _SERIF_PATHS["bold"])
        pdf.add_font("TimesRU", "I", _SERIF_PATHS["italic"])
        return "TimesRU"

    # Попытка 2: DejaVu Sans
    if os.path.exists(_FONT_PATHS["regular"]):
        pdf.add_font("DejaVu", "", _FONT_PATHS["regular"])
        if os.path.exists(_FONT_PATHS["bold"]):
            pdf.add_font("DejaVu", "B", _FONT_PATHS["bold"])
        else:
            pdf.add_font("DejaVu", "B", _FONT_PATHS["regular"])
        if os.path.exists(_FONT_PATHS["italic"]):
            pdf.add_font("DejaVu", "I", _FONT_PATHS["italic"])
        else:
            pdf.add_font("DejaVu", "I", _FONT_PATHS["regular"])
        return "DejaVu"

    # Попытка 3: скачивание DejaVu с GitHub
    all_ok = True
    for key in ["regular", "bold", "italic"]:
        if not _download_font(_DEJAVU_URLS[key], _DEJAVU_LOCAL[key]):
            all_ok = False
            break
    if all_ok and os.path.exists(_DEJAVU_LOCAL["regular"]):
        pdf.add_font("DejaVu", "", _DEJAVU_LOCAL["regular"])
        pdf.add_font("DejaVu", "B", _DEJAVU_LOCAL["bold"])
        pdf.add_font("DejaVu", "I", _DEJAVU_LOCAL["italic"])
        return "DejaVu"

    return "Helvetica"


# ═══════════════════════════════════════════════════════════════════
# 3. PDF-КЛАСС — AI-OSINT REPORT
# ═══════════════════════════════════════════════════════════════════

class _ReportPDF(FPDF):
    """
    Расширенный FPDF с хедером и футером в стиле AI-OSINT.

    Attributes
    ----------
    font_family : str
        Имя зарегистрированного семейства шрифтов.
    report_date : str
        Дата формирования отчёта (ДД.ММ.ГГГГ).
    """

    def __init__(self, font_family: str, report_date: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.font_family = font_family
        self.report_date = report_date
        self.set_auto_page_break(auto=True, margin=25)

    def header(self) -> None:
        """Верхний колонтитул: минимальный технологичный стиль."""
        if self.page_no() == 1:
            return

        # Тонкая акцентная линия сверху
        self.set_fill_color(*_C_INDIGO)
        self.rect(0, 0, self.w, 1.5, "F")

        self.set_y(5)
        self.set_font(self.font_family, "B", 7.5)
        self.set_text_color(*_C_NAVY)
        self.cell(0, 4, f"AI-OSINT", align="L")

        self.set_font(self.font_family, "", 7.5)
        self.set_text_color(*_C_GRAY)
        self.cell(0, 4, self.report_date, align="R", new_x="LMARGIN", new_y="NEXT")

        self.set_draw_color(220, 220, 230)
        self.set_line_width(0.15)
        self.line(self.l_margin, self.get_y() + 1,
                  self.w - self.r_margin, self.get_y() + 1)
        self.ln(4)

    def footer(self) -> None:
        """Нижний колонтитул: тонкая линия + номер страницы."""
        self.set_y(-15)
        self.set_draw_color(200, 200, 210)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(),
                  self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font(self.font_family, "", 7)
        self.set_text_color(150, 150, 160)
        self.cell(0, 5, f"AI-OSINT  |  {self.report_date}", align="L")
        self.cell(0, 5, f"{self.page_no()}/{{nb}}", align="R")


# ═══════════════════════════════════════════════════════════════════
# 4. КОМПОНЕНТЫ ОТЧЁТА
# ═══════════════════════════════════════════════════════════════════

def _draw_title_page(pdf: _ReportPDF) -> None:
    """Компактный заголовок с блоком О документе — всё на стр. 1."""
    pdf.add_page()

    # ── Яркая акцентная полоса сверху ──
    pdf.set_fill_color(*_C_INDIGO)
    pdf.rect(0, 0, pdf.w, 4, "F")
    pdf.set_fill_color(*_C_YELLOW)
    pdf.rect(0, 4, 40, 1.5, "F")

    # ── AI-OSINT + дата ──
    pdf.set_xy(pdf.l_margin, 10)
    pdf.set_font(pdf.font_family, "B", 22)
    pdf.set_text_color(*_C_NAVY)
    pdf.cell(100, 11, APP_TITLE)

    pdf.set_font(pdf.font_family, "", 11)
    pdf.set_text_color(*_C_GRAY)
    pdf.cell(0, 11, pdf.report_date, align="R")
    pdf.ln(13)

    # Подзаголовок
    pdf.set_font(pdf.font_family, "", 11)
    pdf.set_text_color(*_C_TEXT2)
    pdf.cell(0, 6, "Аналитический отчёт  \u2014  " + APP_METHOD,
             new_x="LMARGIN", new_y="NEXT")

    # Разделитель
    pdf.ln(3)
    y = pdf.get_y()
    pdf.set_draw_color(*_C_INDIGO)
    pdf.set_line_width(0.8)
    pdf.line(pdf.l_margin, y, pdf.l_margin + 35, y)
    pdf.set_draw_color(200, 205, 215)
    pdf.set_line_width(0.2)
    pdf.line(pdf.l_margin + 37, y, pdf.w - pdf.r_margin, y)
    pdf.ln(5)

    # ── Блок «О документе» ──
    card_y = pdf.get_y()
    card_w = pdf.w - pdf.l_margin - pdf.r_margin
    # Calculate card height based on content
    card_h = 52
    pdf.set_fill_color(*_C_BG_CARD)
    pdf.rect(pdf.l_margin, card_y, card_w, card_h, "F")
    pdf.set_fill_color(*_C_INDIGO)
    pdf.rect(pdf.l_margin, card_y, 3, card_h, "F")

    pdf.set_xy(pdf.l_margin + 8, card_y + 4)
    pdf.set_font(pdf.font_family, "B", 14)
    pdf.set_text_color(*_C_NAVY)
    pdf.cell(0, 7, "О документе")
    pdf.ln(8)
    pdf.set_x(pdf.l_margin + 8)
    pdf.set_font(pdf.font_family, "", 12)
    pdf.set_text_color(*_C_TEXT)
    pdf.multi_cell(card_w - 14, 6, (
        "Настоящий отчёт сформирован автоматически аналитической "
        "платформой AI-OSINT на основе результатов вычислительного "
        "имитационного моделирования информационного поля Казахстана "
        "в глобальном медиапространстве. Отчёт содержит сводную таблицу "
        "экспериментов, значения шести индикаторов обнаружения "
        "информационных операций, агрегированный Индекс угрозы (0\u2013100), "
        "рекомендации экспертной системы и параметры воспроизводимости."
    ))

    pdf.set_y(card_y + card_h + 4)


def _draw_section_header(pdf: _ReportPDF, title: str) -> None:
    """Заголовок секции: яркий, крупный."""
    if pdf.get_y() > pdf.h - 40:
        pdf.add_page()

    pdf.ln(5)
    y = pdf.get_y()

    # Фон
    pdf.set_fill_color(230, 238, 250)
    pdf.rect(pdf.l_margin, y, pdf.w - pdf.l_margin - pdf.r_margin, 11, "F")

    # Индиго-акцент слева
    pdf.set_fill_color(*_C_INDIGO)
    pdf.rect(pdf.l_margin, y, 3, 11, "F")

    # Жёлтый акцент снизу
    pdf.set_fill_color(*_C_YELLOW)
    pdf.rect(pdf.l_margin, y + 11, 25, 1.2, "F")

    pdf.set_xy(pdf.l_margin + 7, y + 1)
    pdf.set_font(pdf.font_family, "B", 14)
    pdf.set_text_color(*_C_NAVY)
    pdf.cell(0, 9, title)
    pdf.set_y(y + 15)


def _draw_summary_table(
    pdf: _ReportPDF,
    records: List[SimulationRecord],
) -> None:
    """Сводная таблица экспериментов."""
    _draw_section_header(pdf, "1. Сводная таблица экспериментов")

    pdf.set_font(pdf.font_family, "", 14)
    pdf.set_text_color(*_C_TEXT)
    pdf.multi_cell(0, 7, (
        "Перечень симуляций, включённых в отчёт."
    ))
    pdf.ln(3)

    # Заголовки таблицы
    col_widths = [32, 40, 20, 20, 36, 22]  # мм
    headers = ["Дата", "Сценарий", "Агентов", "Суток", "Макс. фаза", "Индекс"]

    pdf.set_font(pdf.font_family, "B", 8)
    pdf.set_fill_color(*_C_INDIGO)
    pdf.set_text_color(*_C_WHITE)
    pdf.set_draw_color(220, 220, 230)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=0, fill=True, align="C")
    pdf.ln()

    # Строки данных
    pdf.set_font(pdf.font_family, "", 8)
    pdf.set_text_color(*_C_TEXT)

    for idx, rec in enumerate(records):
        if idx % 2 == 0:
            pdf.set_fill_color(248, 248, 252)
        else:
            pdf.set_fill_color(*_C_WHITE)

        row = [
            rec.display_time or "—",
            rec.scenario_label or rec.scenario_key,
            str(rec.n_agents),
            str(rec.n_steps),
            rec.max_state_label or "—",
            f"{rec.threat_index:.1f}",
        ]
        for i, cell_text in enumerate(row):
            pdf.cell(col_widths[i], 7, cell_text, border=0,
                     fill=True, align="C")
        # Нижняя линия строки
        pdf.ln()
        pdf.set_draw_color(235, 235, 240)
        pdf.set_line_width(0.1)
        pdf.line(pdf.l_margin, pdf.get_y(),
                 pdf.l_margin + sum(col_widths), pdf.get_y())

    pdf.ln(3)


def _traffic_color(css_class: str) -> Tuple[int, int, int]:
    """RGB-кортеж по CSS-классу светофора."""
    if css_class == "badge-green":
        return _C_GREEN
    elif css_class == "badge-yellow":
        return _C_YELLOW
    return _C_RED


def _traffic_text(css_class: str) -> str:
    """Текстовый статус по CSS-классу."""
    if css_class == "badge-green":
        return "Норма"
    elif css_class == "badge-yellow":
        return "Внимание"
    return "Угроза"


def _traffic_marker(css_class: str) -> str:
    """Текстовый маркер светофора для PDF (вместо эмодзи)."""
    if css_class == "badge-green":
        return "[OK]"
    elif css_class == "badge-yellow":
        return "[!]"
    return "[!!]"


def _threat_marker(threat_index: float) -> str:
    """Текстовый маркер Индекса угрозы для PDF (вместо эмодзи)."""
    if threat_index <= THREAT_INDEX_THRESHOLDS[0]:
        return "[OK]"
    elif threat_index <= THREAT_INDEX_THRESHOLDS[1]:
        return "[!]"
    return "[!!]"


def _draw_indicators(
    pdf: _ReportPDF,
    detection: DetectionResult,
    record_label: str = "",
) -> None:
    """Блок индикаторов: 6 карточек + Индекс угрозы."""
    suffix = f" ({record_label})" if record_label else ""
    _draw_section_header(pdf, f"2. Индикаторы обнаружения{suffix}")

    pdf.set_font(pdf.font_family, "", 14)
    pdf.set_text_color(*_C_TEXT)
    pdf.multi_cell(0, 7, (
        "Шесть индикаторов обнаружения информационных операций. "
        "Каждый имеет три порога: норма, внимание, угроза."
    ))
    pdf.ln(3)

    # Порядок индикаторов — как в спеке
    indicator_order = [
        "anomaly_index",
        "synchrony_coeff",
        "text_homogeneity",
        "tonal_shift",
        "spread_speed",
        "bot_activity",
    ]

    for key in indicator_order:
        ind = detection.indicators.get(key)
        if ind is None:
            continue

        if pdf.get_y() > pdf.h - 35:
            pdf.add_page()

        cfg = INDICATORS.get(key)
        color = _traffic_color(ind.css_class)

        # ── Фоновая карточка ──
        y_start = pdf.get_y()
        card_h = 28
        pdf.set_fill_color(*_C_BG_CARD)
        pdf.rect(pdf.l_margin, y_start, pdf.w - pdf.l_margin - pdf.r_margin, card_h, "F")

        # Акцентная полоска слева
        pdf.set_fill_color(*color)
        pdf.rect(pdf.l_margin, y_start, 3, card_h, "F")

        # СТРОКА 1: Маркер + название
        pdf.set_xy(pdf.l_margin + 7, y_start + 2)
        pdf.set_font(pdf.font_family, "B", 13)
        pdf.set_text_color(*_C_TEXT)
        marker = _traffic_marker(ind.css_class)
        pdf.cell(0, 7, f"{marker}  {ind.label_ru}",
                 new_x="LMARGIN", new_y="NEXT")

        # СТРОКА 2: Значение + статус-бейдж + norm
        pdf.set_x(pdf.l_margin + 7)
        pdf.set_font(pdf.font_family, "B", 20)
        pdf.set_text_color(*color)
        val_str = f"{ind.value:.2f}" if ind.value < 100 else f"{ind.value:.1f}"
        pdf.cell(40, 9, val_str, align="L")

        # Статус-бейдж
        pdf.set_font(pdf.font_family, "B", 10)
        if ind.css_class == "badge-yellow":
            pdf.set_text_color(*_C_TEXT)
        else:
            pdf.set_text_color(*_C_WHITE)
        pdf.set_fill_color(*color)
        badge_w = pdf.get_string_width(ind.status_ru) + 12
        pdf.cell(badge_w, 9, ind.status_ru, align="C", fill=True)

        pdf.set_font(pdf.font_family, "", 9)
        pdf.set_text_color(*_C_GRAY)
        pdf.cell(0, 9, f"  norm: {ind.normalized:.2f}", align="L")
        pdf.ln()

        # СТРОКА 3: Формула + пороги с кружочками
        if cfg:
            pdf.set_x(pdf.l_margin + 7)
            pdf.set_font(pdf.font_family, "", 9)
            pdf.set_text_color(*_C_TEXT2)

            g, y_thresh, r = cfg.thresholds
            if key == "spread_speed":
                thresh_str = f"Пороги:  >{g} суток  |  >{y_thresh} суток  |  <={r} суток"
            else:
                thresh_str = f"Пороги:  <{g}  |  <{y_thresh}  |  >={r}"

            proxy_str = "  |  модельная оценка" if ind.is_proxy else ""
            pdf.cell(0, 5, f"{cfg.formula}  |  {thresh_str}{proxy_str}",
                     new_x="LMARGIN", new_y="NEXT")

        pdf.set_y(y_start + card_h + 3)


def _draw_threat_index(
    pdf: _ReportPDF,
    detection: DetectionResult,
) -> None:
    """Блок агрегированного Индекса угрозы с визуальной шкалой."""
    _draw_section_header(pdf, "3. Агрегированный Индекс угрозы")

    idx = detection.threat_index
    pct = min(idx / 100.0, 1.0)

    # Определение цвета
    if idx <= THREAT_INDEX_THRESHOLDS[0]:
        bar_color = _C_GREEN
    elif idx <= THREAT_INDEX_THRESHOLDS[1]:
        bar_color = _C_YELLOW
    else:
        bar_color = _C_RED

    # Большое число
    pdf.set_font(pdf.font_family, "B", 42)
    pdf.set_text_color(*bar_color)
    pdf.cell(0, 22, f"{idx:.1f}", align="C",
             new_x="LMARGIN", new_y="NEXT")

    # Уровень
    pdf.set_font(pdf.font_family, "B", 16)
    pdf.cell(0, 10, detection.threat_label, align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Визуальная шкала (горизонтальная полоса)
    bar_x = pdf.l_margin + 10
    bar_w = pdf.w - pdf.l_margin - pdf.r_margin - 20
    bar_y = pdf.get_y()
    bar_h = 10

    # Фон шкалы
    pdf.set_fill_color(*_C_SILVER)
    pdf.rect(bar_x, bar_y, bar_w, bar_h, "F")

    # Заполненная часть
    pdf.set_fill_color(*bar_color)
    pdf.rect(bar_x, bar_y, bar_w * pct, bar_h, "F")

    # Рамка
    pdf.set_draw_color(*_C_GRAY)
    pdf.set_line_width(0.2)
    pdf.rect(bar_x, bar_y, bar_w, bar_h, "D")

    pdf.set_y(bar_y + bar_h + 2)

    # Подписи шкалы
    pdf.set_font(pdf.font_family, "", 11)
    pdf.set_text_color(*_C_GRAY)
    pdf.set_x(bar_x)
    pdf.cell(bar_w / 3, 6, "0 \u2014 Норма", align="L")
    pdf.cell(bar_w / 3, 6, "30\u201360 \u2014 Внимание", align="C")
    pdf.cell(bar_w / 3, 6, "100 \u2014 Критическая", align="R")
    pdf.ln(8)

    # Пояснение
    pdf.set_font(pdf.font_family, "", 14)
    pdf.set_text_color(*_C_TEXT)
    pdf.multi_cell(0, 7, (
        "Индекс угрозы (0\u2013100) = сумма взвешенных "
        "нормализованных значений 6 индикаторов обнаружения."
    ))

    if detection.has_proxy:
        pdf.ln(2)
        pdf.set_font(pdf.font_family, "I", 11)
        pdf.set_text_color(*_C_GRAY)
        pdf.multi_cell(0, 5.5, (
            "Примечание: часть индикаторов вычислена через "
            "модельные прокси-оценки. Точность снижена."
        ))


def _draw_recommendations(
    pdf: _ReportPDF,
    recommendations: List[str],
) -> None:
    """Блок рекомендаций экспертной системы."""
    _draw_section_header(pdf, "4. Рекомендации экспертной системы")

    if not recommendations:
        pdf.set_font(pdf.font_family, "I", 14)
        pdf.set_text_color(*_C_TEXT2)
        pdf.cell(0, 8, "Рекомендации не сформированы.",
                 new_x="LMARGIN", new_y="NEXT")
        return

    for i, rec in enumerate(recommendations, 1):
        if pdf.get_y() > pdf.h - 30:
            pdf.add_page()

        pdf.set_font(pdf.font_family, "B", 14)
        pdf.set_text_color(*_C_INDIGO)
        pdf.cell(10, 7, f"{i}.", align="R")

        rec_text = rec
        if len(rec_text) > 2 and not rec_text[0].isalnum():
            for skip in [4, 3, 2]:
                if len(rec_text) > skip and rec_text[skip - 1] == " ":
                    rec_text = rec_text[skip:]
                    break

        pdf.set_font(pdf.font_family, "", 14)
        pdf.set_text_color(*_C_TEXT)
        pdf.set_x(pdf.l_margin + 12)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 14, 7, rec_text)
        pdf.ln(3)


def _draw_experiment_details(
    pdf: _ReportPDF,
    records: List[SimulationRecord],
) -> None:
    """Детали каждого эксперимента: параметры + Монте-Карло."""
    _draw_section_header(pdf, "5. Параметры экспериментов")

    for idx, rec in enumerate(records, 1):
        if pdf.get_y() > pdf.h - 50:
            pdf.add_page()

        # Заголовок записи
        pdf.set_font(pdf.font_family, "B", 10)
        pdf.set_text_color(*_C_NAVY)
        pdf.cell(0, 7,
                 f"Эксперимент {idx}: {rec.scenario_label}",
                 new_x="LMARGIN", new_y="NEXT")

        # Акцентная линия
        pdf.set_draw_color(*_C_YELLOW)
        pdf.set_line_width(0.4)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 25, pdf.get_y())
        pdf.ln(3)

        # Параметры — компактная таблица
        params = [
            ("Дата", rec.display_time or "—"),
            ("Сценарий", rec.scenario_label or rec.scenario_key),
            ("Агентов", str(rec.n_agents)),
            ("Шагов (суток)", str(rec.n_steps)),
            ("Seed", str(rec.seed)),
            ("Макс. фаза", rec.max_state_label or "—"),
            ("Финальная фаза", rec.final_state_label or "—"),
            ("Активных агентов", f"{rec.final_active_ratio:.1%}"),
            ("Индекс угрозы", f"{rec.threat_index:.1f}"),
            ("Уровень", rec.threat_label or "—"),
        ]

        pdf.set_font(pdf.font_family, "", 14)
        for label, value in params:
            pdf.set_text_color(*_C_TEXT2)
            pdf.cell(55, 7, label + ":", align="R")
            pdf.set_text_color(*_C_TEXT)
            pdf.set_font(pdf.font_family, "B", 14)
            pdf.cell(0, 7, f"  {value}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(pdf.font_family, "", 14)

        # Монте-Карло (если есть)
        if rec.mc_viral_probability is not None:
            pdf.ln(2)
            pdf.set_font(pdf.font_family, "B", 9)
            pdf.set_text_color(*_C_ULTRA)
            pdf.cell(0, 5, "Монте-Карло:", new_x="LMARGIN", new_y="NEXT")

            pdf.set_font(pdf.font_family, "", 8.5)
            mc_params = [
                ("Итераций", str(rec.mc_n_simulations or "—")),
                ("P(вирусная фаза)", f"{rec.mc_viral_probability:.1%}"),
            ]

            if rec.mc_viral_ci_lower is not None and rec.mc_viral_ci_upper is not None:
                mc_params.append((
                    "95% ДИ",
                    f"[{rec.mc_viral_ci_lower:.1%} — {rec.mc_viral_ci_upper:.1%}]",
                ))

            if rec.mc_mean_steps_to_viral is not None:
                mc_params.append((
                    "Среднее шагов до вирусной",
                    f"{rec.mc_mean_steps_to_viral:.1f} суток",
                ))

            for label, value in mc_params:
                pdf.set_text_color(*_C_TEXT2)
                pdf.cell(55, 5, label + ":", align="R")
                pdf.set_text_color(*_C_TEXT)
                pdf.set_font(pdf.font_family, "B", 8.5)
                pdf.cell(0, 5, f"  {value}", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font(pdf.font_family, "", 8.5)

        # Примечание
        if rec.note:
            pdf.ln(2)
            pdf.set_font(pdf.font_family, "I", 8)
            pdf.set_text_color(*_C_GRAY)
            pdf.multi_cell(0, 4.5, f"Примечание: {rec.note}")

        pdf.ln(5)


def _draw_methodology_note(pdf: _ReportPDF) -> None:
    """Краткая методологическая справка в конце отчёта."""
    if pdf.get_y() > pdf.h - 60:
        pdf.add_page()

    _draw_section_header(pdf, "6. Методологическая справка")

    pdf.set_font(pdf.font_family, "", 14)
    pdf.set_text_color(*_C_TEXT)
    pdf.multi_cell(0, 7, (
        "Результаты получены методами вычислительного "
        "имитационного моделирования:"
    ))
    pdf.ln(3)

    methods = [
        ("OSINT", "разведка на основе открытых источников (GDELT Project)"),
        ("NLP", "обработка естественного языка (TF-IDF, тональность)"),
        ("АОМ", "агентно-ориентированное моделирование (Барабаши\u2013Альберт)"),
        ("Марков", "5 состояний нарратива с динамической матрицей"),
        ("М-Карло", "стохастическое моделирование (100\u20135000 итераций)"),
    ]

    for abbr, desc in methods:
        pdf.set_font(pdf.font_family, "B", 12)
        pdf.set_text_color(*_C_NAVY)
        pdf.cell(25, 7, abbr, align="R")
        pdf.set_font(pdf.font_family, "", 12)
        pdf.set_text_color(*_C_TEXT)
        pdf.cell(0, 7, f"  \u2014 {desc}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)

    pdf.set_font(pdf.font_family, "I", 10)
    pdf.set_text_color(*_C_GRAY)
    pdf.multi_cell(0, 5.5, (
        "Калибровка: Stanford IO (Nimmo, 2019), Meta CIB (Gleicher, 2020), "
        "Botometer (Varol et al., 2017), GDELT Documentation."
    ))


def _draw_significance(pdf: _ReportPDF) -> None:
    """Блок значимости исследования (по материалам Саятбек С.)."""
    if pdf.get_y() > pdf.h - 60:
        pdf.add_page()

    _draw_section_header(pdf, "7. Значимость исследования")

    sections = [
        ("Теоретическая значимость", (
            "Проект опирается на современные подходы к исследованию "
            "информационной безопасности и распространения нарративов "
            "в сетевой среде. Применяются методы агентно-ориентированного "
            "моделирования, марковские цепи и метод Монте-Карло для "
            "оценки вероятностных сценариев эскалации."
        )),
        ("Социальная значимость", (
            "Система направлена на укрепление устойчивости общества "
            "к деструктивным информационным воздействиям: выявление "
            "волн, затрагивающих межэтнические отношения, региональные "
            "противоречия, протестные настроения, чувствительные темы "
            "идентичности и исторической памяти."
        )),
        ("Политическая значимость", (
            "Инструмент аналитического сопровождения государственных "
            "решений в сфере информационной безопасности и стратегических "
            "коммуникаций. Мониторинг внешнего информационного давления, "
            "укрепление цифрового суверенитета."
        )),
    ]

    for title, text in sections:
        if pdf.get_y() > pdf.h - 30:
            pdf.add_page()

        pdf.set_font(pdf.font_family, "B", 9)
        pdf.set_text_color(*_C_NAVY)
        pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font(pdf.font_family, "", 8.5)
        pdf.set_text_color(*_C_TEXT)
        pdf.multi_cell(0, 4.5, text)
        pdf.ln(3)


# ═══════════════════════════════════════════════════════════════════
# 5. ГЛАВНАЯ ФУНКЦИЯ — ГЕНЕРАЦИЯ PDF
# ═══════════════════════════════════════════════════════════════════

def generate_report(
    records: List[SimulationRecord],
    report_date: Optional[str] = None,
) -> bytes:
    """
    Генерирует PDF-отчёт из списка SimulationRecord.

    Основная точка входа модуля. Вызывается из app.py (вкладка 7).

    Parameters
    ----------
    records : List[SimulationRecord]
        Записи из библиотеки результатов.
        Каждая запись должна содержать detection_data (сериализованный
        DetectionResult через detection_to_dict).
    report_date : str, optional
        Дата отчёта в формате «ДД.ММ.ГГГГ». Если не указана —
        текущая дата UTC.

    Returns
    -------
    bytes
        PDF-документ в виде байтовой строки.
        Готов для st.download_button(data=...).

    Raises
    ------
    ValueError
        Если список записей пуст.

    Example
    -------
    ::

        from report_gen import generate_report
        from library import ResultsLibrary

        lib = ResultsLibrary()
        records = lib.get_selected(selected_ids)
        pdf_bytes = generate_report(records)
        st.download_button("Скачать PDF", data=pdf_bytes,
                           file_name="report.pdf", mime="application/pdf")
    """
    if not records:
        raise ValueError("Список записей для отчёта пуст.")

    if report_date is None:
        report_date = datetime.now(timezone.utc).strftime("%d.%m.%Y")

    # ── Инициализация PDF ──
    pdf = _ReportPDF(font_family="Helvetica", report_date=report_date)
    font = _register_fonts(pdf)
    pdf.font_family = font
    pdf.alias_nb_pages()

    # ── 1. Заголовок + сводная таблица (на одной странице) ──
    _draw_title_page(pdf)
    _draw_summary_table(pdf, records)

    # ── 3–4. Индикаторы + Индекс угрозы ──
    # Если одна запись — показываем её индикаторы.
    # Если несколько — агрегируем (средние по всем записям).
    detections: List[DetectionResult] = []
    for rec in records:
        if rec.detection_data:
            det = detection_from_dict(rec.detection_data)
            detections.append(det)

    if detections:
        if len(detections) == 1:
            # Единственная запись
            _draw_indicators(pdf, detections[0])
            _draw_threat_index(pdf, detections[0])
            _draw_recommendations(pdf, detections[0].recommendations)
        else:
            # Несколько записей — выводим по каждой,
            # затем агрегированную сводку
            for i, (rec, det) in enumerate(zip(records, detections)):
                label = f"{rec.scenario_label}, {rec.display_time}"
                _draw_indicators(pdf, det, record_label=label)

            # Агрегированный индекс — среднее
            avg_threat = sum(d.threat_index for d in detections) / len(detections)
            t_emoji, t_css, t_label = threat_index_level(avg_threat)

            # Собираем все рекомендации (уникальные)
            all_recs: List[str] = []
            seen: set = set()
            for det in detections:
                for r in det.recommendations:
                    if r not in seen:
                        all_recs.append(r)
                        seen.add(r)

            # Создаём агрегированный DetectionResult
            # (берём индикаторы первой записи как репрезентативные)
            agg_detection = DetectionResult(
                indicators=detections[0].indicators,
                threat_index=avg_threat,
                threat_emoji=t_emoji,
                threat_css=t_css,
                threat_label=t_label,
                recommendations=all_recs,
                has_proxy=any(d.has_proxy for d in detections),
            )

            _draw_section_header(pdf, "Агрегированная оценка")
            pdf.set_font(pdf.font_family, "I", 9)
            pdf.set_text_color(*_C_TEXT2)
            pdf.multi_cell(0, 5, (
                f"Средний Индекс угрозы по {len(detections)} "
                f"экспериментам."
            ))
            pdf.ln(2)

            _draw_threat_index(pdf, agg_detection)
            _draw_recommendations(pdf, all_recs)
    else:
        # Нет данных индикаторов
        pdf.add_page()
        pdf.set_font(pdf.font_family, "I", 10)
        pdf.set_text_color(*_C_TEXT2)
        pdf.cell(0, 8,
                 "Данные индикаторов отсутствуют в выбранных записях.",
                 new_x="LMARGIN", new_y="NEXT")

    # ── 5. Параметры экспериментов ──
    _draw_experiment_details(pdf, records)

    # ── 6. Методологическая справка ──
    _draw_methodology_note(pdf)

    # ── Выгрузка в bytes ──
    return bytes(pdf.output())
