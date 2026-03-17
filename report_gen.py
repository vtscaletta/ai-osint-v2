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


# Предвычисленные RGB-кортежи из палитры 和色
_C_BG_MAIN    = _hex_to_rgb(COLORS.bg_main)        # Молочно-белый
_C_BG_CARD    = _hex_to_rgb(COLORS.bg_card)         # Слоновая кость
_C_BG_DARK    = _hex_to_rgb(COLORS.bg_dark)         # Чернильный
_C_NAVY       = _hex_to_rgb(COLORS.blue_navy)       # Тёмно-синий
_C_INDIGO     = _hex_to_rgb(COLORS.blue_indigo)     # Индиго
_C_ULTRA      = _hex_to_rgb(COLORS.blue_ultra)      # Ультрамарин
_C_GREEN      = _hex_to_rgb(COLORS.green_deep)      # Вечнозелёный
_C_YELLOW     = _hex_to_rgb(COLORS.yellow_kerria)   # Керрия
_C_RED        = _hex_to_rgb(COLORS.red_crimson)      # Алый
_C_GRAY       = _hex_to_rgb(COLORS.gray_nezumi)     # Мышиный серый
_C_SILVER     = _hex_to_rgb(COLORS.gray_silver)     # Серебристый
_C_TEXT       = _hex_to_rgb(COLORS.text_primary)     # Основной текст
_C_TEXT2      = _hex_to_rgb(COLORS.text_secondary)   # Вторичный текст
_C_WHITE      = (255, 255, 255)


# ═══════════════════════════════════════════════════════════════════
# 2. ШРИФТЫ — DejaVu Sans (кириллица)
# ═══════════════════════════════════════════════════════════════════

# Стандартные пути к DejaVu на Linux (Streamlit Cloud — Debian)
_FONT_PATHS = {
    "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "bold":    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "italic":  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
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
    Регистрирует TTF-шрифты с поддержкой кириллицы.

    Порядок поиска:
        1. Системные пути (/usr/share/fonts)
        2. /tmp/fonts (скачанные ранее)
        3. Скачивание с GitHub
        4. Helvetica (fallback без кириллицы)
    """
    # Попытка 1: системные шрифты
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

    # Попытка 2: поиск в системе
    for search_dir in ["/usr/share/fonts", "/usr/local/share/fonts"]:
        for root, _dirs, files in os.walk(search_dir):
            for f in files:
                if f == "DejaVuSans.ttf":
                    base = root
                    pdf.add_font("DejaVu", "", os.path.join(base, "DejaVuSans.ttf"))
                    bold_path = os.path.join(base, "DejaVuSans-Bold.ttf")
                    pdf.add_font("DejaVu", "B", bold_path if os.path.exists(bold_path) else os.path.join(base, "DejaVuSans.ttf"))
                    italic_path = os.path.join(base, "DejaVuSans-Oblique.ttf")
                    pdf.add_font("DejaVu", "I", italic_path if os.path.exists(italic_path) else os.path.join(base, "DejaVuSans.ttf"))
                    return "DejaVu"

    # Попытка 3: скачивание с GitHub
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

    # Крайний fallback
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
        """Верхний колонтитул: логотип + название + линия."""
        if self.page_no() == 1:
            # Титульная страница — без стандартного хедера
            return

        self.set_font(self.font_family, "B", 8)
        self.set_text_color(*_C_NAVY)
        self.cell(0, 5, f"{APP_TITLE} — {APP_SUBTITLE}", align="L")

        self.set_font(self.font_family, "", 8)
        self.set_text_color(*_C_GRAY)
        self.cell(0, 5, self.report_date, align="R", new_x="LMARGIN", new_y="NEXT")

        # Линия-разделитель (индиго)
        self.set_draw_color(*_C_INDIGO)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y() + 1,
                  self.w - self.r_margin, self.get_y() + 1)
        self.ln(5)

    def footer(self) -> None:
        """Нижний колонтитул: копирайт + номер страницы."""
        self.set_y(-15)
        self.set_font(self.font_family, "", 7)
        self.set_text_color(*_C_GRAY)
        self.cell(0, 5, COPYRIGHT, align="L")
        self.cell(0, 5, f"Страница {self.page_no()}/{{nb}}", align="R")


# ═══════════════════════════════════════════════════════════════════
# 4. КОМПОНЕНТЫ ОТЧЁТА
# ═══════════════════════════════════════════════════════════════════

def _draw_title_page(pdf: _ReportPDF) -> None:
    """Титульная страница: темный блок + метаданные + авторы."""
    pdf.add_page()

    # ── Тёмный блок-хедер (имитация aio-header) ──
    block_h = 70
    pdf.set_fill_color(*_C_BG_DARK)
    pdf.rect(0, 0, pdf.w, block_h, "F")

    # Акцентная линия слева (индиго)
    pdf.set_fill_color(*_C_INDIGO)
    pdf.rect(0, 0, 5, block_h, "F")

    # Название системы
    pdf.set_xy(15, 15)
    pdf.set_font(pdf.font_family, "B", 22)
    pdf.set_text_color(*_C_WHITE)
    pdf.cell(0, 10, APP_TITLE, new_x="LMARGIN", new_y="NEXT")

    # Подзаголовок
    pdf.set_x(15)
    pdf.set_font(pdf.font_family, "", 11)
    pdf.set_text_color(*_C_SILVER)
    pdf.multi_cell(pdf.w - 30, 6, APP_SUBTITLE)

    # Метод
    pdf.set_x(15)
    pdf.set_font(pdf.font_family, "I", 9)
    pdf.set_text_color(*_C_SILVER)
    pdf.cell(0, 8, APP_METHOD)

    # ── Основной блок ──
    pdf.set_y(block_h + 15)
    pdf.set_text_color(*_C_TEXT)

    # Заголовок отчёта
    pdf.set_font(pdf.font_family, "B", 16)
    pdf.cell(0, 10, "Аналитический отчёт", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Дата формирования
    pdf.set_font(pdf.font_family, "", 11)
    pdf.set_text_color(*_C_TEXT2)
    pdf.cell(0, 7,
             f"Дата формирования: {pdf.report_date}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # ── Авторы ──
    pdf.set_font(pdf.font_family, "B", 11)
    pdf.set_text_color(*_C_NAVY)
    pdf.cell(0, 8, "Авторы", new_x="LMARGIN", new_y="NEXT")

    pdf.set_draw_color(*_C_YELLOW)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 30, pdf.get_y())
    pdf.ln(3)

    pdf.set_font(pdf.font_family, "", 10)
    pdf.set_text_color(*_C_TEXT)
    for author in AUTHORS:
        pdf.cell(0, 6,
                 f"{author['name']} — {author['role']} ({author['title']})",
                 new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    pdf.set_font(pdf.font_family, "", 10)
    pdf.set_text_color(*_C_TEXT2)
    pdf.cell(0, 6, UNIVERSITY, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"{YEAR} г.", new_x="LMARGIN", new_y="NEXT")

    # ── Линия-разделитель внизу ──
    pdf.ln(15)
    pdf.set_draw_color(*_C_INDIGO)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(),
             pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(8)

    # Аннотация
    pdf.set_font(pdf.font_family, "I", 9)
    pdf.set_text_color(*_C_TEXT2)
    pdf.multi_cell(0, 5, (
        "Настоящий отчёт сформирован автоматически на основе "
        "результатов вычислительного имитационного моделирования "
        "информационного поля Казахстана в глобальном медиапространстве. "
        "Содержит значения индикаторов обнаружения информационных "
        "операций, агрегированный Индекс угрозы и рекомендации "
        "экспертной системы."
    ))


def _draw_section_header(pdf: _ReportPDF, title: str) -> None:
    """Заголовок секции: жирный текст + жёлтая линия."""
    if pdf.get_y() > pdf.h - 40:
        pdf.add_page()

    pdf.ln(6)
    pdf.set_font(pdf.font_family, "B", 13)
    pdf.set_text_color(*_C_NAVY)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")

    pdf.set_draw_color(*_C_YELLOW)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 45, pdf.get_y())
    pdf.ln(5)


def _draw_summary_table(
    pdf: _ReportPDF,
    records: List[SimulationRecord],
) -> None:
    """Сводная таблица экспериментов."""
    _draw_section_header(pdf, "1. Сводная таблица экспериментов")

    pdf.set_font(pdf.font_family, "", 9)
    pdf.set_text_color(*_C_TEXT2)
    pdf.multi_cell(0, 5, (
        "Перечень симуляций, включённых в отчёт. Каждая строка — "
        "один эксперимент с указанием сценария, параметров и "
        "агрегированного Индекса угрозы."
    ))
    pdf.ln(3)

    # Заголовки таблицы
    col_widths = [32, 40, 20, 20, 36, 22]  # мм
    headers = ["Дата", "Сценарий", "Агентов", "Суток", "Макс. фаза", "Индекс"]

    pdf.set_font(pdf.font_family, "B", 8)
    pdf.set_fill_color(*_C_NAVY)
    pdf.set_text_color(*_C_WHITE)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
    pdf.ln()

    # Строки данных
    pdf.set_font(pdf.font_family, "", 8)
    pdf.set_text_color(*_C_TEXT)

    for idx, rec in enumerate(records):
        # Чередование фона строк
        if idx % 2 == 0:
            pdf.set_fill_color(*_C_BG_CARD)
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
            pdf.cell(col_widths[i], 6, cell_text, border=1,
                     fill=True, align="C")
        pdf.ln()

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

    pdf.set_font(pdf.font_family, "", 9)
    pdf.set_text_color(*_C_TEXT2)
    pdf.multi_cell(0, 5, (
        "Шесть индикаторов обнаружения информационных операций. "
        "Каждый индикатор имеет три порога: норма, повышенное "
        "внимание, угроза. Значения нормализованы для "
        "агрегации в Индекс угрозы."
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

        if pdf.get_y() > pdf.h - 30:
            pdf.add_page()

        cfg = INDICATORS.get(key)
        color = _traffic_color(ind.css_class)

        # Акцентная полоска слева
        y_start = pdf.get_y()
        pdf.set_fill_color(*color)
        pdf.rect(pdf.l_margin, y_start, 2, 14, "F")

        # Название + эмодзи + значение
        pdf.set_x(pdf.l_margin + 5)
        pdf.set_font(pdf.font_family, "B", 10)
        pdf.set_text_color(*_C_TEXT)
        pdf.cell(85, 7, f"{_traffic_marker(ind.css_class)}  {ind.label_ru}", align="L")

        # Значение
        pdf.set_font(pdf.font_family, "B", 12)
        pdf.set_text_color(*color)
        pdf.cell(25, 7, f"{ind.value:.2f}", align="C")

        # Статус-бейдж
        pdf.set_font(pdf.font_family, "B", 8)
        pdf.set_fill_color(*color)
        if ind.css_class == "badge-yellow":
            pdf.set_text_color(*_C_TEXT)
        else:
            pdf.set_text_color(*_C_WHITE)
        pdf.cell(30, 7, ind.status_ru, align="C", fill=True)

        # Нормализованное значение
        pdf.set_font(pdf.font_family, "", 8)
        pdf.set_text_color(*_C_GRAY)
        pdf.cell(0, 7, f"[norm: {ind.normalized:.2f}]", align="R")
        pdf.ln()

        # Формула и пороги
        if cfg:
            pdf.set_x(pdf.l_margin + 5)
            pdf.set_font(pdf.font_family, "I", 7.5)
            pdf.set_text_color(*_C_TEXT2)

            g, y, r = cfg.thresholds
            if key == "spread_speed":
                thresh_str = f">{g} / >{y} / <={r} суток"
            else:
                thresh_str = f"<{g} / <{y} / >={r}"

            pdf.cell(0, 5,
                     f"Формула: {cfg.formula}  |  Пороги: {thresh_str}",
                     new_x="LMARGIN", new_y="NEXT")

        # Прокси-метка
        if ind.is_proxy:
            pdf.set_x(pdf.l_margin + 5)
            pdf.set_font(pdf.font_family, "I", 7)
            pdf.set_text_color(*_C_GRAY)
            pdf.cell(0, 4, "(модельная оценка — данные GDELT недоступны)",
                     new_x="LMARGIN", new_y="NEXT")

        pdf.ln(2)


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
    pdf.set_font(pdf.font_family, "B", 28)
    pdf.set_text_color(*bar_color)
    pdf.cell(0, 15, f"{_threat_marker(idx)}  {idx:.1f}", align="C",
             new_x="LMARGIN", new_y="NEXT")

    # Уровень
    pdf.set_font(pdf.font_family, "B", 12)
    pdf.cell(0, 8, detection.threat_label, align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Визуальная шкала (горизонтальная полоса)
    bar_x = pdf.l_margin + 10
    bar_w = pdf.w - pdf.l_margin - pdf.r_margin - 20
    bar_y = pdf.get_y()
    bar_h = 8

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
    pdf.set_font(pdf.font_family, "", 7)
    pdf.set_text_color(*_C_GRAY)
    pdf.set_x(bar_x)
    pdf.cell(bar_w / 3, 4, "0 — Норма", align="L")
    pdf.cell(bar_w / 3, 4, "30–60 — Внимание", align="C")
    pdf.cell(bar_w / 3, 4, "100 — Критическая", align="R")
    pdf.ln(5)

    # Пояснение формулы
    pdf.set_font(pdf.font_family, "I", 8)
    pdf.set_text_color(*_C_TEXT2)
    pdf.multi_cell(0, 4.5, (
        "Индекс угрозы (0-100) = сумма взвешенных нормализованных "
        "значений 6 индикаторов. Веса отражают диагностическую "
        "значимость каждого индикатора для обнаружения "
        "координированных информационных операций."
    ))

    if detection.has_proxy:
        pdf.ln(2)
        pdf.set_font(pdf.font_family, "I", 7.5)
        pdf.set_text_color(*_C_GRAY)
        pdf.multi_cell(0, 4, (
            "Примечание: часть индикаторов вычислена через модельные "
            "прокси-оценки (данные GDELT недоступны). Точность "
            "агрегированного индекса снижена."
        ))


def _draw_recommendations(
    pdf: _ReportPDF,
    recommendations: List[str],
) -> None:
    """Блок рекомендаций экспертной системы."""
    _draw_section_header(pdf, "4. Рекомендации экспертной системы")

    if not recommendations:
        pdf.set_font(pdf.font_family, "I", 10)
        pdf.set_text_color(*_C_TEXT2)
        pdf.cell(0, 7, "Рекомендации не сформированы.",
                 new_x="LMARGIN", new_y="NEXT")
        return

    pdf.set_font(pdf.font_family, "", 9)
    pdf.set_text_color(*_C_TEXT2)
    pdf.multi_cell(0, 5, (
        "Экспертная система на основе продукционных правил "
        "(rule-based engine) анализирует комбинации значений "
        "индикаторов и формирует рекомендации."
    ))
    pdf.ln(3)

    for i, rec in enumerate(recommendations, 1):
        if pdf.get_y() > pdf.h - 25:
            pdf.add_page()

        # Номер (индиго)
        pdf.set_font(pdf.font_family, "B", 9)
        pdf.set_text_color(*_C_INDIGO)
        pdf.cell(8, 5, f"{i}.", align="R")

        # Текст рекомендации
        # Убираем эмодзи в начале (если есть) для чистоты PDF
        rec_text = rec
        if len(rec_text) > 2 and not rec_text[0].isalnum():
            # Пропускаем эмодзи-префикс (2-4 символа)
            for skip in [4, 3, 2]:
                if len(rec_text) > skip and rec_text[skip - 1] == " ":
                    rec_text = rec_text[skip:]
                    break

        pdf.set_font(pdf.font_family, "", 9)
        pdf.set_text_color(*_C_TEXT)
        # Используем multi_cell для длинных рекомендаций
        x_before = pdf.get_x()
        pdf.set_x(pdf.l_margin + 10)
        pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 12, 5, rec_text)
        pdf.ln(1)


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

        pdf.set_font(pdf.font_family, "", 8.5)
        for label, value in params:
            pdf.set_text_color(*_C_TEXT2)
            pdf.cell(45, 5, label + ":", align="R")
            pdf.set_text_color(*_C_TEXT)
            pdf.set_font(pdf.font_family, "B", 8.5)
            pdf.cell(0, 5, f"  {value}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(pdf.font_family, "", 8.5)

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

    pdf.set_font(pdf.font_family, "", 8.5)
    pdf.set_text_color(*_C_TEXT)
    pdf.multi_cell(0, 5, (
        "Результаты получены методами вычислительного имитационного "
        "моделирования. Система интегрирует:"
    ))
    pdf.ln(2)

    methods = [
        ("OSINT", "разведка на основе открытых источников (GDELT Project)"),
        ("NLP", "обработка естественного языка (TF-IDF, анализ тональности)"),
        ("АОМ", "агентно-ориентированное моделирование (граф Барабаши-Альберт, 4 типа агентов)"),
        ("Цепи Маркова", "5 состояний нарратива (латентная - зарождение - рост - вирусная - затухание)"),
        ("Монте-Карло", "стохастическое моделирование (100-5000 итераций, 95% доверительный интервал)"),
    ]

    for abbr, desc in methods:
        pdf.set_font(pdf.font_family, "B", 8.5)
        pdf.set_text_color(*_C_NAVY)
        pdf.cell(30, 5, abbr, align="R")
        pdf.set_font(pdf.font_family, "", 8.5)
        pdf.set_text_color(*_C_TEXT2)
        pdf.cell(0, 5, f"  — {desc}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)

    pdf.set_font(pdf.font_family, "I", 8)
    pdf.set_text_color(*_C_GRAY)
    pdf.multi_cell(0, 4.5, (
        "Индикаторы калиброваны на основе: Stanford Internet Observatory "
        "(Nimmo, 2019), Meta CIB Reports (Gleicher, 2020), Botometer "
        "(Varol et al., 2017), GDELT Technical Documentation. "
        "Экспертная система — rule-based engine (Buchanan & Shortliffe, 1984)."
    ))


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

    # ── 1. Титульная страница ──
    _draw_title_page(pdf)

    # ── 2. Сводная таблица ──
    pdf.add_page()
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
    return pdf.output()
