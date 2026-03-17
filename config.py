"""
AI-OSINT — Конфигурация системы
=====================================
Палитра 和色, константы моделирования, пороги индикаторов,
настройки GDELT API, метаданные авторов.

Все цвета, параметры и текстовые метки — единый источник истины.
Остальные модули импортируют отсюда.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# 1. ЦВЕТОВАЯ ПАЛИТРА — 和色 (WASHOKU)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WashokuPalette:
    """Японская палитра 和色 — единый источник цветов для всего UI."""

    # --- Фоны ---
    bg_main: str = "#F8F8F4"        # 乳白色 Nyūhaku — молочно-белый
    bg_card: str = "#F6F1D3"        # 象牙色 Zōge — слоновая кость
    bg_dark: str = "#2B2B2B"        # 墨色 Sumi — чернильный

    # --- Синие (навигация, акценты) ---
    blue_indigo: str = "#004C71"    # 藍色 Ai-iro — индиго
    blue_ultra: str = "#1F4788"     # 群青色 Gunjō — ультрамарин
    blue_navy: str = "#223A70"      # 紺色 Kon — тёмно-синий

    # --- Красные (аномалия, опасность) ---
    red_crimson: str = "#C3272B"    # 赤紅 Aka-beni — алый
    red_rouge: str = "#9D2933"      # 臙脂色 Enji — тёмно-малиновый
    red_beni: str = "#B7282E"       # 紅 Kurenai — багряный

    # --- Зелёные (норма, успех) ---
    green_deep: str = "#316745"     # 千歳緑 Chitose-midori — вечнозелёный
    green_bamboo: str = "#6DAA2C"   # 竹色 Take — бамбуковый
    green_wasabi: str = "#7BA05B"   # 山葵色 Wasabi — васаби

    # --- Жёлтые / Оранжевые (внимание, предупреждение) ---
    yellow_kerria: str = "#FFB11B"  # 山吹色 Yamabuki — керрия
    yellow_sun: str = "#FFC800"     # 向日葵色 Himawari — подсолнух
    orange_kaki: str = "#C46243"    # 柿色 Kaki — хурма
    orange_daidai: str = "#F28500"  # 橙色 Daidai — мандарин

    # --- Фиолетовые ---
    purple_murasaki: str = "#6A0DAD"  # 紫 Murasaki
    purple_edo: str = "#77428D"       # 江戸紫 Edo-murasaki

    # --- Серые ---
    gray_nezumi: str = "#6D6D6D"    # 鼠色 Nezumi — мышиный серый
    gray_silver: str = "#B0B0B0"    # 銀鼠 Gin-nezumi — серебро

    # --- Текст ---
    text_primary: str = "#2B2B2B"   # 墨色 Sumi — основной текст
    text_secondary: str = "#6D6D6D" # 鼠色 Nezumi — вторичный текст

    # --- Светофорная система ---
    @property
    def traffic_green(self) -> str:
        return self.green_deep

    @property
    def traffic_yellow(self) -> str:
        return self.yellow_kerria

    @property
    def traffic_red(self) -> str:
        return self.red_crimson

    # --- Цвета агентов (4 типа) ---
    @property
    def agent_colors(self) -> Dict[str, str]:
        return {
            "initiator":  self.red_rouge,     # 臙脂色
            "amplifier":  self.orange_daidai,  # 橙色
            "mediator":   self.blue_ultra,     # 群青色
            "recipient":  self.gray_silver,    # 銀鼠
        }

    # --- Цвета состояний Маркова (5 фаз) ---
    @property
    def markov_colors(self) -> Dict[str, str]:
        return {
            "latent":    self.gray_nezumi,     # Латентная
            "emerging":  self.yellow_kerria,   # Зарождение
            "growing":   self.orange_kaki,     # Рост
            "viral":     self.red_crimson,     # Вирусная
            "declining": self.blue_indigo,     # Затухание
        }

    # --- Цвета для круговых диаграмм (языки / страны) ---
    @property
    def pie_sequence(self) -> List[str]:
        return [
            self.blue_indigo, self.red_beni, self.green_deep,
            self.yellow_kerria, self.orange_kaki, self.purple_edo,
            self.blue_ultra, self.gray_nezumi, self.green_wasabi,
            self.orange_daidai,
        ]


# Глобальный экземпляр палитры
COLORS = WashokuPalette()


# ═══════════════════════════════════════════════════════════════════
# 2. STREAMLIT CSS — светлая тема 和色
# ═══════════════════════════════════════════════════════════════════

CUSTOM_CSS = f"""
<style>
    /* --- Общий фон и шрифт --- */
    .stApp {{
        background-color: {COLORS.bg_main};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Noto Sans", "Helvetica Neue", Arial, sans-serif;
        color: {COLORS.text_primary};
    }}

    /* --- Боковая панель --- */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS.bg_dark};
    }}
    section[data-testid="stSidebar"] * {{
        color: {COLORS.bg_main} !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {{
        color: {COLORS.gray_silver} !important;
    }}

    /* --- Хедер (используется через st.markdown) --- */
    .aio-header {{
        background: linear-gradient(135deg, {COLORS.bg_dark} 0%, {COLORS.blue_navy} 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border-left: 5px solid {COLORS.blue_indigo};
    }}
    .aio-header h1 {{
        color: {COLORS.bg_main};
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: 0.5px;
    }}
    .aio-header p {{
        color: {COLORS.gray_silver};
        font-size: 0.85rem;
        margin: 0;
    }}

    /* --- Карточки метрик --- */
    .aio-card {{
        background: {COLORS.bg_card};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
    }}
    .aio-card h3 {{
        color: {COLORS.blue_navy};
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}
    .aio-card .value {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS.text_primary};
    }}

    /* --- Светофорные бейджи --- */
    .badge-green {{
        background: {COLORS.green_deep};
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .badge-yellow {{
        background: {COLORS.yellow_kerria};
        color: {COLORS.text_primary};
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .badge-red {{
        background: {COLORS.red_crimson};
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    /* --- Всплывающие пояснения --- */
    .aio-tooltip {{
        background: {COLORS.bg_dark};
        color: {COLORS.bg_main};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        font-size: 0.88rem;
        line-height: 1.5;
        border-left: 4px solid {COLORS.blue_indigo};
        margin: 0.5rem 0;
    }}
    .aio-tooltip .tooltip-title {{
        color: {COLORS.yellow_kerria};
        font-weight: 700;
        font-size: 0.92rem;
        margin-bottom: 0.4rem;
    }}

    /* --- Подвал --- */
    .aio-footer {{
        background: {COLORS.bg_dark};
        color: {COLORS.gray_silver};
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 2rem;
        font-size: 0.82rem;
    }}
    .aio-footer a {{
        color: {COLORS.blue_indigo};
        text-decoration: none;
    }}

    /* --- Вкладки --- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: {COLORS.bg_card};
        border-radius: 10px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 500;
        color: {COLORS.text_secondary};
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS.blue_navy} !important;
        color: white !important;
    }}

    /* --- Plotly-контейнеры (через st.plotly_chart) --- */
    .js-plotly-plot .plotly .main-svg {{
        background: transparent !important;
    }}

    /* --- Убрать дефолтный footer Streamlit --- */
    footer {{
        visibility: hidden;
    }}
</style>
"""


# ═══════════════════════════════════════════════════════════════════
# 3. ПАРАМЕТРЫ АОМ (АГЕНТНО-ОРИЕНТИРОВАННОЕ МОДЕЛИРОВАНИЕ)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AgentConfig:
    """Конфигурация типа агента."""
    key: str               # Системный ключ
    label_ru: str          # Название в интерфейсе
    share: float           # Доля в популяции (0–1)
    description: str       # Пояснение для нетехнаря
    activation_mod: float  # Модификатор вероятности активации


# Определения агентов — строго по спеку
AGENT_TYPES: Dict[str, AgentConfig] = {
    "initiator": AgentConfig(
        key="initiator",
        label_ru="Инициатор",
        share=0.05,
        description=(
            "Источник информационного воздействия. Создаёт оригинальный контент "
            "и запускает нарратив. Аналог: пресс-служба, государственное СМИ, "
            "координатор кампании."
        ),
        activation_mod=1.0,
    ),
    "amplifier": AgentConfig(
        key="amplifier",
        label_ru="Усилитель",
        share=0.20,
        description=(
            "Тиражирует контент инициатора: репосты, цитирование, "
            "адаптация под локальную аудиторию. Может быть ботом, "
            "наёмным аккаунтом или идеологически мотивированным актором."
        ),
        activation_mod=0.8,
    ),
    "mediator": AgentConfig(
        key="mediator",
        label_ru="Медиатор",
        share=0.10,
        description=(
            "Модерирует, верифицирует или опровергает нарратив. "
            "Аналог: фактчекер, независимое СМИ, аналитический центр. "
            "Замедляет переход нарратива к вирусной фазе."
        ),
        activation_mod=0.3,
    ),
    "recipient": AgentConfig(
        key="recipient",
        label_ru="Реципиент",
        share=0.65,
        description=(
            "Конечный потребитель информации. Принимает, игнорирует или "
            "перенаправляет контент. Составляет большинство сети."
        ),
        activation_mod=0.5,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# 4. СЦЕНАРИИ ИНФОРМАЦИОННЫХ КАМПАНИЙ
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScenarioConfig:
    """Конфигурация сценария."""
    key: str
    label_ru: str
    amplification: float    # Коэффициент усиления (0–1)
    description: str
    example: str            # Реальный пример для комиссии


SCENARIOS: Dict[str, ScenarioConfig] = {
    "organic": ScenarioConfig(
        key="organic",
        label_ru="Органический фон",
        amplification=0.05,
        description=(
            "Естественный информационный поток: журналисты пишут по факту, "
            "пользователи делятся по интересу. Нет координации."
        ),
        example=(
            "Реакция мировых СМИ на визит президента Казахстана в ЕС — "
            "разнородная тональность, отсутствие синхронных всплесков."
        ),
    ),
    "amplified": ScenarioConfig(
        key="amplified",
        label_ru="Усиленная кампания",
        amplification=0.35,
        description=(
            "Контент продвигается через сеть усилителей: таргетированные "
            "репосты, платное продвижение. Типичный PR / лоббизм."
        ),
        example=(
            "Продвижение инвестиционного имиджа Казахстана через "
            "платные публикации в Forbes, Bloomberg (Astana Hub, AIFC)."
        ),
    ),
    "coordinated": ScenarioConfig(
        key="coordinated",
        label_ru="Координированная кампания",
        amplification=0.60,
        description=(
            "Централизованное управление: единый нарратив, синхронные "
            "публикации, высокая текстовая гомогенность. Классическая ИО."
        ),
        example=(
            "Информационные операции вокруг событий января 2022: "
            "синхронные публикации в Telegram-каналах с идентичным текстом."
        ),
    ),
    "hybrid": ScenarioConfig(
        key="hybrid",
        label_ru="Гибридная кампания",
        amplification=0.25,
        description=(
            "Смешанный тип: часть контента органическая, часть — "
            "координированная. Труднее всего обнаружить."
        ),
        example=(
            "Смешение реальных протестных настроений с внешней "
            "координацией через бот-сети и анонимные Telegram-каналы."
        ),
    ),
}


# ═══════════════════════════════════════════════════════════════════
# 5. ЦЕПИ МАРКОВА — СОСТОЯНИЯ НАРРАТИВА
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MarkovState:
    """Состояние нарратива в цепи Маркова."""
    key: str
    label_ru: str
    description: str
    analogy: str     # Аналогия для нетехнаря (модель болезни)


MARKOV_STATES: Dict[str, MarkovState] = {
    "latent": MarkovState(
        key="latent",
        label_ru="Латентная фаза",
        description="Нарратив существует, но не распространяется. Единичные упоминания.",
        analogy="Как инкубационный период болезни: возбудитель присутствует, но симптомов нет.",
    ),
    "emerging": MarkovState(
        key="emerging",
        label_ru="Фаза зарождения",
        description="Нарратив начинает появляться в нескольких источниках. Рост заметен.",
        analogy="Первые симптомы: температура, недомогание. Ещё не ясно, что за болезнь.",
    ),
    "growing": MarkovState(
        key="growing",
        label_ru="Фаза роста",
        description="Активное распространение. Нарратив подхватывают крупные источники.",
        analogy="Болезнь в разгаре: симптомы нарастают, требуется вмешательство.",
    ),
    "viral": MarkovState(
        key="viral",
        label_ru="Вирусная фаза",
        description="Максимальный охват. Нарратив доминирует в медиапространстве.",
        analogy="Пик болезни: максимальная температура, критическое состояние.",
    ),
    "declining": MarkovState(
        key="declining",
        label_ru="Фаза затухания",
        description="Внимание снижается, новые нарративы вытесняют текущий.",
        analogy="Выздоровление: организм справился, температура падает.",
    ),
}

# Порядок состояний для матрицы переходов (индексы 0–4)
MARKOV_STATE_ORDER: List[str] = ["latent", "emerging", "growing", "viral", "declining"]

# Базовая матрица переходов 5×5
# Строки = текущее состояние, столбцы = следующее
# Сумма каждой строки = 1.0
MARKOV_BASE_MATRIX: np.ndarray = np.array([
    # LAT    EMG    GRW    VIR    DEC
    [0.70,  0.20,  0.05,  0.00,  0.05],  # Латентная
    [0.10,  0.50,  0.30,  0.05,  0.05],  # Зарождение
    [0.00,  0.10,  0.45,  0.35,  0.10],  # Рост
    [0.00,  0.00,  0.10,  0.50,  0.40],  # Вирусная
    [0.15,  0.05,  0.05,  0.00,  0.75],  # Затухание
], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
# 6. ИНДИКАТОРЫ ОБНАРУЖЕНИЯ
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IndicatorConfig:
    """Конфигурация индикатора обнаружения."""
    key: str
    label_ru: str
    formula: str
    thresholds: Tuple[float, float, float]  # (зелёный<, жёлтый<, красный≥)
    description: str
    methodology: str     # Откуда взяты пороги
    example_url: str     # Ссылка на реальный пример
    example_text: str    # Описание примера


INDICATORS: Dict[str, IndicatorConfig] = {
    "anomaly_index": IndicatorConfig(
        key="anomaly_index",
        label_ru="Индекс аномальности",
        formula="Z = (x − μ₃₀) / σ₃₀",
        thresholds=(1.5, 2.5, 3.5),
        description=(
            "Z-оценка текущего объёма публикаций относительно "
            "30-дневного скользящего среднего. Показывает, насколько "
            "текущий всплеск отклоняется от нормы."
        ),
        methodology=(
            "Пороги основаны на стандартных статистических конвенциях: "
            "Z>1.5 — умеренное отклонение, Z>2.5 — значимое (p<0.01), "
            "Z≥3.5 — экстремальное (p<0.001). Применяется в GDELT Anomaly Detection."
        ),
        example_url="https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/",
        example_text=(
            "Пример: во время событий января 2022 Z-оценка упоминаний "
            "Казахстана в мировых СМИ превысила 8.0 за сутки."
        ),
    ),
    "synchrony_coeff": IndicatorConfig(
        key="synchrony_coeff",
        label_ru="Коэффициент синхронности",
        formula="S = posts_in_window / total_posts",
        thresholds=(0.15, 0.35, 0.55),
        description=(
            "Доля публикаций, вышедших в узком временном окне (1–2 часа), "
            "от общего объёма за сутки. Высокая синхронность — признак "
            "координированной кампании."
        ),
        methodology=(
            "На основе исследований Stanford Internet Observatory: "
            "органические темы — S<0.15, PR-кампании — до 0.35, "
            "координированные ИО — 0.55+. (Nimmo, 2019; Gleicher, 2020)."
        ),
        example_url="https://cyber.fsi.stanford.edu/io",
        example_text=(
            "Пример: Meta CIB Report (2023) — удалённая сеть аккаунтов "
            "из Ирана: 78% постов за сутки выходили в окне 90 минут."
        ),
    ),
    "text_homogeneity": IndicatorConfig(
        key="text_homogeneity",
        label_ru="Текстовая гомогенность",
        formula="H = mean(cosine_similarity(TF-IDF))",
        thresholds=(0.4, 0.7, 0.85),
        description=(
            "Средняя косинусная близость TF-IDF-векторов публикаций "
            "одного кластера. Высокая однородность = копипаст / шаблоны."
        ),
        methodology=(
            "Порог 0.85 — уровень near-duplicate detection (Broder, 1997). "
            "Органический контент: 0.15–0.35, PR-кампании: 0.4–0.6, "
            "ботнеты: 0.85+ (IRA Troll Dataset, FiveThirtyEight, 2018)."
        ),
        example_url="https://github.com/fivethirtyeight/russian-troll-tweets/",
        example_text=(
            "Пример: в датасете IRA Trolls (538) — средняя косинусная "
            "близость в пределах одной операции = 0.89."
        ),
    ),
    "tonal_shift": IndicatorConfig(
        key="tonal_shift",
        label_ru="Тональный сдвиг",
        formula="|Δsentiment| = |tone_today − tone_yesterday|",
        thresholds=(0.2, 0.4, 0.7),
        description=(
            "Абсолютное изменение средней тональности публикаций за 24 часа. "
            "Резкий сдвиг без реального события — индикатор воздействия."
        ),
        methodology=(
            "GDELT Average Tone: шкала −10…+10. Нормальная суточная "
            "волатильность: ±0.1–0.2. Скачок 0.7+ за сутки без объективного "
            "повода — аномалия (GDELT Technical Documentation)."
        ),
        example_url="https://blog.gdeltproject.org/gdelt-doc-2-0-api-unveiled/",
        example_text=(
            "Пример: тональность публикаций о Казахстане упала с +1.2 "
            "до −3.8 за 12 часов в январе 2022."
        ),
    ),
    "spread_speed": IndicatorConfig(
        key="spread_speed",
        label_ru="Скорость распространения",
        formula="V = шагов АОМ до порога активации 50%",
        thresholds=(20.0, 10.0, 5.0),  # NB: обратная логика (>20 = зелёный)
        description=(
            "Количество шагов (суток) АОМ-симуляции, за которое "
            "нарратив достигает порога 50% активных агентов. "
            "Чем меньше шагов — тем агрессивнее кампания."
        ),
        methodology=(
            "Калибровка по ABM-литературе: органический нарратив — "
            "20+ суток до 50% охвата, координированная кампания — "
            "5 и менее (Axelrod, 1997; Epstein & Axtell, 1996)."
        ),
        example_url="https://about.fb.com/news/2021/05/influence-operations-threat-report/",
        example_text=(
            "Пример: координированные кампании, удалённые Meta в 2021–2023, "
            "достигали массового охвата за 2–4 суток."
        ),
    ),
    "bot_activity": IndicatorConfig(
        key="bot_activity",
        label_ru="Бот-активность",
        formula="B = f(CV_timing, TTR, repost_ratio)",
        thresholds=(0.1, 0.3, 0.6),
        description=(
            "Композитный индикатор ботоподобного поведения: "
            "низкая вариативность интервалов между постами (CV<0.1), "
            "низкое лексическое разнообразие (TTR<0.3), "
            "доля репостов >90%."
        ),
        methodology=(
            "Критерии адаптированы из Indiana University BotOrNot / "
            "Botometer (Varol et al., 2017) и Digital Forensic Research "
            "Lab (DFRLab) — методология Atlantic Council."
        ),
        example_url="https://botometer.osome.iu.edu/",
        example_text=(
            "Пример: бот-ферма, обнаруженная DFRLab (2022): "
            "250 аккаунтов, CV=0.04, TTR=0.18, 96% — ретвиты."
        ),
    ),
}

# Агрегированный Индекс Угрозы — пороги
THREAT_INDEX_THRESHOLDS: Tuple[int, int, int] = (30, 60, 100)  # 🟢0-30 🟡31-60 🔴61-100


# ═══════════════════════════════════════════════════════════════════
# 7. НАСТРОЙКИ GDELT API
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GDELTConfig:
    """Параметры подключения к GDELT."""
    doc_api_base: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    timeline_api_base: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    query_term: str = "Kazakhstan"

    # 7 языков анализа: 6 ООН + казахский
    # Коды языков GDELT (sourcelang)
    languages: Dict[str, str] = field(default_factory=lambda: {
        "english":  "Английский",
        "russian":  "Русский",
        "chinese":  "Китайский",
        "french":   "Французский",
        "spanish":  "Испанский",
        "arabic":   "Арабский",
        "kazakh":   "Казахский",
    })

    # GDELT language codes для фильтрации
    lang_codes: Dict[str, str] = field(default_factory=lambda: {
        "english":  "eng",
        "russian":  "rus",
        "chinese":  "zho",
        "french":   "fra",
        "spanish":  "spa",
        "arabic":   "ara",
        "kazakh":   "kaz",
    })

    rate_limit_seconds: float = 5.0   # 1 запрос / 5 сек
    cache_ttl_seconds: int = 3600     # Кэш на 1 час
    max_records: int = 250            # Макс. записей за запрос
    timeout_seconds: int = 30         # Таймаут соединения

    # Период по умолчанию
    default_days_back: int = 90       # 3 месяца


GDELT = GDELTConfig()


# ═══════════════════════════════════════════════════════════════════
# 8. МАРКИРОВКА НАДЁЖНОСТИ ИСТОЧНИКОВ
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SourceReliability:
    """Степень надёжности источника (НЕ 'фейк/не фейк')."""
    key: str
    label_ru: str
    color: str
    description: str


SOURCE_RELIABILITY_LEVELS: Dict[str, SourceReliability] = {
    "verified": SourceReliability(
        key="verified",
        label_ru="Проверенное СМИ",
        color=COLORS.green_deep,
        description="Издание с устоявшейся редакционной политикой и фактчекингом.",
    ),
    "state": SourceReliability(
        key="state",
        label_ru="Государственное СМИ",
        color=COLORS.blue_ultra,
        description=(
            "СМИ, финансируемое или контролируемое государственными структурами. "
            "Требует перекрёстной проверки."
        ),
    ),
    "tabloid": SourceReliability(
        key="tabloid",
        label_ru="Жёлтая пресса",
        color=COLORS.yellow_kerria,
        description="Издание с низким уровнем фактчекинга и сенсационным стилем.",
    ),
    "unknown": SourceReliability(
        key="unknown",
        label_ru="Неизвестный источник",
        color=COLORS.gray_nezumi,
        description="Источник не идентифицирован или отсутствует в базах надёжности.",
    ),
}


# ═══════════════════════════════════════════════════════════════════
# 9. ПАРАМЕТРЫ АОМ-СЕТИ (БАРАБАШИ-АЛЬБЕРТ)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NetworkConfig:
    """Параметры графа Барабаши-Альберт."""
    default_n_agents: int = 200         # Агентов по умолчанию
    min_agents: int = 50                # Минимум (слайдер)
    max_agents: int = 1000              # Максимум (слайдер)
    ba_m: int = 3                       # Число рёбер для нового узла
    activation_threshold: float = 0.5   # Порог для «активирован»


NETWORK = NetworkConfig()


# ═══════════════════════════════════════════════════════════════════
# 10. ПАРАМЕТРЫ МОНТЕ-КАРЛО
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MonteCarloConfig:
    """Параметры стохастического моделирования."""
    default_n_simulations: int = 500
    min_simulations: int = 100
    max_simulations: int = 5000
    step_simulations: int = 100
    default_n_steps: int = 60           # Суток моделирования
    min_steps: int = 10
    max_steps: int = 180
    matrix_noise: float = 0.05          # ±5% шум матрицы Маркова
    confidence_level: float = 0.95      # Доверительный интервал 95%


MONTE_CARLO = MonteCarloConfig()


# ═══════════════════════════════════════════════════════════════════
# 11. UI-ТЕКСТЫ (ВСЁ НА РУССКОМ)
# ═══════════════════════════════════════════════════════════════════

TAB_NAMES: List[str] = [
    "Панель мониторинга",
    "АОМ-симуляция",
    "Цепи Маркова",
    "Монте-Карло",
    "Индикаторы обнаружения",
    "Библиотека результатов",
    "Генерация отчёта",
    "О системе",
]

APP_TITLE: str = "AI-OSINT"
APP_SUBTITLE: str = (
    "Цифровой анализ информационного поля Казахстана "
    "в глобальном медиапространстве"
)
APP_METHOD: str = "Вычислительное имитационное моделирование"

SIDEBAR_TITLE: str = "Параметры моделирования"


# ═══════════════════════════════════════════════════════════════════
# 12. МЕТАДАННЫЕ — АВТОРЫ, ВУЗ
# ═══════════════════════════════════════════════════════════════════

AUTHORS: List[Dict[str, str]] = [
    {
        "name": "Абсаттаров Г.Р.",
        "role": "Научный руководитель",
        "title": "к.полит.н., ассоц. проф.",
    },
    {
        "name": "Саятбек С.",
        "role": "Разработчик",
        "title": "докторант",
    },
    {
        "name": "Кайролла А.Б.",
        "role": "Разработчик",
        "title": "магистрант",
    },
]

UNIVERSITY: str = "КазУМОиМЯ им. Абылай хана"
YEAR: int = 2026
COPYRIGHT: str = f"© {YEAR} {UNIVERSITY}. Все права защищены."


# ═══════════════════════════════════════════════════════════════════
# 13. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════

def traffic_light(value: float, thresholds: Tuple[float, float, float],
                  inverse: bool = False) -> str:
    """
    Возвращает CSS-класс светофора по значению и порогам.

    Параметры:
        value: текущее значение индикатора
        thresholds: (зелёный<, жёлтый<, красный≥)
        inverse: True для индикаторов, где меньше = хуже (скорость распространения)

    Возвращает:
        'badge-green', 'badge-yellow' или 'badge-red'
    """
    g, y, r = thresholds
    if inverse:
        # Обратная логика: больше = лучше (напр. скорость распространения)
        if value > g:
            return "badge-green"
        elif value > y:
            return "badge-yellow"
        else:
            return "badge-red"
    else:
        # Прямая логика: меньше = лучше
        if value < g:
            return "badge-green"
        elif value < y:
            return "badge-yellow"
        else:
            return "badge-red"


def traffic_emoji(value: float, thresholds: Tuple[float, float, float],
                  inverse: bool = False) -> str:
    """Возвращает эмодзи светофора: 🟢, 🟡 или 🔴."""
    cls = traffic_light(value, thresholds, inverse)
    return {"badge-green": "🟢", "badge-yellow": "🟡", "badge-red": "🔴"}[cls]


def threat_index_level(index: float) -> Tuple[str, str, str]:
    """
    Классификация Индекса Угрозы (0–100).

    Возвращает: (emoji, css_class, label_ru)
    """
    if index <= THREAT_INDEX_THRESHOLDS[0]:
        return ("🟢", "badge-green", "Норма")
    elif index <= THREAT_INDEX_THRESHOLDS[1]:
        return ("🟡", "badge-yellow", "Повышенное внимание")
    else:
        return ("🔴", "badge-red", "Высокая угроза")


def render_header() -> str:
    """HTML-блок хедера."""
    return f"""
    <div class="aio-header">
        <h1>🛡️ {APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
        <p style="margin-top: 4px; font-size: 0.78rem; color: {COLORS.gray_silver};">
            {APP_METHOD} &nbsp;|&nbsp; {UNIVERSITY} &nbsp;|&nbsp; {YEAR}
        </p>
    </div>
    """


def render_footer() -> str:
    """HTML-блок подвала."""
    authors_str = " &nbsp;|&nbsp; ".join(
        f"{a['name']}, {a['title']}" for a in AUTHORS
    )
    return f"""
    <div class="aio-footer">
        <div style="margin-bottom: 0.4rem;">{authors_str}</div>
        <div>{COPYRIGHT}</div>
    </div>
    """


def render_tooltip(title: str, text: str) -> str:
    """HTML-блок всплывающего пояснения."""
    return f"""
    <div class="aio-tooltip">
        <div class="tooltip-title">{title}</div>
        <div>{text}</div>
    </div>
    """


def render_metric_card(title: str, value: str, subtitle: str = "") -> str:
    """HTML-блок карточки метрики."""
    sub_html = f'<div style="color: {COLORS.text_secondary}; font-size: 0.78rem; margin-top: 0.3rem;">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="aio-card">
        <h3>{title}</h3>
        <div class="value">{value}</div>
        {sub_html}
    </div>
    """
