"""
AI-OSINT — Главный интерфейс
====================================
Модуль 4: app.py — интерфейс Streamlit, вкладки, пояснения,
визуализации, боковая панель, «О системе».

Собирает все модули:
    config.py       → палитра, константы, UI-тексты
    data_engine.py  → GDELT API, живые данные
    abm_engine.py   → АОМ + Марков + Монте-Карло
    indicators.py   → 6 индикаторов + Индекс угрозы + рекомендации
    library.py      → Библиотека результатов
    report_gen.py   → Генерация PDF-отчёта

Авторы:
    Абсаттаров Г.Р., к.полит.н., ассоц. проф.
    Саятбек С., Кайролла А.Б.
    КазУМОиМЯ им. Абылай хана | 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── Внутренние модули ──
from config import (
    COLORS, CUSTOM_CSS, AGENT_TYPES, SCENARIOS, MARKOV_STATES,
    MARKOV_STATE_ORDER, MARKOV_BASE_MATRIX, INDICATORS,
    THREAT_INDEX_THRESHOLDS, NETWORK, MONTE_CARLO, GDELT,
    TAB_NAMES, APP_TITLE, APP_SUBTITLE, APP_METHOD,
    SIDEBAR_TITLE, AUTHORS, UNIVERSITY, YEAR, COPYRIGHT,
    traffic_emoji, threat_index_level,
    render_header, render_footer, render_tooltip, render_metric_card,
)
from data_engine import (
    DataSnapshot, load_data, compute_moving_average,
    generate_digest, reliability_badge_html,
    get_topic_distribution,
)
from abm_engine import (
    run_full_simulation, run_monte_carlo, MonteCarloResult,
)
from indicators import (
    IndicatorEngine, DetectionResult,
    render_indicator_card, render_indicator_detail,
    render_threat_gauge, render_recommendations,
    detection_to_dict, detection_from_dict,
)
from library import (
    ResultsLibrary, create_record, SimulationRecord,
    render_library_card, render_library_empty, render_library_stats,
)
from report_gen import generate_report


# ═══════════════════════════════════════════════════════════════════
# 0. НАСТРОЙКА СТРАНИЦЫ
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=f"{APP_TITLE} — {APP_SUBTITLE}",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Инъекция CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Хедер
st.markdown(render_header(), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 1. BRIDGE-ФУНКЦИЯ: DataSnapshot → gdelt_data dict (для indicators)
# ═══════════════════════════════════════════════════════════════════

def snapshot_to_gdelt_data(snap: DataSnapshot) -> dict:
    """
    Заметка из чата 3 (стыковка data_engine → indicators).

    Переводит DataSnapshot из data_engine.py в словарь gdelt_data,
    который ожидает IndicatorEngine.compute_all().

    Индикаторы TF-IDF similarity, CV timing, TTR — в data_engine нет
    (GDELT не отдаёт полные тексты). Эти 3 индикатора автоматически
    переключаются на АОМ-прокси. Это штатный режим.
    """
    d: dict = {}

    # Индикатор 1: Индекс аномальности — volumes_30d + current_volume
    if not snap.volume_df.empty:
        d["volumes_30d"] = snap.volume_df["volume"].values
        d["current_volume"] = snap.anomaly.get("current_volume", 0)

    # Индикатор 4: Тональный сдвиг — tone_today + tone_yesterday
    if not snap.tone_df.empty and len(snap.tone_df) >= 2:
        d["tone_today"] = float(snap.tone_df["tone"].iloc[-1])
        d["tone_yesterday"] = float(snap.tone_df["tone"].iloc[-2])

    # Индикатор 2: Коэффициент синхронности — timestamps_hours
    if not snap.articles_df.empty:
        dates = pd.to_datetime(
            snap.articles_df["date"], errors="coerce", utc=True,
        )
        d["timestamps_hours"] = (
            dates.dt.hour.values + dates.dt.minute.values / 60.0
        )

    # Индикаторы 3, 6: TF-IDF, CV, TTR — НЕТ в GDELT
    # → IndicatorEngine автоматически переключится на ABM-прокси

    return d


# ═══════════════════════════════════════════════════════════════════
# 2. ИНИЦИАЛИЗАЦИЯ SESSION_STATE
# ═══════════════════════════════════════════════════════════════════

def _init_state():
    """Инициализация всех ключей session_state."""
    defaults = {
        "data_loaded": False,
        "snapshot": None,
        "sim_result": None,
        "mc_result": None,
        "detection": None,
        "last_scenario": "organic",
        "last_n_agents": NETWORK.default_n_agents,
        "last_n_steps": 60,
        "last_seed": 42,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ═══════════════════════════════════════════════════════════════════
# 3. БОКОВАЯ ПАНЕЛЬ — ПАРАМЕТРЫ
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{COLORS.bg_main};margin-top:0;'>"
        f"⚙️ {SIDEBAR_TITLE}</h2>",
        unsafe_allow_html=True,
    )

    # ── Сценарий ──
    scenario_options = {v["label_ru"]: k for k, v in SCENARIOS.items()}
    selected_label = st.selectbox(
        "Сценарий информационной кампании",
        options=list(scenario_options.keys()),
        index=0,
        help="Определяет долю координированных агентов в модели.",
    )
    scenario_key = scenario_options[selected_label]

    # Пояснение сценария
    scn = SCENARIOS[scenario_key]
    st.caption(scn["description"])
    st.caption(f"💡 *Пример: {scn['example']}*")

    st.divider()

    # ── Параметры модели ──
    n_agents = st.slider(
        "Количество агентов",
        min_value=NETWORK.min_agents,
        max_value=NETWORK.max_agents,
        value=NETWORK.default_n_agents,
        step=50,
        help="Размер информационной сети (узлы графа Барабаши–Альберт).",
    )

    n_steps = st.slider(
        "Горизонт моделирования (суток)",
        min_value=MONTE_CARLO.min_steps,
        max_value=MONTE_CARLO.max_steps,
        value=60,
        step=5,
        help="1 шаг = 1 сутки.",
    )

    seed = st.number_input(
        "Seed (воспроизводимость)",
        min_value=0,
        max_value=99999,
        value=42,
        help="Фиксирует случайную генерацию для воспроизводимости.",
    )

    st.divider()

    # ── Монте-Карло ──
    mc_simulations = st.slider(
        "Монте-Карло: итерации",
        min_value=MONTE_CARLO.min_simulations,
        max_value=MONTE_CARLO.max_simulations,
        value=MONTE_CARLO.default_n_simulations,
        step=MONTE_CARLO.step_simulations,
        help="Число стохастических прогонов для оценки P(вирусная фаза).",
    )

    st.divider()

    # ── Глубина данных GDELT ──
    days_back = st.slider(
        "Глубина данных GDELT (дни)",
        min_value=7,
        max_value=180,
        value=GDELT.default_days_back,
        step=7,
        help="Период выгрузки из GDELT API.",
    )

    st.divider()

    # ── Кнопки запуска ──
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        btn_load = st.button("📡 Загрузить данные", use_container_width=True)
    with col_b2:
        btn_sim = st.button("▶️ Запуск модели", use_container_width=True)

    btn_mc = st.button(
        f"🎲 Монте-Карло ({mc_simulations} итер.)",
        use_container_width=True,
    )

    # ── Статус подключения ──
    if st.session_state.snapshot is not None:
        snap = st.session_state.snapshot
        if snap.is_live:
            st.success(f"🟢 Живые данные GDELT | {snap.timestamp}")
        else:
            st.warning(f"🟡 Демо-режим | {snap.timestamp}")
            if snap.error_msg:
                st.caption(f"Причина: {snap.error_msg}")


# ═══════════════════════════════════════════════════════════════════
# 4. ОБРАБОТКА КНОПОК
# ═══════════════════════════════════════════════════════════════════

# ── Загрузка данных ──
if btn_load:
    with st.spinner("Загрузка данных из GDELT API..."):
        snap = load_data(query=GDELT.query_term, days_back=days_back)
        st.session_state.snapshot = snap
        st.session_state.data_loaded = True
    st.rerun()

# ── Запуск ABM + Markov ──
if btn_sim:
    # Автозагрузка данных если ещё не загружены
    if not st.session_state.data_loaded:
        with st.spinner("Загрузка данных из GDELT API..."):
            snap = load_data(query=GDELT.query_term, days_back=days_back)
            st.session_state.snapshot = snap
            st.session_state.data_loaded = True

    with st.spinner("Запуск АОМ-симуляции..."):
        sim_result = run_full_simulation(
            n_agents=n_agents,
            n_steps=n_steps,
            scenario_key=scenario_key,
            seed=seed,
        )
        st.session_state.sim_result = sim_result
        st.session_state.last_scenario = scenario_key
        st.session_state.last_n_agents = n_agents
        st.session_state.last_n_steps = n_steps
        st.session_state.last_seed = seed

    # Вычисление индикаторов
    with st.spinner("Вычисление индикаторов обнаружения..."):
        gdelt_data = snapshot_to_gdelt_data(st.session_state.snapshot)
        engine = IndicatorEngine()
        detection = engine.compute_all(
            gdelt_data=gdelt_data,
            sim_result=sim_result,
        )
        st.session_state.detection = detection

    st.rerun()

# ── Монте-Карло ──
if btn_mc:
    # Автозагрузка + автосимуляция если не было
    if not st.session_state.data_loaded:
        with st.spinner("Загрузка данных из GDELT API..."):
            snap = load_data(query=GDELT.query_term, days_back=days_back)
            st.session_state.snapshot = snap
            st.session_state.data_loaded = True

    if st.session_state.sim_result is None:
        with st.spinner("Запуск АОМ-симуляции..."):
            sim_result = run_full_simulation(
                n_agents=n_agents, n_steps=n_steps,
                scenario_key=scenario_key, seed=seed,
            )
            st.session_state.sim_result = sim_result
            st.session_state.last_scenario = scenario_key
            st.session_state.last_n_agents = n_agents
            st.session_state.last_n_steps = n_steps
            st.session_state.last_seed = seed

    with st.spinner(f"Монте-Карло: {mc_simulations} итераций..."):
        mc_result = run_monte_carlo(
            n_agents=n_agents,
            n_simulations=mc_simulations,
            n_steps=n_steps,
            scenario_key=scenario_key,
            base_seed=seed,
        )
        st.session_state.mc_result = mc_result

    # Пересчёт индикаторов с МК-данными
    with st.spinner("Обновление индикаторов..."):
        gdelt_data = snapshot_to_gdelt_data(st.session_state.snapshot)
        engine = IndicatorEngine()
        detection = engine.compute_all(
            gdelt_data=gdelt_data,
            sim_result=st.session_state.sim_result,
            mc_result=mc_result,
        )
        st.session_state.detection = detection

    st.rerun()


# ═══════════════════════════════════════════════════════════════════
# 5. УТИЛИТЫ ВИЗУАЛИЗАЦИИ (Plotly в стиле 和色)
# ═══════════════════════════════════════════════════════════════════

def _plotly_layout(**kwargs) -> dict:
    """Базовый layout Plotly в стиле 和色."""
    base = dict(
        font=dict(family="sans-serif", color=COLORS.text_primary),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS.bg_main,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
    )
    base.update(kwargs)
    return base


def _show_chart(fig, key=None):
    """Рендер Plotly-графика."""
    st.plotly_chart(fig, use_container_width=True, key=key)


# ═══════════════════════════════════════════════════════════════════
# 6. ВКЛАДКИ
# ═══════════════════════════════════════════════════════════════════

tabs = st.tabs(TAB_NAMES)


# ─────────────────────────────────────────────────────────────────
# ВКЛАДКА 1: ПАНЕЛЬ МОНИТОРИНГА
# ─────────────────────────────────────────────────────────────────
with tabs[0]:
    if not st.session_state.data_loaded:
        st.info(
            "Нажмите **📡 Загрузить данные** в боковой панели "
            "для подключения к GDELT API."
        )
    else:
        snap: DataSnapshot = st.session_state.snapshot

        # Режим данных
        if not snap.is_live:
            st.warning(
                "⚠️ **Демо-режим**: GDELT API недоступен. "
                "Отображаются демонстрационные данные для иллюстрации "
                "работы платформы."
            )

        # ── Карточки метрик ──
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            # Последняя ненулевая точка timeline
            vol_val = 0
            vol_label = "Последние полные сутки"
            if not snap.volume_df.empty:
                nonzero = snap.volume_df[snap.volume_df["volume"] > 0]
                if not nonzero.empty:
                    vol_val = int(nonzero["volume"].iloc[-1])

            # Fallback: если timeline пустой, но статьи есть —
            # считаем количество статей за вчера из articles_df
            if vol_val == 0 and not snap.articles_df.empty and "date" in snap.articles_df.columns:
                try:
                    from datetime import datetime, timedelta
                    yesterday = datetime.utcnow().date() - timedelta(days=1)
                    yesterday_articles = snap.articles_df[
                        snap.articles_df["date"].dt.date == yesterday
                    ]
                    if not yesterday_articles.empty:
                        vol_val = len(yesterday_articles)
                        vol_label = "За вчера (из статей)"
                    else:
                        # Все статьи за весь период
                        vol_val = len(snap.articles_df)
                        vol_label = f"Всего за {days_back} дней"
                except Exception:
                    vol_val = len(snap.articles_df)
                    vol_label = f"Всего за {days_back} дней"

            st.markdown(
                render_metric_card(
                    "Публикации о Казахстане",
                    f"{int(vol_val):,}".replace(",", " "),
                    f"{vol_label} | MA30: {snap.anomaly.get('ma30', 0):.0f}",
                ),
                unsafe_allow_html=True,
            )
        with m2:
            z_val = snap.anomaly.get("z_score", 0)
            emoji_z = traffic_emoji(z_val, (1.5, 2.5, 3.5))
            st.markdown(
                render_metric_card(
                    "Z-оценка аномальности",
                    f"{emoji_z} {z_val:.2f}",
                ),
                unsafe_allow_html=True,
            )
        with m3:
            tone_last = (
                float(snap.tone_df["tone"].iloc[-1])
                if not snap.tone_df.empty else 0.0
            )
            st.markdown(
                render_metric_card(
                    "Тональность (последний день)",
                    f"{tone_last:+.2f}",
                    "Шкала GDELT: −10 … +10",
                ),
                unsafe_allow_html=True,
            )
        with m4:
            n_articles = len(snap.articles_df) if not snap.articles_df.empty else 0
            st.markdown(
                render_metric_card(
                    "Статей из GDELT",
                    f"{n_articles}",
                    f"По запросу «Kazakhstan» за {days_back} дней, 7 языков",
                ),
                unsafe_allow_html=True,
            )

        # ── Частота упоминаний (временной ряд + MA30) ──
        st.subheader("Частота упоминаний Казахстана")
        with st.expander("ℹ️ Что это?"):
            st.markdown(render_tooltip(
                "Частота упоминаний",
                "Количество публикаций в мировых СМИ, содержащих "
                "ключевое слово «Kazakhstan», по данным GDELT. "
                "Оранжевая линия — скользящее среднее за 30 дней (MA30). "
                "Всплески выше MA30 могут указывать на информационные события "
                "или координированные кампании.",
            ), unsafe_allow_html=True)

        if not snap.volume_df.empty:
            vdf = snap.volume_df.copy()
            vdf["ma30"] = compute_moving_average(vdf["volume"], window=30)

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vdf["date"], y=vdf["volume"],
                name="Объём (сутки)",
                line=dict(color=COLORS.blue_indigo, width=1.5),
                fill="tozeroy",
                fillcolor="rgba(0,76,113,0.08)",
            ))
            fig_vol.add_trace(go.Scatter(
                x=vdf["date"], y=vdf["ma30"],
                name="MA30",
                line=dict(color=COLORS.orange_daidai, width=2.5, dash="dot"),
            ))
            fig_vol.update_layout(**_plotly_layout(
                title="Публикации в сутки",
                yaxis_title="Количество",
                legend=dict(orientation="h", y=-0.15),
            ))
            _show_chart(fig_vol, key="vol_chart")

        # ── Тональность (временной ряд + MA30) ──
        st.subheader("Тональность упоминаний")
        with st.expander("ℹ️ Что это?"):
            st.markdown(render_tooltip(
                "Тональность (Average Tone)",
                "Средний тон публикаций мировых СМИ, упоминающих Казахстан. "
                "Шкала GDELT: от −10 (крайне негативный) до +10 (крайне "
                "позитивный). Резкие сдвиги без объективного повода могут "
                "указывать на информационное воздействие.",
            ), unsafe_allow_html=True)

        if not snap.tone_df.empty:
            tdf = snap.tone_df.copy()
            tdf["ma30"] = compute_moving_average(tdf["tone"], window=30)

            fig_tone = go.Figure()
            fig_tone.add_trace(go.Scatter(
                x=tdf["date"], y=tdf["tone"],
                name="Тональность (сутки)",
                line=dict(color=COLORS.blue_navy, width=1.5),
            ))
            fig_tone.add_trace(go.Scatter(
                x=tdf["date"], y=tdf["ma30"],
                name="MA30",
                line=dict(color=COLORS.orange_daidai, width=2.5, dash="dot"),
            ))
            # Нулевая линия
            fig_tone.add_hline(
                y=0, line_dash="dash",
                line_color=COLORS.gray_nezumi,
                annotation_text="Нейтральная тональность",
            )
            fig_tone.update_layout(**_plotly_layout(
                title="Средняя тональность (GDELT Average Tone)",
                yaxis_title="Тон (−10 … +10)",
                legend=dict(orientation="h", y=-0.15),
            ))
            _show_chart(fig_tone, key="tone_chart")

        # ── Источники по странам + Языковое распределение ──
        col_pie1, col_pie2 = st.columns(2)

        with col_pie1:
            st.subheader("Источники по странам")
            with st.expander("ℹ️ Что это?"):
                st.markdown(render_tooltip(
                    "Источники по странам",
                    "GDELT атрибутирует источник по домену. Доля (%) — "
                    "от общего потока по запросу «Kazakhstan».",
                ), unsafe_allow_html=True)

            if not snap.countries_df.empty:
                cdf = snap.countries_df.head(10)
                fig_c = go.Figure(go.Pie(
                    labels=cdf["country"], values=cdf["count"],
                    hole=0.45,
                    marker=dict(colors=COLORS.pie_sequence[:len(cdf)]),
                    textinfo="label+percent",
                    textfont=dict(size=11),
                ))
                fig_c.update_layout(**_plotly_layout(
                    title="Топ-10 стран-источников",
                    showlegend=False,
                ))
                _show_chart(fig_c, key="country_pie")

        with col_pie2:
            st.subheader("Языковое распределение")
            with st.expander("ℹ️ Что это?"):
                st.markdown(render_tooltip(
                    "Языковое распределение",
                    "Распределение публикаций по 7 целевым языкам "
                    "(6 ООН + казахский). Язык определяется GDELT "
                    "автоматически (точность ~90–95%). Агрегация по "
                    "всем загруженным статьям.",
                ), unsafe_allow_html=True)

            if not snap.languages_df.empty:
                ldf = snap.languages_df
                fig_l = go.Figure(go.Pie(
                    labels=ldf["lang_label"], values=ldf["count"],
                    hole=0.45,
                    marker=dict(colors=COLORS.pie_sequence[:len(ldf)]),
                    textinfo="label+percent",
                    textfont=dict(size=11),
                ))
                fig_l.update_layout(**_plotly_layout(
                    title="7 языков анализа",
                    showlegend=False,
                ))
                _show_chart(fig_l, key="lang_pie")

        # ── Тематический мониторинг (классификация Санжара) ──
        st.subheader("Тематический мониторинг")
        with st.expander("ℹ️ Что это?"):
            st.markdown(render_tooltip(
                "Тематическая классификация",
                "Автоматическая категоризация загруженных статей по "
                "чувствительным темам казахстанского информационного "
                "пространства: межэтнические отношения, языковой вопрос, "
                "региональный раскол, протестные настроения, геополитическое "
                "давление, безопасность, репутация, энергетика.",
            ), unsafe_allow_html=True)

        if not snap.topics_df.empty:
            fig_topics = go.Figure(go.Bar(
                x=snap.topics_df["topic_label"],
                y=snap.topics_df["count"],
                marker_color=snap.topics_df["topic_color"],
                text=[f"{s}%" for s in snap.topics_df["share"]],
                textposition="outside",
            ))
            fig_topics.update_layout(**_plotly_layout(
                title="Распределение по тематическим категориям",
                yaxis_title="Количество статей",
                height=400,
            ))
            _show_chart(fig_topics, key="topics_bar")
        else:
            st.caption("Нет данных для тематической классификации")

        # ── Дайджест ──
        st.subheader("Дайджест")
        st.caption(
            "Топ-10 публикаций с максимальной тональностью "
            "(позитивной или негативной) за весь загруженный период. "
            "Если тональность не определена — показаны самые свежие статьи. "
            "Шкала GDELT: от −10 до +10."
        )

        if not snap.articles_df.empty:
            # Берём топ-10 по |тональности| из всего периода
            digest_df = snap.articles_df.copy()
            scored = digest_df[digest_df["tone"].abs() > 0.05]

            if not scored.empty:
                # Есть статьи с тональностью — сортируем по |tone|
                scored["abs_tone"] = scored["tone"].abs()
                digest_df = scored.sort_values("abs_tone", ascending=False).head(10)
                digest_df = digest_df.drop(columns=["abs_tone"])
            else:
                # Все tone=0 — показываем 10 самых свежих по дате
                digest_df = digest_df.head(10)

            if not digest_df.empty:
                for _, row in digest_df.iterrows():
                    tone_v = row.get("tone", 0)
                    color = COLORS.red_crimson if tone_v < -1 else (
                        COLORS.green_deep if tone_v > 1 else COLORS.gray_nezumi
                    )
                    rel_badge = reliability_badge_html(row.get("reliability", "unknown"))
                    topic_lbl = row.get("topic_label", "")
                    topic_clr = row.get("topic_color", COLORS.gray_nezumi)
                    topic_html = (
                        f' <span style="background:{topic_clr};color:#202124;'
                        f'padding:1px 8px;border-radius:10px;font-size:0.68rem;'
                        f'font-weight:600;">{topic_lbl}</span>'
                        if topic_lbl and topic_lbl != "Прочее" else ""
                    )
                    # Дата публикации
                    date_str = ""
                    if "date" in row and pd.notna(row["date"]):
                        try:
                            date_str = f' | {pd.to_datetime(row["date"]).strftime("%d.%m.%Y")}'
                        except Exception:
                            pass
                    st.markdown(
                        f'<div class="aio-card" style="padding:0.7rem 1rem;">'
                        f'<div style="display:flex;justify-content:space-between;">'
                        f'<div style="flex:1;min-width:0;">'
                        f'<a href="{row.get("url","#")}" target="_blank" '
                        f'style="color:{COLORS.blue_indigo};font-size:0.88rem;'
                        f'text-decoration:none;font-weight:500;">'
                        f'{row.get("title","—")}</a>'
                        f'<div style="font-size:0.75rem;color:{COLORS.text_secondary};'
                        f'margin-top:3px;">'
                        f'{row.get("source_domain","")}{date_str} | '
                        f'{rel_badge}{topic_html}</div></div>'
                        f'<div style="text-align:right;min-width:60px;">'
                        f'<span style="color:{color};font-weight:700;'
                        f'font-size:1rem;">{tone_v:+.1f}</span></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Нет статей с ненулевой тональностью.")


# ─────────────────────────────────────────────────────────────────
# ВКЛАДКА 2: АОМ-СИМУЛЯЦИЯ
# ─────────────────────────────────────────────────────────────────
with tabs[1]:
    if st.session_state.sim_result is None:
        st.info(
            "Нажмите **▶️ Запуск модели** в боковой панели для "
            "запуска агентно-ориентированной модели."
        )
    else:
        sim = st.session_state.sim_result
        abm = sim["abm"]
        history = sim["history"]
        graph_data = sim["graph_data"]
        timeline = sim["activation_timeline"]

        # ── Пояснение ──
        with st.expander("ℹ️ Что такое АОМ?"):
            st.markdown(render_tooltip(
                "Агентно-ориентированное моделирование (АОМ)",
                "Метод вычислительного моделирования, в котором автономные "
                "агенты (информационные акторы) взаимодействуют в сети. "
                "Сеть построена по модели Барабаши–Альберт (масштабно-свободная "
                "сеть с hub-узлами). Каждый агент — один из 4 типов. "
                "1 шаг симуляции = 1 сутки."
                "<br><br>"
                "<strong>Формула активации:</strong> "
                "P = P_state × S_agent × (1 + N_active / N_neighbors)"
                "<br><br>"
                "<strong>4 типа агентов:</strong><br>"
                "• Инициатор (5%) — генерирует нарратив<br>"
                "• Усилитель (20%) — тиражирует (боты, репосты)<br>"
                "• Медиатор (10%) — проверяет, легитимизирует<br>"
                "• Реципиент (65%) — конечный потребитель",
            ), unsafe_allow_html=True)

        # ── Метрики ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(render_metric_card(
                "Агентов в сети", str(abm.n_agents),
            ), unsafe_allow_html=True)
        with c2:
            ar = abm.get_active_ratio()
            st.markdown(render_metric_card(
                "Активных", f"{ar:.1%}",
                f"{sum(1 for a in abm.agents if a.active)} из {abm.n_agents}",
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(render_metric_card(
                "Сценарий", selected_label,
            ), unsafe_allow_html=True)
        with c4:
            st.markdown(render_metric_card(
                "Шагов (суток)", str(n_steps),
            ), unsafe_allow_html=True)

        # ── Таймлайн активации по типам ──
        st.subheader("Динамика активации агентов")

        fig_act = go.Figure()
        type_labels = {
            "initiator": "Инициаторы",
            "amplifier": "Усилители",
            "mediator": "Медиаторы",
            "recipient": "Реципиенты",
        }
        agent_colors = COLORS.agent_colors
        for atype in ["initiator", "amplifier", "mediator", "recipient"]:
            fig_act.add_trace(go.Scatter(
                x=list(range(1, len(timeline[atype]) + 1)),
                y=timeline[atype],
                name=type_labels[atype],
                line=dict(color=agent_colors[atype], width=2),
                stackgroup="one",
            ))
        fig_act.update_layout(**_plotly_layout(
            title="Кумулятивная активация по типам агентов",
            xaxis_title="Шаг (сутки)",
            yaxis_title="Активных агентов",
            legend=dict(orientation="h", y=-0.18),
        ))
        _show_chart(fig_act, key="act_timeline")

        # ── Фильтр типов агентов ──
        st.subheader("Граф информационной сети")
        with st.expander("ℹ️ О графе"):
            st.markdown(render_tooltip(
                "Граф Барабаши–Альберт",
                "Масштабно-свободная сеть: большинство узлов имеет мало "
                "связей, но есть hub-узлы (крупные СМИ, инфлюенсеры) "
                "с большим числом связей. Такая структура эмпирически "
                "соответствует реальным медиасетям.",
            ), unsafe_allow_html=True)

        show_types = st.multiselect(
            "Показать типы агентов",
            options=["initiator", "amplifier", "mediator", "recipient"],
            default=["initiator", "amplifier", "mediator", "recipient"],
            format_func=lambda x: type_labels.get(x, x),
        )

        # Визуализация графа (весь граф — слишком много узлов для plotly,
        # показываем hub-подграф: top-50 по степени)
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        filtered_uids = {n["uid"] for n in nodes if n["type"] in show_types}

        # Top-N по degree среди отфильтрованных
        show_n = min(120, len(filtered_uids))
        sorted_nodes = sorted(
            [n for n in nodes if n["uid"] in filtered_uids],
            key=lambda x: -x["degree"],
        )[:show_n]
        show_uids = {n["uid"] for n in sorted_nodes}

        # Positions: simple spring layout via networkx coords
        import networkx as nx
        subgraph = nx.Graph()
        subgraph.add_nodes_from(show_uids)
        for u, v in edges:
            if u in show_uids and v in show_uids:
                subgraph.add_edge(u, v)

        if len(subgraph.nodes) > 0:
            pos = nx.spring_layout(subgraph, seed=seed, k=0.4)

            # Edge traces
            edge_x, edge_y = [], []
            for u, v in subgraph.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig_graph = go.Figure()
            fig_graph.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(color="rgba(0,0,0,0.06)", width=0.5),
                hoverinfo="none",
                showlegend=False,
            ))

            # Node traces by type
            for atype in ["initiator", "amplifier", "mediator", "recipient"]:
                if atype not in show_types:
                    continue
                tnodes = [n for n in sorted_nodes if n["type"] == atype]
                if not tnodes:
                    continue
                fig_graph.add_trace(go.Scatter(
                    x=[pos[n["uid"]][0] for n in tnodes if n["uid"] in pos],
                    y=[pos[n["uid"]][1] for n in tnodes if n["uid"] in pos],
                    mode="markers",
                    name=type_labels[atype],
                    marker=dict(
                        color=agent_colors[atype],
                        size=[
                            max(6, min(25, n["degree"] * 1.5))
                            for n in tnodes if n["uid"] in pos
                        ],
                        line=dict(
                            width=[
                                2 if n["active"] else 0
                                for n in tnodes if n["uid"] in pos
                            ],
                            color="white",
                        ),
                        opacity=[
                            1.0 if n["active"] else 0.35
                            for n in tnodes if n["uid"] in pos
                        ],
                    ),
                    text=[
                        f"{type_labels[atype]}<br>"
                        f"Степень: {n['degree']}<br>"
                        f"{'🟢 Активен' if n['active'] else '⚪ Неактивен'}"
                        for n in tnodes if n["uid"] in pos
                    ],
                    hoverinfo="text",
                ))

            fig_graph.update_layout(**_plotly_layout(
                title=f"Информационная сеть (топ-{show_n} узлов по степени)",
                showlegend=True,
                legend=dict(orientation="h", y=-0.05),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            ))
            _show_chart(fig_graph, key="network_graph")


# ─────────────────────────────────────────────────────────────────
# ВКЛАДКА 3: ЦЕПИ МАРКОВА
# ─────────────────────────────────────────────────────────────────
with tabs[2]:
    if st.session_state.sim_result is None:
        st.info("Нажмите **▶️ Запуск модели** для начала моделирования.")
    else:
        sim = st.session_state.sim_result
        markov = sim["markov"]
        trajectory = sim["trajectory"]
        traj_names = sim["trajectory_names"]

        # ── Пояснение ──
        with st.expander("ℹ️ Что такое цепи Маркова?"):
            st.markdown(render_tooltip(
                "Цепи Маркова в модели нарратива",
                "Стохастическая модель с 5 состояниями, описывающая "
                "жизненный цикл информационного нарратива. Следующее "
                "состояние зависит только от текущего.<br><br>"
                "<strong>Аналогия:</strong> как болезнь — инкубация → "
                "первые симптомы → обострение → пик → выздоровление.<br><br>"
                "<strong>Ключевая новизна:</strong> матрица переходов "
                "динамически модифицируется на основе состояния агентной "
                "сети (усилители ускоряют, медиаторы легитимизируют).",
            ), unsafe_allow_html=True)

        # ── Метрики ──
        mc1, mc2, mc3 = st.columns(3)
        markov_colors = COLORS.markov_colors
        state_keys_list = list(markov_colors.keys())

        with mc1:
            max_st = sim["max_state"]
            max_key = MARKOV_STATE_ORDER[max_st]
            st.markdown(render_metric_card(
                "Максимальное состояние",
                MARKOV_STATES[max_st]["name_ru"],
            ), unsafe_allow_html=True)
        with mc2:
            fin_st = sim["final_state"]
            st.markdown(render_metric_card(
                "Финальное состояние",
                MARKOV_STATES[fin_st]["name_ru"],
            ), unsafe_allow_html=True)
        with mc3:
            st.markdown(render_metric_card(
                "Шагов в траектории",
                str(len(trajectory)),
            ), unsafe_allow_html=True)

        # ── Траектория нарратива ──
        st.subheader("Траектория нарратива")

        state_labels_map = {
            i: MARKOV_STATES[i]["name_ru"]
            for i in range(len(MARKOV_STATE_ORDER))
        }

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=list(range(len(trajectory))),
            y=trajectory,
            mode="lines+markers",
            line=dict(color=COLORS.blue_navy, width=2.5),
            marker=dict(
                color=[
                    markov_colors[MARKOV_STATE_ORDER[s]] for s in trajectory
                ],
                size=8,
                line=dict(color="white", width=1),
            ),
            text=[state_labels_map[s] for s in trajectory],
            hovertemplate="Шаг %{x}<br>%{text}<extra></extra>",
        ))
        fig_traj.update_layout(**_plotly_layout(
            title="Жизненный цикл нарратива",
            xaxis_title="Шаг (сутки)",
            yaxis=dict(
                tickvals=list(range(5)),
                ticktext=[state_labels_map[i] for i in range(5)],
                gridcolor="rgba(0,0,0,0.06)",
            ),
        ))
        _show_chart(fig_traj, key="markov_traj")

        # ── Матрица переходов (тепловая карта) ──
        st.subheader("Матрица переходов")
        with st.expander("ℹ️ Как читать матрицу"):
            st.markdown(render_tooltip(
                "Матрица переходов 5×5",
                "Каждая ячейка P(i → j) — вероятность перехода из "
                "состояния i (строка) в состояние j (столбец) за один шаг. "
                "Сумма каждой строки = 100%. Матрица <em>динамическая</em>: "
                "модифицируется на основе активности усилителей и медиаторов.",
            ), unsafe_allow_html=True)

        # Отображаем финальную динамическую матрицу
        dyn_matrix = markov.get_dynamic_matrix(sim["abm"])
        labels = [MARKOV_STATES[i]["name_ru"] for i in range(5)]

        fig_hm = go.Figure(go.Heatmap(
            z=dyn_matrix,
            x=labels, y=labels,
            colorscale=[
                [0.0, COLORS.bg_main],
                [0.3, COLORS.yellow_kerria],
                [0.6, COLORS.orange_kaki],
                [1.0, COLORS.red_crimson],
            ],
            text=np.round(dyn_matrix * 100, 1).astype(str),
            texttemplate="%{text}%",
            textfont=dict(size=12),
            hovertemplate="P(%{y} → %{x}) = %{z:.3f}<extra></extra>",
            colorbar=dict(title="P"),
        ))
        fig_hm.update_layout(**_plotly_layout(
            title="Динамическая матрица переходов (текущее состояние сети)",
            xaxis_title="Следующее состояние →",
            yaxis_title="Текущее состояние ↓",
            yaxis=dict(autorange="reversed"),
            height=450,
        ))
        _show_chart(fig_hm, key="markov_heatmap")

        # Пояснение состояний
        st.subheader("Состояния нарратива")
        for i, key in enumerate(MARKOV_STATE_ORDER):
            state = MARKOV_STATES[key]
            if hasattr(state, "label_ru"):
                label = state.label_ru
                desc = state.description
                analogy = state.analogy
            else:
                label = state["label_ru"]
                desc = state["description"]
                analogy = state["analogy"]

            with st.expander(f"{label}"):
                st.markdown(f"**Описание:** {desc}")
                st.markdown(f"**Аналогия:** {analogy}")


# ─────────────────────────────────────────────────────────────────
# ВКЛАДКА 4: МОНТЕ-КАРЛО
# ─────────────────────────────────────────────────────────────────
with tabs[3]:
    mc_res: MonteCarloResult = st.session_state.mc_result

    if mc_res is None:
        st.info(
            "Нажмите **🎲 Монте-Карло** в боковой панели для "
            "стохастического прогнозирования."
        )
    else:
        # ── Пояснение ──
        with st.expander("ℹ️ Что такое Монте-Карло?"):
            st.markdown(render_tooltip(
                "Метод Монте-Карло",
                "Класс алгоритмов, использующих повторную случайную "
                "выборку для оценки вероятностей (Метрополис, Улам, 1949). "
                "Каждая итерация — один «возможный мир», в котором "
                "информационная кампания разворачивается при случайной "
                "вариации параметров. Результат: вероятность достижения "
                "вирусной фазы с доверительным интервалом.",
            ), unsafe_allow_html=True)

        # ── Метрики ──
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(render_metric_card(
                "P(вирусная фаза)",
                f"{mc_res.viral_probability:.1%}",
                f"ДИ 95%: [{mc_res.viral_ci_lower:.1%} – "
                f"{mc_res.viral_ci_upper:.1%}]",
            ), unsafe_allow_html=True)
        with mc2:
            steps_str = (
                f"{mc_res.mean_steps_to_viral:.1f} ± "
                f"{mc_res.std_steps_to_viral:.1f}"
                if mc_res.mean_steps_to_viral is not None else "—"
            )
            st.markdown(render_metric_card(
                "Среднее шагов до вирусной",
                steps_str,
                "суток (среднее ± σ)",
            ), unsafe_allow_html=True)
        with mc3:
            st.markdown(render_metric_card(
                "Итераций", str(mc_res.n_simulations),
            ), unsafe_allow_html=True)
        with mc4:
            st.markdown(render_metric_card(
                "Шагов / итерация", str(mc_res.n_steps),
            ), unsafe_allow_html=True)

        # ── Гистограмма пиковых состояний (4 столбика без DECLINING) ──
        st.subheader("Распределение пиковых состояний")
        with st.expander("ℹ️ Почему 4 столбика?"):
            st.markdown(render_tooltip(
                "Гистограмма пиковых состояний",
                "Показывает максимальное состояние, достигнутое нарративом "
                "в каждой из N симуляций. <strong>Без фазы затухания</strong> — "
                "потому что пиковое состояние по определению не может быть "
                "«затуханием» (нарратив сначала растёт, потом снижается).",
            ), unsafe_allow_html=True)

        peak_dist = np.asarray(mc_res.peak_distribution, dtype=np.float64)
        peak_labels = [MARKOV_STATES[i]["name_ru"] for i in range(4)]
        peak_colors = [
            markov_colors[MARKOV_STATE_ORDER[i]] for i in range(4)
        ]
        peak_pct = peak_dist[:4] * 100
        fig_peak = go.Figure(go.Bar(
            x=peak_labels,
            y=peak_pct,
            marker_color=peak_colors,
            text=[f"{v:.1f}%" for v in peak_pct],
            textposition="outside",
        ))
        fig_peak.update_layout(**_plotly_layout(
            title=f"Пиковые состояния ({mc_res.n_simulations} симуляций)",
            yaxis_title="Доля симуляций (%)",
            height=400,
        ))
        _show_chart(fig_peak, key="mc_peak")

        # ── Круговая финальных состояний (5 секторов) ──
        st.subheader("Распределение финальных состояний")
        with st.expander("ℹ️ Почему круговая отличается от гистограммы?"):
            st.markdown(render_tooltip(
                "Гистограмма ≠ круговая диаграмма",
                "Гистограмма (выше) показывает <em>пиковое</em> состояние — "
                "максимум, достигнутый за всю симуляцию. "
                "Круговая (здесь) показывает <em>финальное</em> состояние — "
                "где нарратив оказался в конце. Нарратив может достичь "
                "вирусной фазы, а затем затухнуть. Поэтому здесь 5 секторов "
                "(включая фазу затухания), и распределение отличается.",
            ), unsafe_allow_html=True)

        final_dist = np.asarray(mc_res.final_state_distribution, dtype=np.float64)
        final_labels = [MARKOV_STATES[i]["name_ru"] for i in range(5)]
        final_colors = [
            markov_colors[MARKOV_STATE_ORDER[i]] for i in range(5)
        ]
        final_pct = final_dist * 100
        fig_final = go.Figure(go.Pie(
            labels=final_labels,
            values=final_pct,
            hole=0.45,
            marker=dict(colors=final_colors),
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_final.update_layout(**_plotly_layout(
            title=f"Финальные состояния ({mc_res.n_simulations} симуляций)",
        ))
        _show_chart(fig_final, key="mc_final")


# ─────────────────────────────────────────────────────────────────
# ВКЛАДКА 5: ИНДИКАТОРЫ ОБНАРУЖЕНИЯ
# ─────────────────────────────────────────────────────────────────
with tabs[4]:
    det: DetectionResult = st.session_state.detection

    if det is None:
        st.info(
            "Нажмите **▶️ Запуск модели** для вычисления индикаторов."
        )
    else:
        # ── Пояснение ──
        with st.expander("ℹ️ Об индикаторах обнаружения"):
            st.markdown(render_tooltip(
                "6 индикаторов обнаружения",
                "Система из 6 индикаторов позволяет определить, является ли "
                "медийный фон вокруг Казахстана естественным или результатом "
                "координированного внешнего воздействия.<br><br>"
                "Индикаторы 1–4 вычисляются из данных GDELT (при их наличии). "
                "Индикаторы 5–6 — из результатов АОМ-симуляции. При отсутствии "
                "GDELT-данных индикаторы 1–4 переключаются на модельные "
                "прокси-оценки (помечаются в интерфейсе).",
            ), unsafe_allow_html=True)

        # Индекс угрозы — крупная карточка
        st.markdown(render_threat_gauge(det), unsafe_allow_html=True)

        if det.has_proxy:
            st.caption(
                "⚙️ Часть индикаторов вычислена через модельные прокси-оценки "
                "(данные GDELT недоступны или неполны)."
            )

        # ── 6 индикаторов в 3 колонки × 2 ряда ──
        indicator_keys = list(det.indicators.keys())
        for row_start in range(0, 6, 3):
            cols = st.columns(3)
            for col_idx, key in enumerate(
                indicator_keys[row_start:row_start + 3]
            ):
                with cols[col_idx]:
                    ind = det.indicators[key]
                    st.markdown(
                        render_indicator_card(ind),
                        unsafe_allow_html=True,
                    )
                    with st.expander("🔍 Подробнее"):
                        st.html(render_indicator_detail(ind))

        # ── Рекомендации ──
        rec_html = render_recommendations(det.recommendations)
        if rec_html:
            st.html(rec_html)

        # ── Кнопка сохранения в библиотеку ──
        st.divider()
        col_save, col_note = st.columns([1, 2])
        with col_note:
            save_note = st.text_input(
                "Примечание (необязательно)",
                placeholder="Комментарий к эксперименту...",
            )
        with col_save:
            if st.button("💾 Сохранить в библиотеку", use_container_width=True):
                record = create_record(
                    sim_result=st.session_state.sim_result,
                    mc_result=st.session_state.mc_result,
                    detection=det,
                    scenario_key=st.session_state.last_scenario,
                    n_agents=st.session_state.last_n_agents,
                    n_steps=st.session_state.last_n_steps,
                    seed=st.session_state.last_seed,
                    note=save_note,
                )
                lib = ResultsLibrary()
                rid = lib.add(record)
                st.success(f"✅ Сохранено! ID: {rid}")


# ─────────────────────────────────────────────────────────────────
# ВКЛАДКА 6: БИБЛИОТЕКА РЕЗУЛЬТАТОВ
# ─────────────────────────────────────────────────────────────────
with tabs[5]:
    with st.expander("ℹ️ О библиотеке"):
        st.markdown(render_tooltip(
            "Библиотека результатов",
            "Журнал всех сохранённых экспериментов. Каждая запись содержит "
            "параметры, результаты моделирования, значения индикаторов "
            "и рекомендации. Записи можно удалить или выбрать для "
            "генерации отчёта.",
        ), unsafe_allow_html=True)

    lib = ResultsLibrary()

    if lib.is_empty:
        st.html(render_library_empty())
    else:
        st.html(render_library_stats(lib))

        records = lib.get_all()
        selected_ids: list = []

        for rec in records:
            col_card, col_actions = st.columns([5, 1])
            with col_card:
                st.html(render_library_card(rec))
            with col_actions:
                cb = st.checkbox(
                    "📋",
                    key=f"sel_{rec.record_id}",
                    help="Выбрать для отчёта",
                )
                if cb:
                    selected_ids.append(rec.record_id)

                if st.button(
                    "🗑️", key=f"del_{rec.record_id}",
                    help="Удалить запись",
                ):
                    lib.delete(rec.record_id)
                    st.rerun()

        # Выбранные для отчёта
        if selected_ids:
            st.session_state["report_record_ids"] = selected_ids
            st.success(
                f"Выбрано для отчёта: {len(selected_ids)} записей. "
                f"Перейдите на вкладку «Генерация отчёта»."
            )

        st.divider()
        if st.button("🗑️ Очистить всю библиотеку", type="secondary"):
            cnt = lib.clear_all()
            st.warning(f"Удалено записей: {cnt}")
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# ВКЛАДКА 7: ГЕНЕРАЦИЯ ОТЧЁТА
# ─────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Генерация аналитического отчёта")
    with st.expander("ℹ️ Об отчёте"):
        st.markdown(render_tooltip(
            "Аналитический отчёт",
            "PDF-документ с графиками, значениями индикаторов и "
            "рекомендациями экспертной системы. Формируется на основе "
            "выбранных записей из библиотеки результатов.",
        ), unsafe_allow_html=True)

    report_ids = st.session_state.get("report_record_ids", [])
    lib = ResultsLibrary()

    if not report_ids or lib.is_empty:
        st.info(
            "Выберите записи на вкладке «Библиотека результатов» "
            "(отметьте 📋), затем вернитесь сюда."
        )
    else:
        selected_records = lib.get_selected(report_ids)
        st.write(f"Выбрано записей: **{len(selected_records)}**")

        # Сводная таблица
        summary = lib.export_summary_table(report_ids)
        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True)

        # Кнопка генерации
        if st.button("📄 Подготовить отчёт", type="primary", use_container_width=True):
            with st.spinner("Формирование PDF-отчёта..."):
                try:
                    pdf_bytes = generate_report(
                        records=selected_records,
                        report_date=datetime.now().strftime("%d.%m.%Y"),
                    )
                    st.success("Отчёт сформирован.")
                    st.download_button(
                        label="⬇️ Скачать PDF-отчёт",
                        data=pdf_bytes,
                        file_name=f"ai_osint_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Ошибка генерации отчёта: {e}")

        # Экспорт JSON
        if st.button("📥 Экспорт JSON", use_container_width=True):
            json_str = lib.export_json(report_ids)
            st.download_button(
                label="⬇️ Скачать JSON",
                data=json_str,
                file_name=f"ai_osint_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
            )


# ─────────────────────────────────────────────────────────────────
# ВКЛАДКА 8: О СИСТЕМЕ
# ─────────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("О системе AI-OSINT")

    # ── Что это ──
    st.markdown(f"""
    <div class="aio-card">
        <h3>Что это</h3>
        <p style="line-height:1.7;">
            <strong>AI-OSINT</strong> — аналитическая платформа, определяющая,
            является ли медийный фон вокруг Казахстана в глобальном
            информационном пространстве естественным или результатом
            координированного внешнего воздействия.
        </p>
        <p style="line-height:1.7;">
            Интегрирует методы OSINT (разведка на основе открытых источников),
            NLP (обработка естественного языка), агентно-ориентированное
            моделирование (АОМ), цепи Маркова и стохастическое моделирование
            методом Монте-Карло.
        </p>
        <p style="line-height:1.7; color:{COLORS.text_secondary};">
            Зонтичный термин: <em>вычислительное имитационное моделирование</em>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Как пользоваться ──
    st.markdown(f"""
    <div class="aio-card">
        <h3>Как пользоваться</h3>
        <div style="line-height:1.8;">
            <strong>1.</strong> Нажмите <strong>📡 Загрузить данные</strong> —
            система подключится к GDELT API и получит актуальный поток
            публикаций за указанный период.<br>
            <strong>2.</strong> Изучите <strong>Панель мониторинга</strong> —
            частота упоминаний, тональность, географическая и языковая
            структура потока.<br>
            <strong>3.</strong> Выберите сценарий и нажмите <strong>▶️ Запуск
            модели</strong> — АОМ-симуляция покажет, как нарратив
            распространяется в информационной сети.<br>
            <strong>4.</strong> Запустите <strong>🎲 Монте-Карло</strong> —
            стохастическая оценка вероятности достижения вирусной фазы.<br>
            <strong>5.</strong> Перейдите на вкладку <strong>Индикаторы
            обнаружения</strong> — 6 индикаторов + агрегированный Индекс
            угрозы + рекомендации экспертной системы.<br>
            <strong>6.</strong> Сохраните результат в <strong>Библиотеку</strong>
            и сформируйте <strong>отчёт</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Методология ──
    st.markdown(f"""
    <div class="aio-card">
        <h3>Методология</h3>
        <div style="line-height:1.8;">
            <strong>Данные:</strong> GDELT Project — открытый глобальный
            мониторинг СМИ, 100+ языков, обновление каждые 15 минут.
            7 языков анализа: 6 ООН + казахский.<br><br>
            <strong>АОМ:</strong> 4 типа агентов (Инициатор, Усилитель,
            Медиатор, Реципиент) в масштабно-свободной сети Барабаши–Альберт.
            Калибровка по Stanford IO Reports, Meta CIB Reports,
            IRA Troll Tweets (FiveThirtyEight).<br><br>
            <strong>Цепи Маркова:</strong> 5 состояний нарратива с
            динамической модификацией матрицы переходов на основе
            состояния агентной сети.<br><br>
            <strong>Монте-Карло:</strong> стохастическое прогнозирование
            (100–5000 итераций) с доверительными интервалами.<br><br>
            <strong>Индикаторы:</strong> 6 маркеров координации
            (аномальность, синхронность, текстовая гомогенность,
            тональный сдвиг, скорость распространения, бот-активность) +
            агрегированный Индекс угрозы (0–100).<br><br>
            <strong>Экспертная система:</strong> набор продукционных правил
            (if-then) по комбинациям значений индикаторов
            (Buchanan, Shortliffe, 1984).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Теоретическая значимость ──
    st.html(f"""
    <div class="aio-card">
        <h3>Теоретическая значимость</h3>
        <p style="line-height:1.7;">
            Теоретическая база проекта опирается на современные подходы
            к исследованию информационной безопасности, цифровой коммуникации
            и распространения нарративов в сетевой среде. Информационные
            кампании в цифровую эпоху развиваются не линейно, а как сложные
            сетевые процессы, в которых участвуют различные типы акторов:
            инициаторы, усилители, медиаторы и массовая аудитория.
        </p>
        <p style="line-height:1.7;">
            Для моделирования такой среды применяются методы
            агентно-ориентированного моделирования, позволяющие воспроизводить
            поведение множества автономных участников информационного
            пространства, а также оценивать механизмы их взаимного влияния.
            Дополнительно используются марковские цепи, отражающие переход
            нарратива между стадиями латентности, возникновения, роста,
            вирусного распространения и затухания, и метод Монте-Карло,
            позволяющий не только фиксировать текущие аномалии,
            но и оценивать вероятностные сценарии дальнейшей эскалации.
        </p>
    </div>
    """)

    # ── Социальная значимость ──
    st.html(f"""
    <div class="aio-card">
        <h3>Социальная значимость</h3>
        <p style="line-height:1.7;">
            Проект направлен на укрепление устойчивости общества
            к деструктивным информационным воздействиям, способным
            провоцировать недоверие к государственным институтам,
            усиливать социальную поляризацию и распространять
            манипулятивные или панические интерпретации событий.
        </p>
        <p style="line-height:1.7;">
            Система позволяет своевременно выявлять информационные волны,
            затрагивающие межэтнические отношения, региональные противоречия,
            протестные настроения, а также чувствительные темы, связанные
            с идентичностью, языком, исторической памятью и социальным
            неравенством. Тем самым проект ориентирован не только на
            технологическое решение аналитической задачи, но и на поддержку
            общественной стабильности и информационной гигиены.
        </p>
    </div>
    """)

    # ── Политическая значимость ──
    st.html(f"""
    <div class="aio-card">
        <h3>Политическая значимость</h3>
        <p style="line-height:1.7;">
            Проект создаёт инструмент, способный повысить качество
            аналитического сопровождения государственных решений в сфере
            информационной безопасности и стратегических коммуникаций.
        </p>
        <p style="line-height:1.7;">
            В условиях роста внешнего информационного давления особое
            значение приобретает способность государства своевременно
            выявлять и интерпретировать попытки внешнего или трансграничного
            воздействия на общественное мнение. Платформа AI-OSINT может
            использоваться для анализа кампаний, направленных на дискредитацию
            внешнеполитического курса Казахстана, подрыв доверия к институтам
            власти, распространение нарративов о якобы существующих
            внутренних расколах, а также для мониторинга тем, способных
            быть использованными в качестве триггеров информационной
            дестабилизации.
        </p>
        <p style="line-height:1.7; color:{COLORS.text_secondary};">
            Проект соотносится с задачами укрепления цифрового суверенитета,
            повышения национальной устойчивости и развития собственных
            аналитических решений в сфере искусственного интеллекта.
        </p>
    </div>
    """)

    # ── Возможности применения ──
    st.html(f"""
    <div class="aio-card">
        <h3>Возможности применения</h3>
        <div style="line-height:1.8;">
            <strong>1. Раннее выявление координированных информационных атак</strong><br>
            <span style="color:{COLORS.text_secondary};">
            Фиксация аномального роста упоминаний, резкого сдвига тональности
            и синхронного распространения однотипных сообщений. Переход
            от реактивного мониторинга к проактивному обнаружению угроз.</span><br><br>

            <strong>2. Анализ деструктивных нарративов</strong><br>
            <span style="color:{COLORS.text_secondary};">
            Выявление и отслеживание нарративов о межэтнической
            напряжённости, языковом вопросе, региональном расколе,
            территориальной нестабильности. Определение скорости
            распространения, тональности, географии источников
            и признаков координации.</span><br><br>

            <strong>3. Определение географии информационного давления</strong><br>
            <span style="color:{COLORS.text_secondary};">
            Выявление стран и медиасегментов, из которых чаще всего
            возникают или тиражируются чувствительные сюжеты
            о Казахстане.</span><br><br>

            <strong>4. Оценка рисков перед международными мероприятиями</strong><br>
            <span style="color:{COLORS.text_secondary};">
            Мониторинг информационного фона перед саммитами, форумами,
            переговорами, визитами иностранных лидеров.</span><br><br>

            <strong>5. Поддержка аналитических подразделений</strong><br>
            <span style="color:{COLORS.text_secondary};">
            Подготовка ситуационных справок, аналитических записок
            и предупреждений для МИД, структур информационной
            безопасности, экспертных организаций.</span><br><br>

            <strong>6. Выявление бот-сетей</strong><br>
            <span style="color:{COLORS.text_secondary};">
            Анализ признаков координированного недостоверного поведения:
            синхронность публикаций, повторяемость формулировок,
            аномальная доля репостов, автоматизированная активность.</span><br><br>

            <strong>7. Научный и учебный анализ</strong><br>
            <span style="color:{COLORS.text_secondary};">
            Исследовательский инструмент для изучения информационных
            операций, сетевого распространения нарративов, цифровой
            политической коммуникации.</span><br><br>

            <strong>8. Оценка вероятности эскалации</strong><br>
            <span style="color:{COLORS.text_secondary};">
            Прогноз развития ситуации через марковские цепи и симуляцию
            Монте-Карло: вероятность достижения вирусной стадии
            и скорость эскалации.</span><br><br>

            <strong>9. Мониторинг репутационных рисков</strong><br>
            <span style="color:{COLORS.text_secondary};">
            Отслеживание восприятия Казахстана в глобальных СМИ
            по темам внешней политики, инвестиций, безопасности,
            прав человека, энергетики и транзита. Долгосрочные
            изменения международного информационного образа страны.</span>
        </div>
    </div>
    """)

    # ── Источники ──
    st.markdown(f"""
    <div class="aio-card">
        <h3>Источники и калибровка</h3>
        <div style="line-height:1.8; font-size:0.88rem;">
            • GDELT Project — <a href="https://gdeltproject.org"
              target="_blank" style="color:{COLORS.blue_indigo};">
              gdeltproject.org</a><br>
            • Stanford Internet Observatory —
              <a href="https://cyber.fsi.stanford.edu/io" target="_blank"
              style="color:{COLORS.blue_indigo};">cyber.fsi.stanford.edu/io</a><br>
            • Meta CIB Reports —
              <a href="https://about.fb.com/news/tag/coordinated-inauthentic-behavior/"
              target="_blank" style="color:{COLORS.blue_indigo};">
              about.fb.com</a><br>
            • IRA Troll Tweets (FiveThirtyEight) —
              <a href="https://github.com/fivethirtyeight/russian-troll-tweets/"
              target="_blank" style="color:{COLORS.blue_indigo};">
              github.com/fivethirtyeight</a><br>
            • Botometer (Indiana University) —
              <a href="https://botometer.osome.iu.edu/" target="_blank"
              style="color:{COLORS.blue_indigo};">botometer.osome.iu.edu</a><br>
            • Barabási A.-L., Albert R. (1999) — масштабно-свободные сети<br>
            • Beskow D., Carley K. (2019) — фреймворк BEND для IO<br>
            • Nimmo B. (2019) — The Breakout Scale (DFRLab)<br>
            • Gleicher N. (2020) — Meta CIB taxonomy<br>
            • Varol O. et al. (2017) — Botometer / BotOrNot
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Авторы ──
    st.markdown(f"""
    <div class="aio-card">
        <h3>Авторы</h3>
        <div style="line-height:1.8;">
            {'<br>'.join(
                f'<strong>{a["name"]}</strong>, {a["title"]} — {a["role"]}'
                for a in AUTHORS
            )}
            <br><br>
            <strong>{UNIVERSITY}</strong> | {YEAR}<br>
            Конкурс «AI SANA — Digital Kazakhstan: Projects of the Future»
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 7. ПОДВАЛ
# ═══════════════════════════════════════════════════════════════════

st.markdown(render_footer(), unsafe_allow_html=True)
