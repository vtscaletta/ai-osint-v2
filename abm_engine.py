"""
AI-OSINT — Ядро вычислительного имитационного моделирования
============================================================

Три интегрированных движка:

1. AgentBasedModel   — агентно-ориентированная модель информационных
                       кампаний с четырьмя типами агентов на графе
                       Барабаши–Альберт (Barabási, Albert, 1999).

2. MarkovNarrative   — марковские цепи с 5 состояниями нарратива
                       и динамической модификацией матрицы переходов
                       на основе состояния агентной сети.

3. MonteCarloEngine  — стохастическое прогнозирование методом
                       Монте-Карло (N итераций, доверительные
                       интервалы, распределение пиковых состояний).

Теоретические основания:
    - Шеллинг Т. (1971) — модели сегрегации, первые ABM
    - Аксельрод Р. (1984) — эволюция кооперации
    - Седерман Л.-Э. (1997) — агентные модели в МО
    - Бесков Д., Карли К. (2019) — фреймворк BEND для IO
    - Барабаши А.-Л., Альберт Р. (1999) — масштабно-свободные сети

Калибровка:
    - Stanford IO «Unheard Voice» (август 2022): синхронность > 0.4,
      bot-сети на 7+ платформах
    - Meta CIB Reports (2017–2025): 200+ сетей, reposts > 90%,
      CV < 0.1
    - IRA Troll Tweets (FiveThirtyEight): 3 млн твитов, 2 848 аккаунтов,
      TTR < 0.28

Зависимости:
    numpy, networkx, config (внутренний модуль)

Авторы:
    Абсаттаров Г.Р., Саятбек С., Кайролла А.Б.
    КазУМОиМЯ им. Абылай хана | 2026
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import (
    AGENT_TYPES,
    SCENARIOS,
    MARKOV_STATES,
    MARKOV_BASE_MATRIX,
    ABM_DEFAULTS,
    INDICATOR_THRESHOLDS,
)


# ═══════════════════════════════════════════════════════════════════
# 1.  АГЕНТ — единица моделирования
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Agent:
    """
    Информационный агент в сети.

    Параметры определяются типом (config.AGENT_TYPES) и стохастически
    варьируются при инициализации для обеспечения гетерогенности
    популяции — ключевое требование ABM (Epstein, Axtell, 1996).

    Attributes
    ----------
    uid : int
        Уникальный идентификатор (совпадает с узлом графа).
    agent_type : str
        Один из: «initiator», «amplifier», «mediator», «recipient».
    active : bool
        Активирован ли агент (передаёт нарратив).
    influence : float
        Способность воздействовать на соседей [0, 1].
    susceptibility : float
        Восприимчивость к чужому нарративу [0, 1].
    reach : int
        Охват аудитории (количество потенциальных реципиентов).
    bot_probability : float
        Вероятность того, что агент — бот (только для amplifier).
    repost_rate : float
        Доля репостов в общей активности (только для amplifier).
    language : str
        Язык агента: «ru», «kz», «en».
    exposure_count : int
        Счётчик воздействий (сколько раз получил нарратив).
    activation_step : int | None
        Шаг, на котором агент был впервые активирован.
    """

    uid: int
    agent_type: str
    active: bool = False
    influence: float = 0.0
    susceptibility: float = 0.0
    reach: int = 0
    bot_probability: float = 0.0
    repost_rate: float = 0.0
    language: str = "ru"
    exposure_count: int = 0
    activation_step: Optional[int] = None

    # ──────── фабрика ────────

    @classmethod
    def create(cls, uid: int, agent_type: str, rng: np.random.Generator) -> "Agent":
        """
        Фабричный метод: создаёт агента с параметрами из config.AGENT_TYPES
        и стохастической вариацией.

        Parameters
        ----------
        uid : int
            Идентификатор узла в графе.
        agent_type : str
            Тип агента.
        rng : numpy.random.Generator
            Генератор случайных чисел (воспроизводимость через seed).

        Returns
        -------
        Agent
        """
        spec = AGENT_TYPES[agent_type]

        influence = rng.uniform(*spec["influence_range"])
        susceptibility = rng.uniform(*spec["susceptibility_range"])
        reach = int(rng.uniform(*spec["reach_range"]))
        language = rng.choice(
            ["ru", "kz", "en"],
            p=[0.55, 0.30, 0.15],
        )

        bot_probability = 0.0
        repost_rate = 0.0
        if agent_type == "amplifier":
            bot_probability = rng.uniform(*spec["bot_probability_range"])
            repost_rate = rng.uniform(*spec["repost_rate_range"])

        return cls(
            uid=uid,
            agent_type=agent_type,
            active=(agent_type == "initiator"),
            influence=influence,
            susceptibility=susceptibility,
            reach=reach,
            bot_probability=bot_probability,
            repost_rate=repost_rate,
            language=language,
        )


# ═══════════════════════════════════════════════════════════════════
# 2.  АГЕНТНО-ОРИЕНТИРОВАННАЯ МОДЕЛЬ (АОМ)
# ═══════════════════════════════════════════════════════════════════

class AgentBasedModel:
    """
    Агентно-ориентированная модель распространения информационных
    нарративов в масштабно-свободной сети.

    Архитектура
    -----------
    Агенты четырёх типов размещены в графе Барабаши–Альберт,
    характеризующемся наличием hub-узлов (крупные СМИ, инфлюенсеры).
    Такая топология эмпирически соответствует структуре реальных
    медиасетей (Barabási, Albert, 1999).

    Типология агентов (по фреймворку BEND, Beskow & Carley, 2019):
        - Инициатор  (5%)  — генерация оригинального нарратива
        - Усилитель  (20%) — бот-сети, массовый репост
        - Медиатор   (10%) — СМИ, легитимизация нарратива
        - Реципиент  (65%) — обычные пользователи

    Пропорции определены на основе:
        - Meta CIB Reports: «seed accounts» ~5%
        - IRA Troll Tweets: Hashtag Gamer (доля репостов > 90%) ~20%
        - Stanford IO: media amplifiers ~10%
        - Стандартная пропорция аудитории ~65%

    Формула активации
    -----------------
    P(activation) = P_state × S_agent × (1 + N_active / N_neighbors)

    где:
        P_state      — вероятность, определяемая текущим состоянием
                       цепи Маркова (чем «правее» состояние, тем выше)
        S_agent      — восприимчивость агента (susceptibility)
        N_active     — число активных соседей в графе
        N_neighbors  — общее число соседей

    Каждый шаг = 1 сутки (указано в спецификации, раздел 7).

    Parameters
    ----------
    n_agents : int
        Размер популяции (по умолчанию из ABM_DEFAULTS).
    scenario_key : str
        Ключ сценария из config.SCENARIOS.
    seed : int
        Seed для воспроизводимости.
    """

    # Вероятность активации, привязанная к состоянию Маркова
    # Индекс = номер состояния (0=LATENT … 4=DECLINING)
    # Низкие базовые значения: без координации нарратив не разгоняется
    STATE_ACTIVATION_PROB: np.ndarray = np.array(
        [0.01, 0.05, 0.18, 0.50, 0.08],
        dtype=np.float64,
    )

    def __init__(
        self,
        n_agents: int = ABM_DEFAULTS["n_agents"],
        scenario_key: str = "organic",
        seed: int = ABM_DEFAULTS["seed"],
    ) -> None:
        self.n_agents = n_agents
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tick: int = 0

        # ── Сценарий ──
        if scenario_key not in SCENARIOS:
            raise ValueError(
                f"Неизвестный сценарий «{scenario_key}». "
                f"Допустимые: {list(SCENARIOS.keys())}"
            )
        self.scenario: dict = SCENARIOS[scenario_key]

        # ── Граф Барабаши–Альберт ──
        # m=3: каждый новый узел присоединяется к 3 существующим
        # (предпочтительное присоединение → power-law distribution)
        self.graph: nx.Graph = nx.barabasi_albert_graph(
            n_agents, m=3, seed=seed,
        )

        # ── Популяция агентов ──
        self.agents: List[Agent] = self._init_agents()

        # ── Синхронизация графа с агентами ──
        for agent in self.agents:
            self.graph.nodes[agent.uid]["type"] = agent.agent_type
            self.graph.nodes[agent.uid]["active"] = agent.active

        # ── Журнал симуляции ──
        self.history: List[Dict] = []

    # ──────── инициализация популяции ────────

    def _init_agents(self) -> List[Agent]:
        """
        Создаёт N агентов с типами, распределёнными по пропорциям
        из config.AGENT_TYPES.

        В сценарии с координированной кампанией доля усилителей
        определяется полем scenario["amp_ratio"].
        """
        proportions = {
            "initiator": AGENT_TYPES["initiator"]["share"],
            "amplifier": self.scenario["amp_ratio"],
            "mediator": AGENT_TYPES["mediator"]["share"],
        }
        # Реципиенты — остаток
        proportions["recipient"] = max(
            0.0,
            1.0 - sum(proportions.values()),
        )

        # Нормализуем (на случай amp_ratio > стандартного)
        total = sum(proportions.values())
        p = np.array(
            [proportions[t] / total for t in
             ["initiator", "amplifier", "mediator", "recipient"]],
            dtype=np.float64,
        )

        type_keys = ["initiator", "amplifier", "mediator", "recipient"]
        assigned_types = self.rng.choice(
            type_keys, size=self.n_agents, p=p,
        )

        return [
            Agent.create(uid=i, agent_type=t, rng=self.rng)
            for i, t in enumerate(assigned_types)
        ]

    # ──────── ядро: один шаг симуляции ────────

    def step(self, markov_state: int) -> Dict:
        """
        Выполняет один шаг (= 1 сутки) симуляции.

        Алгоритм:
            1. Для каждого неактивного агента проверить P(activation):
               P = P_state × S × (1 + N_active / N_neighbors)
            2. Активные усилители: двукратная вероятность активации
               соседей (bot-механизм).
            3. Медиаторы: при активации увеличивают reach.
            4. Записать снимок в историю.

        Parameters
        ----------
        markov_state : int
            Текущее состояние цепи Маркова (0–4).

        Returns
        -------
        dict
            Снимок шага: tick, число активных по типам, общая
            доля активных, средний exposure.
        """
        self.tick += 1
        p_state = self.STATE_ACTIVATION_PROB[markov_state]

        # ── Деактивация: агенты «теряют интерес» ──
        # В органических кампаниях без подпитки нарратив затухает:
        # агенты перестают распространять информацию.
        # В координированных — бот-сети поддерживают активность.
        #
        # Базовая вероятность деактивации = 0.15 (15% в сутки).
        # Усилители (боты) не деактивируются.
        # Координация снижает p_deactivation через amp_ratio.
        base_deactivation = 0.15
        amp_ratio_current = self.get_active_ratio("amplifier")
        p_deactivation = base_deactivation * (1.0 - amp_ratio_current * 0.8)

        for agent in self.agents:
            if (
                agent.active
                and agent.agent_type not in ("initiator", "amplifier")
                and agent.activation_step is not None
                and agent.activation_step < self.tick
            ):
                if self.rng.random() < p_deactivation:
                    agent.active = False
                    self.graph.nodes[agent.uid]["active"] = False

        newly_activated: List[int] = []

        for agent in self.agents:
            if agent.active:
                continue

            neighbors = list(self.graph.neighbors(agent.uid))
            if not neighbors:
                continue

            n_active = sum(
                1 for nid in neighbors
                if self.agents[nid].active
            )
            n_neighbors = len(neighbors)

            if n_active == 0:
                continue

            # ── Формула активации ──
            # P = P_state × S_agent × (1 + N_active / N_neighbors)
            p_activate = (
                p_state
                * agent.susceptibility
                * (1.0 + n_active / n_neighbors)
            )

            # Усилители-соседи дают буст ×2
            has_active_amplifier = any(
                self.agents[nid].active
                and self.agents[nid].agent_type == "amplifier"
                for nid in neighbors
            )
            if has_active_amplifier:
                p_activate *= 2.0

            p_activate = min(p_activate, 1.0)

            if self.rng.random() < p_activate:
                agent.active = True
                agent.activation_step = self.tick
                newly_activated.append(agent.uid)

            # Exposure: каждый контакт с активным соседом += 1
            agent.exposure_count += n_active

        # ── Обновление графа ──
        for uid in newly_activated:
            self.graph.nodes[uid]["active"] = True

        # ── Снимок ──
        snapshot = self._snapshot()
        self.history.append(snapshot)
        return snapshot

    # ──────── снимок состояния ────────

    def _snapshot(self) -> Dict:
        """Формирует снимок текущего состояния сети."""
        by_type = {}
        for t in ["initiator", "amplifier", "mediator", "recipient"]:
            total_t = sum(1 for a in self.agents if a.agent_type == t)
            active_t = sum(
                1 for a in self.agents
                if a.agent_type == t and a.active
            )
            by_type[t] = {
                "total": total_t,
                "active": active_t,
                "ratio": active_t / max(total_t, 1),
            }

        total_active = sum(1 for a in self.agents if a.active)
        mean_exposure = np.mean(
            [a.exposure_count for a in self.agents]
        )

        return {
            "tick": self.tick,
            "total_active": total_active,
            "active_ratio": total_active / self.n_agents,
            "by_type": by_type,
            "mean_exposure": float(mean_exposure),
            "newly_activated": sum(
                1 for a in self.agents
                if a.activation_step == self.tick
            ),
        }

    # ──────── аналитика ────────

    def get_active_ratio(self, agent_type: Optional[str] = None) -> float:
        """Доля активных агентов (всех или конкретного типа)."""
        if agent_type is None:
            return sum(1 for a in self.agents if a.active) / self.n_agents
        subset = [a for a in self.agents if a.agent_type == agent_type]
        if not subset:
            return 0.0
        return sum(1 for a in subset if a.active) / len(subset)

    def get_graph_data(self) -> Dict:
        """
        Экспорт данных графа для визуализации.

        Returns
        -------
        dict
            nodes: список словарей {uid, type, active, degree}
            edges: список кортежей (source, target)
        """
        nodes = []
        for agent in self.agents:
            nodes.append({
                "uid": agent.uid,
                "type": agent.agent_type,
                "active": agent.active,
                "degree": self.graph.degree(agent.uid),
                "influence": agent.influence,
                "reach": agent.reach,
            })
        edges = list(self.graph.edges())
        return {"nodes": nodes, "edges": edges}

    def get_activation_timeline(self) -> Dict[str, List[int]]:
        """
        Таймлайн активации по типам.

        Returns
        -------
        dict
            Ключ = тип агента, значение = список числа активных на каждом шаге.
        """
        timeline = {t: [] for t in ["initiator", "amplifier", "mediator", "recipient"]}
        for snap in self.history:
            for t in timeline:
                timeline[t].append(snap["by_type"][t]["active"])
        return timeline

    def reset(self, seed: Optional[int] = None) -> None:
        """Сброс модели для нового прогона (с опциональным новым seed)."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.tick = 0
        self.history.clear()
        self.agents = self._init_agents()
        for agent in self.agents:
            self.graph.nodes[agent.uid]["type"] = agent.agent_type
            self.graph.nodes[agent.uid]["active"] = agent.active


# ═══════════════════════════════════════════════════════════════════
# 3.  ЦЕПИ МАРКОВА
# ═══════════════════════════════════════════════════════════════════

class MarkovNarrative:
    """
    Марковская модель жизненного цикла информационного нарратива.

    Нарратив проходит 5 дискретных состояний::

        ЛАТЕНТНАЯ → ЗАРОЖДЕНИЕ → РОСТ → ВИРУСНАЯ → ЗАТУХАНИЕ

    Свойство Маркова: следующее состояние зависит только от текущего
    (P(X_{t+1} | X_t, X_{t-1}, …) = P(X_{t+1} | X_t)).

    Аналогия для нетехнаря: инкубационный период → первые симптомы →
    обострение → пик болезни → выздоровление.

    Ключевая новизна: матрица переходов P(i→j) не статична —
    она динамически модифицируется на основе состояния агентной сети:

    1. Усилители (amp_ratio > 0.3):
       boost = amp_ratio × 0.25
       Вероятность «правого» перехода ↑, самоудержания ↓

    2. Медиаторы (active_mediators > 2):
       P(РОСТ → ВИРУСНАЯ) += min(0.15, active_mediators × 0.03)
       Медиаторы легитимизируют нарратив, ускоряя прорыв.

    Калибровка базовой матрицы:
        - P(LATENT→LATENT) = 0.85: без воздействия нарратив на 85%
          остаётся скрытым (Stanford IO: большинство IO не достигают
          аудитории)
        - P(VIRAL→DECLINING) = 0.50: без подпитки вирусный нарратив
          быстро затухает (Meta CIB: после takedown охват падает
          на 50–70% за сутки)
        - P(DECLINING→LATENT) = 0.15: затухший нарратив может
          полностью исчезнуть из повестки

    Parameters
    ----------
    initial_state : int
        Начальное состояние (по умолчанию 0 = ЛАТЕНТНАЯ).
    """

    N_STATES: int = len(MARKOV_STATES)

    def __init__(self, initial_state: int = 0) -> None:
        if not 0 <= initial_state < self.N_STATES:
            raise ValueError(
                f"Состояние должно быть в [0, {self.N_STATES - 1}], "
                f"получено: {initial_state}"
            )
        self.state: int = initial_state
        self.trajectory: List[int] = [initial_state]
        self.base_matrix: np.ndarray = MARKOV_BASE_MATRIX.copy()

    # ──────── динамическая матрица ────────

    def get_dynamic_matrix(self, abm: AgentBasedModel) -> np.ndarray:
        """
        Рассчитывает модифицированную матрицу переходов
        на основе текущего состояния агентной сети.

        Parameters
        ----------
        abm : AgentBasedModel
            Текущая ABM-модель (нужна для подсчёта amp_ratio
            и числа активных медиаторов).

        Returns
        -------
        numpy.ndarray
            Матрица 5×5, строки нормализованы до 1.0.
        """
        M = self.base_matrix.copy()

        # ── Подсчёт состояния сети ──
        amp_ratio = abm.get_active_ratio("amplifier")
        total_active_ratio = abm.get_active_ratio()
        active_mediators = sum(
            1 for a in abm.agents
            if a.agent_type == "mediator" and a.active
        )
        scenario_amp = abm.scenario["amp_ratio"]

        # ── Сценарный модификатор ──
        # Доля усилителей в популяции (scenario.amp_ratio) определяет
        # базовый «уровень координации»: чем выше — тем легче нарративу
        # продвигаться к вирусной фазе, даже до активации всех ботов.
        #
        # Это отражает реальность: координированные сети изначально
        # спроектированы для быстрого распространения (Meta CIB, 2023).
        coordination_factor = scenario_amp * 0.4
        for i in range(4):  # LATENT .. VIRAL (без DECLINING)
            shift = min(coordination_factor, M[i][i] * 0.4)
            M[i][i] -= shift
            M[i][min(i + 1, 4)] += shift

        # ── Условие 1: активные усилители ──
        # Если > 30% усилителей активны → дополнительный разгон «вправо»
        if amp_ratio > 0.3:
            boost = amp_ratio * 0.20
            for i in range(4):
                shift = min(boost, M[i][i] * 0.4)
                M[i][i] -= shift
                M[i][min(i + 1, 4)] += shift

        # ── Условие 2: медиаторы ──
        # Легитимизация: РОСТ → ВИРУСНАЯ ускоряется
        if active_mediators > 2:
            med_boost = min(0.15, active_mediators * 0.03)
            M[2][3] += med_boost
            M[2][2] -= med_boost

        # ── Условие 3: откат при низкой активности ──
        # Если общая доля активных < 30% — нарратив деградирует
        # (без аудитории информационная волна затухает).
        if total_active_ratio < 0.30:
            pullback = 0.10 * (1.0 - total_active_ratio)
            for i in range(1, 5):  # EMERGING .. DECLINING
                shift = min(pullback, M[i][i] * 0.5)
                M[i][i] -= shift
                M[i][max(i - 1, 0)] += shift

        # ── Нормализация строк ──
        M = np.clip(M, 0.001, None)
        for i in range(self.N_STATES):
            M[i] /= M[i].sum()

        return M

    # ──────── шаг ────────

    def step(self, abm: AgentBasedModel) -> int:
        """
        Выполняет один переход цепи Маркова.

        Parameters
        ----------
        abm : AgentBasedModel
            Текущая ABM (для динамической модификации матрицы).

        Returns
        -------
        int
            Новое состояние.
        """
        M = self.get_dynamic_matrix(abm)
        probabilities = M[self.state]
        self.state = abm.rng.choice(self.N_STATES, p=probabilities)
        self.trajectory.append(self.state)
        return self.state

    # ──────── аналитика ────────

    def get_state_name(self, state_idx: Optional[int] = None) -> str:
        """Русскоязычное название состояния."""
        idx = state_idx if state_idx is not None else self.state
        return MARKOV_STATES[idx]["name_ru"]

    def get_trajectory_names(self) -> List[str]:
        """Траектория в виде списка русских названий."""
        return [MARKOV_STATES[s]["name_ru"] for s in self.trajectory]

    def get_max_state(self) -> int:
        """Максимальное достигнутое состояние."""
        return max(self.trajectory)

    def reset(self, initial_state: int = 0) -> None:
        """Сброс цепи."""
        self.state = initial_state
        self.trajectory = [initial_state]


# ═══════════════════════════════════════════════════════════════════
# 4.  МОНТЕ-КАРЛО
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MonteCarloResult:
    """
    Результат стохастической симуляции Монте-Карло.

    Attributes
    ----------
    n_simulations : int
        Количество проведённых итераций.
    n_steps : int
        Количество шагов в каждой итерации.
    scenario_key : str
        Ключ сценария.
    viral_probability : float
        P(VIRAL) — доля итераций, достигших вирусной фазы.
    viral_ci_lower : float
        Нижняя граница 95% доверительного интервала для P(VIRAL).
    viral_ci_upper : float
        Верхняя граница 95% ДИ.
    mean_steps_to_viral : float | None
        Среднее число шагов до вирусной фазы (None если ни разу
        не достигнута).
    std_steps_to_viral : float | None
        Стандартное отклонение шагов до вирусной фазы.
    peak_distribution : numpy.ndarray
        Распределение пиковых состояний (5 элементов, нормализовано).
    final_state_distribution : numpy.ndarray
        Распределение финальных состояний (5 элементов).
    peak_states_raw : List[int]
        Сырые данные: пиковое состояние каждой итерации.
    final_states_raw : List[int]
        Сырые данные: финальное состояние каждой итерации.
    activation_ratios : List[float]
        Финальная доля активных агентов в каждой итерации.
    """

    n_simulations: int
    n_steps: int
    scenario_key: str
    viral_probability: float
    viral_ci_lower: float
    viral_ci_upper: float
    mean_steps_to_viral: Optional[float]
    std_steps_to_viral: Optional[float]
    peak_distribution: np.ndarray
    final_state_distribution: np.ndarray
    peak_states_raw: List[int] = field(default_factory=list)
    final_states_raw: List[int] = field(default_factory=list)
    activation_ratios: List[float] = field(default_factory=list)


class MonteCarloEngine:
    """
    Стохастическое прогнозирование методом Монте-Карло.

    Запускает N полных симуляций (ABM + Markov) с вариацией
    начальных условий и шума матрицы переходов. Собирает
    статистику по распределению исходов.

    Метод Монте-Карло — класс алгоритмов, использующих повторную
    случайную выборку для получения числовых результатов (Metropolis,
    Ulam, 1949). Применительно к данной задаче: каждая итерация —
    это один «возможный мир», в котором информационная кампания
    разворачивается при случайной вариации параметров.

    Доверительный интервал P(VIRAL) рассчитывается по формуле Вальда:

        CI = p̂ ± z × √(p̂(1 − p̂) / n)

    где p̂ — наблюдаемая частота, z = 1.96 (95% ДИ), n — число итераций.

    Стохастический шум матрицы переходов: ±5% (uniform) на каждый
    элемент каждой строки, с последующей нормализацией.

    Parameters
    ----------
    n_agents : int
        Размер популяции в каждой итерации.
    scenario_key : str
        Ключ сценария.
    base_seed : int
        Базовый seed (итерация i получает seed = base_seed + i).
    """

    # Амплитуда шума матрицы переходов (±5%)
    MATRIX_NOISE: float = 0.05

    # z-значение для 95% доверительного интервала
    Z_95: float = 1.96

    def __init__(
        self,
        n_agents: int = ABM_DEFAULTS["n_agents"],
        scenario_key: str = "organic",
        base_seed: int = ABM_DEFAULTS["seed"],
    ) -> None:
        self.n_agents = n_agents
        self.scenario_key = scenario_key
        self.base_seed = base_seed

    # ──────── основной метод ────────

    def run(
        self,
        n_simulations: int = 1000,
        n_steps: int = 40,
    ) -> MonteCarloResult:
        """
        Запускает N полных симуляций.

        Parameters
        ----------
        n_simulations : int
            Количество итераций (рекомендуется 1000+).
        n_steps : int
            Количество шагов в каждой итерации (1 шаг = 1 сутки).

        Returns
        -------
        MonteCarloResult
            Агрегированные результаты с доверительными интервалами.
        """
        peak_states: List[int] = []
        final_states: List[int] = []
        steps_to_viral: List[int] = []
        activation_ratios: List[float] = []

        for i in range(n_simulations):
            iter_seed = self.base_seed + i

            # ── Инициализация ABM и Markov для каждой итерации ──
            abm = AgentBasedModel(
                n_agents=self.n_agents,
                scenario_key=self.scenario_key,
                seed=iter_seed,
            )
            markov = MarkovNarrative(initial_state=0)

            # ── Стохастический шум матрицы переходов ──
            noise_rng = np.random.default_rng(iter_seed + 100_000)
            noise_matrix = noise_rng.uniform(
                -self.MATRIX_NOISE,
                self.MATRIX_NOISE,
                size=(MarkovNarrative.N_STATES, MarkovNarrative.N_STATES),
            )
            markov.base_matrix = np.clip(
                MARKOV_BASE_MATRIX + noise_matrix,
                0.001, None,
            )
            # Нормализация строк после шума
            for row_idx in range(MarkovNarrative.N_STATES):
                markov.base_matrix[row_idx] /= markov.base_matrix[row_idx].sum()

            # ── Прогон ──
            max_state = 0
            viral_step: Optional[int] = None

            for step_num in range(1, n_steps + 1):
                # Шаг Маркова
                new_state = markov.step(abm)
                # Шаг ABM
                abm.step(markov_state=new_state)

                if new_state > max_state:
                    max_state = new_state

                # Первое достижение ВИРУСНОЙ фазы (индекс 3)
                if new_state == 3 and viral_step is None:
                    viral_step = step_num

            peak_states.append(max_state)
            final_states.append(markov.state)
            activation_ratios.append(abm.get_active_ratio())

            if viral_step is not None:
                steps_to_viral.append(viral_step)

        # ── Агрегация ──
        viral_count = sum(1 for ps in peak_states if ps >= 3)
        p_viral = viral_count / n_simulations

        # Доверительный интервал Вальда (95%)
        if n_simulations > 0:
            se = np.sqrt(p_viral * (1 - p_viral) / n_simulations)
            ci_lower = max(0.0, p_viral - self.Z_95 * se)
            ci_upper = min(1.0, p_viral + self.Z_95 * se)
        else:
            ci_lower = ci_upper = 0.0

        # Распределения
        peak_dist = np.bincount(peak_states, minlength=5) / max(n_simulations, 1)
        final_dist = np.bincount(final_states, minlength=5) / max(n_simulations, 1)

        return MonteCarloResult(
            n_simulations=n_simulations,
            n_steps=n_steps,
            scenario_key=self.scenario_key,
            viral_probability=p_viral,
            viral_ci_lower=ci_lower,
            viral_ci_upper=ci_upper,
            mean_steps_to_viral=(
                float(np.mean(steps_to_viral)) if steps_to_viral else None
            ),
            std_steps_to_viral=(
                float(np.std(steps_to_viral)) if steps_to_viral else None
            ),
            peak_distribution=peak_dist,
            final_state_distribution=final_dist,
            peak_states_raw=peak_states,
            final_states_raw=final_states,
            activation_ratios=activation_ratios,
        )


# ═══════════════════════════════════════════════════════════════════
# 5.  ИНТЕГРИРОВАННЫЙ ПРОГОН (фасад для app.py)
# ═══════════════════════════════════════════════════════════════════

def run_full_simulation(
    n_agents: int = ABM_DEFAULTS["n_agents"],
    n_steps: int = ABM_DEFAULTS["n_steps"],
    scenario_key: str = "organic",
    seed: int = ABM_DEFAULTS["seed"],
) -> Dict:
    """
    Единый прогон ABM + Markov (без Монте-Карло).

    Используется для визуализации одной симуляции на вкладках
    «АОМ-симуляция» и «Цепи Маркова».

    Parameters
    ----------
    n_agents : int
        Размер популяции.
    n_steps : int
        Число шагов (1 шаг = 1 сутки).
    scenario_key : str
        Ключ сценария.
    seed : int
        Seed для воспроизводимости.

    Returns
    -------
    dict
        abm : AgentBasedModel — модель после прогона
        markov : MarkovNarrative — цепь после прогона
        history : list — снимки каждого шага
        trajectory : list[int] — траектория Маркова
        trajectory_names : list[str] — русские названия
        max_state : int — максимальное достигнутое состояние
        final_state : int — финальное состояние
        activation_timeline : dict — таймлайн активации по типам
    """
    abm = AgentBasedModel(
        n_agents=n_agents,
        scenario_key=scenario_key,
        seed=seed,
    )
    markov = MarkovNarrative(initial_state=0)

    for _ in range(n_steps):
        new_state = markov.step(abm)
        abm.step(markov_state=new_state)

    return {
        "abm": abm,
        "markov": markov,
        "history": abm.history,
        "trajectory": markov.trajectory,
        "trajectory_names": markov.get_trajectory_names(),
        "max_state": markov.get_max_state(),
        "final_state": markov.state,
        "activation_timeline": abm.get_activation_timeline(),
        "graph_data": abm.get_graph_data(),
    }


def run_monte_carlo(
    n_agents: int = ABM_DEFAULTS["n_agents"],
    n_simulations: int = 1000,
    n_steps: int = ABM_DEFAULTS["n_steps"],
    scenario_key: str = "organic",
    base_seed: int = ABM_DEFAULTS["seed"],
) -> MonteCarloResult:
    """
    Запуск Монте-Карло. Обёртка для app.py.

    Parameters
    ----------
    n_agents : int
        Размер популяции.
    n_simulations : int
        Количество итераций (100–5000).
    n_steps : int
        Шагов в каждой итерации.
    scenario_key : str
        Ключ сценария.
    base_seed : int
        Базовый seed.

    Returns
    -------
    MonteCarloResult
    """
    engine = MonteCarloEngine(
        n_agents=n_agents,
        scenario_key=scenario_key,
        base_seed=base_seed,
    )
    return engine.run(n_simulations=n_simulations, n_steps=n_steps)
