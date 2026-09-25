"""Юниты линии A: маскирование, квота, тождество Q и поиска, батчевый rollout.

Всё на синтетических матрицах и CPU. Это лестница фазы 2, ступень 1: она
обязана ловить ошибки до того, как за них заплатят часами GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from envs.action_index import ActionIndex
from envs.corpus_reader import CorpusReader
from envs.parallel_search_env import ParallelSearchEnv, make_state_memories
from envs.search_env import DenseSearchEnv, SearchDatasetAdapter, make_search_envs
from envs.title_table import TitleTable
from rl.q_module import (
    TextQNet,
    SearchBoltzmannPolicy,
    masked_soft_value,
    normalized_boltzmann_probs,
)
from tests.toy_search import (
    ConstantFeedback,
    ToyActionTower,
    ToyAgent,
    ToyTokenizer,
    ToyTower,
    write_corpus,
    write_offsets,
)


DIM = 8


def toy_matrix(rows: int = 24, dim: int = DIM, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, dim, generator=generator)


def toy_titles(rows: int = 24, chunks_per_title: int = 3) -> np.ndarray:
    """Каждая статья занимает подряд идущие строки, как в Wiki-18."""
    return np.arange(rows, dtype=np.int32) // chunks_per_title


# --------------------------------------------------------------------------
# Поиск и маскирование
# --------------------------------------------------------------------------


def test_search_reproduces_cosine_ranking_on_a_toy_matrix() -> None:
    """Ранжирование поиска обязано совпасть с ранжированием по косинусу.

    Строки матрицы нормированы, поэтому скалярное произведение и косинус
    задают один и тот же порядок; расхождение означало бы ошибку в самом
    поиске, а не в геометрии.
    """
    vectors = torch.nn.functional.normalize(toy_matrix(), dim=-1)
    index = ActionIndex(vectors)
    query = torch.nn.functional.normalize(toy_matrix(rows=2, seed=11), dim=-1)

    result = index.search(query, top_k=5)

    expected = torch.nn.functional.cosine_similarity(
        query[:, None, :], vectors[None, :, :], dim=-1
    )
    for row in range(query.shape[0]):
        order = torch.argsort(expected[row], descending=True)[:5]
        assert result.row_ids[row].tolist() == order.tolist()
        assert torch.allclose(result.scores[row], expected[row][order], atol=1e-6)


def test_mask_is_applied_before_topk() -> None:
    """Замаскированная строка не занимает места в пуле, а исчезает из него."""
    vectors = toy_matrix()
    index = ActionIndex(vectors, toy_titles())
    query = vectors[0:1].clone()

    free = index.search(query, top_k=3)
    blocked_row = int(free.row_ids[0, 0])
    masked = index.search(query, top_k=3, blocked_rows=[[blocked_row]])

    assert blocked_row not in masked.row_ids[0].tolist()
    assert len(set(masked.row_ids[0].tolist())) == 3
    assert bool(masked.masked_argmax[0]) is True
    assert bool(free.masked_argmax[0]) is False


def test_title_quota_removes_every_chunk_of_an_exhausted_article() -> None:
    vectors = toy_matrix()
    titles = toy_titles()
    index = ActionIndex(vectors, titles)
    query = vectors[0:1].clone()

    result = index.search(query, top_k=6, blocked_titles=[[0]])

    found = index.titles_of(result.row_ids)[0].tolist()
    assert 0 not in found


def test_search_refuses_to_return_masked_rows_as_padding() -> None:
    """Если доступных строк меньше K, тихо добить пул замаскированными нельзя."""
    vectors = toy_matrix(rows=6)
    index = ActionIndex(vectors, toy_titles(rows=6, chunks_per_title=2))

    with pytest.raises(RuntimeError, match="unmasked rows"):
        index.search(vectors[0:1], top_k=5, blocked_rows=[[0, 1]])


# --------------------------------------------------------------------------
# V(s) и маска до topk
# --------------------------------------------------------------------------


def test_unavailable_global_argmax_does_not_leak_into_v() -> None:
    """Главный дефект PQN: V схлопывалась в Q недоступного действия.

    Логит недоступного действия здесь на порядок больше всех остальных. При
    сдвиге по глобальному максимуму и alpha=0.005 экспоненты доступных
    действий обнулялись, и V выходила равной `max − alpha·log(...)`, то есть
    оценке действия, которое взять нельзя.
    """
    alpha = 0.005
    logits = torch.tensor([[1.0, 2.0, 50.0]])
    available = torch.tensor([[True, True, False]])

    v1, v2 = masked_soft_value(logits, logits, available, alpha, top_k_actions=3)

    reachable = alpha * torch.logsumexp(logits[0, :2] / alpha, dim=0)
    assert torch.allclose(v1, reachable.reshape(1), atol=1e-5)
    assert torch.allclose(v2, reachable.reshape(1), atol=1e-5)
    assert float(v1) < 3.0


def test_v_matches_plain_logsumexp_when_nothing_is_masked() -> None:
    alpha = 0.1
    logits = torch.tensor([[1.0, 2.0, 0.5, -1.0]])
    available = torch.ones_like(logits, dtype=torch.bool)

    v1, _ = masked_soft_value(logits, logits, available, alpha, top_k_actions=4)

    expected = alpha * torch.logsumexp(logits / alpha, dim=-1)
    assert torch.allclose(v1, expected, atol=1e-6)


def test_top_k_actions_limits_the_pool_v_sees() -> None:
    alpha = 0.5
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    available = torch.ones_like(logits, dtype=torch.bool)

    v_all, _ = masked_soft_value(logits, logits, available, alpha, top_k_actions=4)
    v_two, _ = masked_soft_value(logits, logits, available, alpha, top_k_actions=2)

    expected_two = alpha * torch.logsumexp(logits[0, :2] / alpha, dim=0)
    assert torch.allclose(v_two, expected_two.reshape(1), atol=1e-6)
    assert float(v_two) < float(v_all)


# --------------------------------------------------------------------------
# Q(s, a) на готовых векторах
# --------------------------------------------------------------------------


def test_q_on_ready_vectors_equals_dot_products_of_halves() -> None:
    """`TextQNet` с готовым вектором обязан считать ровно то же, что поиск."""
    torch.manual_seed(0)
    tower = ToyTower(dim=DIM, seed=5)
    critic = TextQNet(tower, action_embed=None)
    tokenizer = ToyTokenizer()

    memories = make_state_memories(["what is a duck", "what is a goose"])
    from envs.utils import stack_memory

    state = stack_memory(memories, tokenizer, max_length=32)
    actions = toy_matrix(rows=2, seed=7)

    logits_1, logits_2 = critic(state, actions)

    embeds = tower(state.input_ids, state.attention_mask)
    half = DIM // 2
    assert torch.allclose(logits_1, (embeds[:, :half] * actions[:, :half]).sum(-1))
    assert torch.allclose(logits_2, (embeds[:, half:] * actions[:, half:]).sum(-1))
    # Тождество поиска и оценки: полная сумма голов — это скор поиска.
    assert torch.allclose(logits_1 + logits_2, (embeds * actions).sum(-1), atol=1e-6)


def test_two_heads_are_kept_and_are_not_equal() -> None:
    """Двухголовость не «упрощается»: лосс фитит половины по отдельности."""
    tower = ToyTower(dim=DIM, seed=5)
    critic = TextQNet(tower, action_embed=None)
    tokenizer = ToyTokenizer()
    from envs.utils import stack_memory

    state = stack_memory(make_state_memories(["a question here"]), tokenizer, 32)
    logits_1, logits_2 = critic(state, toy_matrix(rows=1, seed=9))

    assert logits_1.shape == logits_2.shape == (1,)
    assert not torch.allclose(logits_1, logits_2)


# --------------------------------------------------------------------------
# Exploration
# --------------------------------------------------------------------------


def test_boltzmann_probabilities_are_scale_invariant() -> None:
    """α безразмерна: умножение логитов на константу ничего не меняет.

    Это и есть смысл нормировки на разброс: ‖s‖ за 11 часов обучения упала
    в шесть раз, и с фиксированным α exploration был бы задушен на старте.
    """
    logits = torch.tensor([[1.0, 2.0, 3.0, 2.5], [0.0, -1.0, 4.0, 1.0]])

    base = normalized_boltzmann_probs(logits, alpha=0.7)
    for factor in (0.01, 3.0, 250.0):
        assert torch.allclose(
            normalized_boltzmann_probs(logits * factor, alpha=0.7), base, atol=1e-5
        )
    # Сдвиг тоже безразличен: пул сравнивается сам с собой.
    assert torch.allclose(
        normalized_boltzmann_probs(logits + 17.0, alpha=0.7), base, atol=1e-5
    )


def test_epsilon_zero_never_injects_and_evaluate_is_deterministic() -> None:
    policy = SearchBoltzmannPolicy(epsilon=0.0)
    pool = torch.tensor([[1.0, 5.0, 2.0]])
    gte = torch.tensor([[9.0, 0.0, 0.0]])

    positions, injected = policy(pool, gte, temperature=1.0, evaluate=True)
    assert positions.tolist() == [1]
    assert injected.tolist() == [False]

    _, injected = policy(pool, gte, temperature=1.0)
    assert injected.tolist() == [False]


def test_epsilon_one_always_takes_the_gte_pool() -> None:
    policy = SearchBoltzmannPolicy(epsilon=1.0, injection_sampling="uniform")
    pool = torch.tensor([[1.0, 5.0, 2.0]] * 32)
    gte = torch.tensor([[0.0, 0.0, 9.0]] * 32)

    _, injected = policy(pool, gte, temperature=1.0)

    assert bool(injected.all())


# --------------------------------------------------------------------------
# Среда и батчевый rollout
# --------------------------------------------------------------------------


CORPUS_ROWS = [
    ("Alpha", "alpha one passage"),
    ("Alpha", "alpha two passage"),
    ("Alpha", "alpha three passage"),
    ("Beta", "beta one passage"),
    ("Beta", "beta two passage"),
    ("Beta", "beta three passage"),
    ("Gamma", "gamma one passage"),
    ("Gamma", "gamma two passage"),
    ("Gamma", "gamma three passage"),
    ("Delta", "delta one passage"),
    ("Delta", "delta two passage"),
    ("Delta", "delta three passage"),
]


def make_corpus(tmp_path: Path) -> CorpusReader:
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus = tmp_path / "wiki.jsonl"
    offsets = tmp_path / "offsets.npy"
    write_corpus(corpus, CORPUS_ROWS)
    write_offsets(corpus, offsets)
    return CorpusReader(corpus, offsets, expected_rows=len(CORPUS_ROWS))


def make_title_table() -> TitleTable:
    titles = ["Alpha", "Beta", "Gamma", "Delta"]
    return TitleTable(toy_titles(rows=len(CORPUS_ROWS)), titles)


def make_samples() -> list[dict]:
    return [
        {
            "_id": "q0",
            "question": "who is alpha",
            "answer": "alpha",
            "supporting_facts": [["Alpha", 0], ["Beta", 1]],
            "source": "toy",
            # Кандидатные поля обязаны игнорироваться средой прямого поиска.
            "candidates": [[0, 1]],
            "judgements": [1],
            "betas": [0.5],
            "context": [["Alpha", ["ignored"]]],
        },
        {
            "_id": "q1",
            "question": "who is gamma",
            "answer": "gamma",
            "supporting_facts": [["Gamma", 0], ["Nowhere", 0]],
            "source": "toy",
        },
    ]


def make_parallel(tmp_path: Path, envs_parallel: int, max_steps: int = 4, top_k: int = 4):
    table = make_title_table()
    dataset = SearchDatasetAdapter(make_samples(), table)
    template = DenseSearchEnv(
        dataset=dataset,
        max_steps=max_steps,
        feedback_model=ConstantFeedback(1.0),
        max_chunks_per_title=2,
    )
    envs = make_search_envs(template, envs_parallel, seed=0)
    index = ActionIndex(toy_matrix(rows=len(CORPUS_ROWS)), toy_titles(rows=len(CORPUS_ROWS)))
    parallel = ParallelSearchEnv(
        envs=envs,
        index=index,
        corpus=make_corpus(tmp_path),
        state_tokenizer=ToyTokenizer(),
        max_state_segment_length=32,
        top_k=top_k,
        policy=SearchBoltzmannPolicy(epsilon=0.0),
    )
    return parallel, dataset


def make_agent(alpha: float = 1.0, top_k_actions: int = 4) -> ToyAgent:
    return ToyAgent(
        state_tower=ToyTower(dim=DIM, seed=1),
        target_tower=ToyTower(dim=DIM, seed=2),
        action_tower=ToyActionTower(ToyTower(dim=DIM, seed=3)),
        alpha=alpha,
        top_k_actions=top_k_actions,
    )


def test_dataset_adapter_keeps_only_question_answer_and_supporting_facts(tmp_path: Path) -> None:
    dataset = SearchDatasetAdapter(make_samples(), make_title_table())

    sample = dataset[0]
    assert set(sample) == {
        "id",
        "key",
        "question",
        "answer",
        "answer_variants",
        "supporting_facts",
        "gold_titles",
        "source",
        "gold_titles_covered",
        "gold_title_ids",
    }
    assert sample["gold_titles"] == ["Alpha", "Beta"]
    assert sample["gold_titles_covered"] is True
    # У второго примера один gold-титул отсутствует в корпусе — это и есть
    # флаг, который отделяет решаемые эпизоды от нерешаемых.
    assert dataset[1]["gold_titles_covered"] is False
    assert dataset[1]["gold_title_ids"] == [2]


def test_episode_never_repeats_a_row_and_respects_the_quota(tmp_path: Path) -> None:
    # Эпизоды намеренно не доходят до конца: после reset счётчики очищаются,
    # и проверять было бы нечего.
    parallel, dataset = make_parallel(tmp_path, envs_parallel=2, max_steps=6)
    agent = make_agent()
    for env in parallel.envs:
        env.reset(dataset[0])

    parallel.rollout(batch_size=8, agent=agent, evaluate=True)

    for env in parallel.envs:
        assert len(env.selected_rows) == 4
        assert len(env.selected_rows) == len(set(env.selected_rows))
        assert max(env.title_counts.values()) <= 2
    parallel.close()


def test_batched_rollout_matches_the_stepwise_one(tmp_path: Path) -> None:
    """Батч обязателен по стоимости, но не должен менять сами переходы."""
    agent = make_agent()

    # Шагов меньше, чем `max_steps`: эпизоды не завершаются, и сравнение не
    # зависит от того, какой пример среда вытянет из датасета следующим.
    batched, dataset = make_parallel(tmp_path / "batched", envs_parallel=3, max_steps=5)
    for index, env in enumerate(batched.envs):
        env.reset(dataset[index % len(dataset)])
    _, batch, _ = batched.rollout(batch_size=9, agent=agent, evaluate=True)
    batched_rows = [list(env.selected_rows) for env in batched.envs]
    batched.close()

    stepwise_rows = []
    for index in range(3):
        single, dataset = make_parallel(
            tmp_path / f"single-{index}", envs_parallel=1, max_steps=5
        )
        single.envs[0].reset(dataset[index % len(dataset)])
        _, single_batch, _ = single.rollout(batch_size=3, agent=agent, evaluate=True)
        stepwise_rows.append(list(single.envs[0].selected_rows))
        assert torch.allclose(
            single_batch.q_values[0], batch.q_values[index], atol=1e-5
        )
        single.close()

    assert batched_rows == stepwise_rows


def test_rollout_batch_shapes_line_up_with_the_pqn_update(tmp_path: Path) -> None:
    parallel, dataset = make_parallel(tmp_path, envs_parallel=2, max_steps=3)
    agent = make_agent()
    for env in parallel.envs:
        env.reset(dataset[0])

    returns, batch, stats = parallel.rollout(batch_size=6, agent=agent, evaluate=True)

    num_envs, num_steps = batch.reward.shape
    assert num_envs == 2
    assert batch.not_done.shape == (num_envs, num_steps)
    assert batch.valid.shape == (num_envs, num_steps)
    # `compute_returns` бутстрапится из V(s_{t+1}), поэтому значений на одно
    # больше, чем переходов.
    assert batch.q_values.shape == (num_envs, num_steps + 1)
    assert batch.action.shape == (num_envs * num_steps, DIM)
    assert len(batch.state.input_ids) == num_envs * num_steps
    assert returns  # эпизоды длиной 3 шага успели завершиться
    assert stats["explore/injected_share"] == 0.0
    assert 0.0 <= stats["pool/gold_title_recall"] <= 1.0
    parallel.close()


def test_rollout_batch_feeds_pqn_update_without_reshaping(tmp_path: Path) -> None:
    """Шов между батчевым rollout и `PQN.update`: формы и порядок сходятся.

    Действия здесь — тензор готовых векторов, а не `TextMemoryItem`, поэтому
    критик не перекодирует текст и не крутит RoPE.
    """
    from rl.agents.pqn import PQN, AlphaSchedule

    parallel, dataset = make_parallel(tmp_path, envs_parallel=2, max_steps=4)
    for env in parallel.envs:
        env.reset(dataset[0])
    _, batch, _ = parallel.rollout(batch_size=4, agent=make_agent(), evaluate=True)
    parallel.close()

    class VectorCritic(torch.nn.Module):
        # Контракт TextQNet: train_step сравнивает с таргетом калиброванные
        # головы; некалиброванный критик отдаёт прежние 2·q_i.
        calibrated = False

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(DIM))

        def forward(self, state, action):
            # Готовый вектор, а не TextMemoryItem: перекодировать нечего.
            assert isinstance(action, torch.Tensor)
            assert action.shape[-1] == DIM
            scores = (action * self.weight).sum(-1)
            return scores / 2, scores / 2

        def head_values(self, logits_1, logits_2):
            return 2 * logits_1, 2 * logits_2

    class NullScheduler:
        def step(self) -> None:
            pass

    agent = PQN.__new__(PQN)
    agent.gamma = 0.99
    agent.Lambda = 0.6
    agent.tau = 0.02
    agent.accumulate_grads = 1
    agent.max_grad_norm = 2.0
    agent._update_step = 0
    agent._optim_step = 0
    agent.alpha = 0.005
    agent.alpha_schedule = AlphaSchedule(start=0.01, kind="linear", final=0.001, total=10)
    agent.train_state_embed = False
    agent.train_action_embed = False
    agent.critic = VectorCritic()
    agent.critic_trainable_params = list(agent.critic.parameters())
    agent.critic_optim = torch.optim.SGD(agent.critic_trainable_params, lr=0.0)
    agent.scheduler = NullScheduler()
    agent.train_step = agent.make_train_step()
    agent.train = lambda: None

    loss = agent.update(
        batch.state,
        batch.action,
        None,
        batch.q_values,
        batch.reward,
        batch.not_done,
        batch.valid,
    )

    assert np.isfinite(loss)
    # Один оптимизационный шаг сдвинул α по расписанию, а не по learning rate.
    assert agent._optim_step == 1
    assert agent.alpha == pytest.approx(AlphaSchedule(
        start=0.01, kind="linear", final=0.001, total=10
    ).value(1))


def test_failed_reward_invalidates_the_whole_episode(tmp_path: Path) -> None:
    """Ошибка vLLM — не «reward 0»: эпизод целиком выпадает из лосса."""
    table = make_title_table()
    dataset = SearchDatasetAdapter(make_samples(), table)
    template = DenseSearchEnv(
        dataset=dataset,
        max_steps=2,
        feedback_model=ConstantFeedback(1.0, fail=True),
        max_chunks_per_title=2,
    )
    parallel = ParallelSearchEnv(
        envs=make_search_envs(template, 1),
        index=ActionIndex(
            toy_matrix(rows=len(CORPUS_ROWS)), toy_titles(rows=len(CORPUS_ROWS))
        ),
        corpus=make_corpus(tmp_path),
        state_tokenizer=ToyTokenizer(),
        max_state_segment_length=32,
        top_k=4,
        policy=SearchBoltzmannPolicy(epsilon=0.0),
    )
    parallel.envs[0].reset(dataset[0])

    _, batch, _ = parallel.rollout(batch_size=2, agent=make_agent(), evaluate=True)

    assert batch.valid.tolist() == [[False, False]]
    parallel.close()
