"""Batched rollout of direct-search environments.

The original ``ParallelTextEnv`` batches only action selection and steps the
environments in a plain loop. That is fine for a reranker, whose candidates
come from the dataset. For direct search every environment step reads 60 GiB
from HBM, and the measurement is unambiguous: one query costs 15.9 ms, sixteen
cost 35 ms. Hence a different order here:

    states of all envs → one encoding pass (online, target and GTE) →
    one batched search per tower → transitions handed out

The reader and judge remain blocking HTTP calls made concurrent by a thread
pool: environments are independent, and waiting on them sequentially would add
up the latencies of sixteen calls.
"""

from __future__ import annotations

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from envs.search_env import DenseSearchEnv
from envs.utils import TextMemory, stack_memory
from rl.q_module import calibrated_soft_value, masked_soft_value


SearchTrainBatch = namedtuple("SearchTrainBatch", [
    "state", "action", "reward", "not_done", "q_values", "valid"
])


def make_state_memories(texts: Sequence[str], pool_size: int = 1) -> list[TextMemory]:
    """States in the form ``stack_memory`` expects.

    ``available_mask`` is a dummy here: in direct search the action set is
    defined by the search result, not by the state, but ``stack_memory`` pads
    this field and fails on an empty sequence.
    """
    return [
        TextMemory(
            item_ids=[],
            available_ids=set(),
            available_mask=np.ones(pool_size, dtype=bool),
            text=text,
            input_ids=None,
            attention_mask=None,
        )
        for text in texts
    ]


class ParallelSearchEnv:
    """Batched transition collection from direct-search environments."""

    def __init__(
        self,
        envs: Sequence[DenseSearchEnv],
        index,
        corpus,
        state_tokenizer,
        max_state_segment_length: int,
        top_k: int,
        policy,
        max_workers: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if not envs:
            raise ValueError("At least one environment is required")
        self.envs = list(envs)
        self.index = index
        self.corpus = corpus
        self.state_tokenizer = state_tokenizer
        self.max_state_segment_length = max_state_segment_length
        self.top_k = int(top_k)
        self.policy = policy
        self.device = device if device is not None else torch.get_default_device()
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers or len(self.envs),
            thread_name_prefix="search-env",
        )
        self.episodic_returns = np.zeros(len(self.envs))
        self.monitor = _Monitor()

    def reset_monitor(self) -> None:
        """Start a new statistics window; called after logging."""
        self.monitor = _Monitor()

    def close(self) -> None:
        self.pool.shutdown(wait=True)

    def reset(self) -> list[str]:
        self.episodic_returns[:] = 0.0
        return [env.reset() for env in self.envs]

    def _encode(self, texts: Sequence[str], towers: Sequence[Any]) -> list[Tensor]:
        """Encode the state text with all towers in one pass.

        Tokenization is shared: the state tower, its target copy and the frozen
        action tower are the same GTE with the same tokenizer, so there is no
        point in tokenizing the text three times.
        """
        batch = stack_memory(
            make_state_memories(texts),
            self.state_tokenizer,
            max_length=self.max_state_segment_length,
            device=self.device,
        )
        outputs = []
        for tower in towers:
            embeds = tower(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )
            if isinstance(embeds, dict):
                # EmbedderNone returns {"rope": embeds} without the rotation:
                # line A actions are raw rows of M, not rotated vectors.
                embeds = embeds["rope"]
            outputs.append(embeds)
        return [batch, *outputs]

    @staticmethod
    def _gte_tower(agent):
        """The frozen action tower as the query encoder for ε-injection."""
        embedder = agent.critic.action_embed

        def encode(input_ids, attention_mask):
            return embedder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                positions=None,
            )

        return encode

    def _masks(self) -> tuple[list[list[int]], list[list[int]]]:
        return (
            [env.blocked_rows() for env in self.envs],
            [env.blocked_titles() for env in self.envs],
        )

    def _apply_steps(
        self,
        row_ids: Sequence[int],
        title_ids: Sequence[int],
        envs: Sequence[DenseSearchEnv] | None = None,
    ):
        """Read the selected chunks and fetch rewards concurrently."""

        def work(env: DenseSearchEnv, row_id: int, title_id: int):
            text = str(self.corpus.read_row(int(row_id))["contents"])
            return text, env.step(int(row_id), text, int(title_id))

        futures = [
            self.pool.submit(work, env, row_id, title_id)
            for env, row_id, title_id in zip(
                self.envs if envs is None else envs, row_ids, title_ids
            )
        ]
        return [future.result() for future in futures]

    @torch.no_grad()
    def run_episodes(self, agent, samples: Sequence[dict]) -> list[dict[str, Any]]:
        """Run a fixed set of examples with the deterministic policy.

        There is no target search and no GTE top-K pool here: V(s') is not
        needed at eval and `evaluate=True` never injects, while every extra
        search is another read of the whole matrix.
        """
        results: list[dict[str, Any]] = []
        with agent.online_models_mode(training=False):
            for start in range(0, len(samples), len(self.envs)):
                block = list(samples[start:start + len(self.envs)])
                active = self.envs[:len(block)]
                for env, sample in zip(active, block):
                    env.reset(sample)
                rewards = [0.0] * len(block)
                done = [False] * len(block)
                # A vLLM error at eval does not crash the run, but must not
                # silently pull the curve down with zeros: such episodes are
                # counted separately.
                valid = [True] * len(block)
                while not all(done):
                    running = [i for i, finished in enumerate(done) if not finished]
                    texts = [active[i].state_text for i in running]
                    _, s_online = self._encode(texts, [agent.critic.state_embed])
                    pool = self.index.search(
                        s_online,
                        self.top_k,
                        [active[i].blocked_rows() for i in running],
                        [active[i].blocked_titles() for i in running],
                        with_vectors=False,
                    )
                    chosen = pool.row_ids[:, 0]
                    titles = self.index.titles_of(chosen)
                    steps = self._apply_steps(
                        chosen.tolist(),
                        titles.tolist(),
                        envs=[active[i] for i in running],
                    )
                    for position, (_, step) in zip(running, steps):
                        rewards[position] += step.reward
                        done[position] = step.done
                        valid[position] = valid[position] and step.valid
                for env, sample, reward, ok in zip(active, block, rewards, valid):
                    metrics = {}
                    getter = getattr(env.feedback_model, "get_metrics", None)
                    if getter is not None:
                        metrics = getter()
                    results.append(
                        {
                            "id": sample.get("id"),
                            "key": sample.get("key", sample.get("id")),
                            "source": sample.get("source", "unknown"),
                            "reward": reward,
                            "em": float(metrics.get("EM", 0.0)),
                            "em_alias": float(metrics.get("em_alias", 0.0)),
                            "valid": bool(ok),
                            "gold_titles_covered": sample.get("gold_titles_covered"),
                            "pred_idx": list(env.selected_rows),
                        }
                    )
        return results

    @torch.no_grad()
    def rollout(
        self,
        batch_size: int,
        agent,
        evaluate: bool = False,
        online_models_train_mode: bool = True,
    ) -> tuple[list[float], SearchTrainBatch, dict[str, float]]:
        with agent.online_models_mode(training=online_models_train_mode):
            return self._rollout(batch_size, agent, evaluate)

    def _rollout(self, batch_size: int, agent, evaluate: bool):
        num_envs = len(self.envs)
        alpha = float(agent.alpha)

        states: list[list[TextMemory]] = [[] for _ in range(num_envs)]
        actions: list[list[Tensor]] = [[] for _ in range(num_envs)]
        rewards: list[list[float]] = [[] for _ in range(num_envs)]
        not_done: list[list[int]] = [[] for _ in range(num_envs)]
        valid: list[list[bool]] = [[] for _ in range(num_envs)]
        values: list[list[float]] = [[] for _ in range(num_envs)]
        episode_start = [0] * num_envs
        episode_returns: list[float] = []
        # The monitor persists across rollout calls: one call collects 1–2
        # steps per env, episodes do not finish in every call, and
        # episode-level stats at an eval point would be systematically empty.
        # The training loop resets it after logging (reset_monitor).
        monitor = self.monitor

        collected = 0
        while True:
            texts = [env.state_text for env in self.envs]
            _, s_online, s_target, s_gte = self._encode(
                texts,
                [
                    agent.critic.state_embed,
                    agent.v_net_target.state_embed,
                    self._gte_tower(agent),
                ],
            )
            blocked_rows, blocked_titles = self._masks()

            # Three searches with the same masks: the action pool, the V(s')
            # estimate and the GTE top-K of the current state. A batch of
            # sixteen queries costs only twice as much as one, so they must not
            # be split.
            pool = self.index.search(
                s_online, self.top_k, blocked_rows, blocked_titles
            )
            target_pool = self.index.search(
                s_target, self.top_k, blocked_rows, blocked_titles
            )
            gte_pool = self.index.search(
                s_gte, self.top_k, blocked_rows, blocked_titles, with_vectors=False
            )

            # V(s) is computed over the target's own pool: the target values
            # what it would retrieve itself. The pool is already masked, so
            # the availability mask is all ones and top_k_actions is the pool size.
            dim = s_target.shape[-1] // 2
            target_logits_1 = (
                s_target[:, None, :dim] * target_pool.vectors[..., :dim]
            ).sum(-1)
            target_logits_2 = (
                s_target[:, None, dim:] * target_pool.vectors[..., dim:]
            ).sum(-1)
            if getattr(agent.critic, "calibrated", False):
                # The heads are fitted to the target in calibrated form, so
                # V(s') must use calibrated values too: the target copy of
                # w,b, the same scale the target critic used.
                state_values = calibrated_soft_value(
                    target_logits_1,
                    target_logits_2,
                    torch.ones_like(target_logits_1, dtype=torch.bool),
                    alpha,
                    agent.top_k_actions,
                    agent.q_scale_target,
                    agent.q_bias_target,
                ).reshape(-1)
            else:
                v1, v2 = masked_soft_value(
                    target_logits_1,
                    target_logits_2,
                    torch.ones_like(target_logits_1, dtype=torch.bool),
                    alpha,
                    agent.top_k_actions,
                )
                state_values = (v1 + v2).reshape(-1)

            for env_id in range(num_envs):
                values[env_id].append(float(state_values[env_id]))

            if collected >= batch_size:
                # The bootstrap V(s_{t+1}) is collected; the rollout ends here.
                break

            positions, injected = self.policy(
                pool.scores,
                gte_pool.scores,
                evaluate=evaluate,
            )
            chosen_ids = torch.where(
                injected,
                gte_pool.row_ids.gather(1, positions[:, None]).squeeze(1),
                pool.row_ids.gather(1, positions[:, None]).squeeze(1),
            )
            # The action vector is looked up by row id, not taken from the
            # pool: an injected action has no slot in the online pool.
            action_vectors = self.index.rows(chosen_ids)
            title_ids = self.index.titles_of(chosen_ids)

            monitor.observe_step(
                self.envs,
                pool,
                gte_pool,
                chosen_ids,
                injected,
                s_online,
                s_gte,
                self.index.titles_of(pool.row_ids),
                self.index.titles_of(gte_pool.row_ids),
            )

            results = self._apply_steps(
                chosen_ids.tolist(), title_ids.tolist()
            )

            state_memories = make_state_memories(texts)
            for env_id, (env, (_, step)) in enumerate(zip(self.envs, results)):
                states[env_id].append(state_memories[env_id])
                actions[env_id].append(action_vectors[env_id])
                rewards[env_id].append(step.reward)
                not_done[env_id].append(0 if step.done else 1)
                valid[env_id].append(step.valid)
                self.episodic_returns[env_id] += step.reward
                collected += 1

                if not step.valid:
                    # Reward arrives only at the terminal step, so a missing
                    # one poisons the λ-returns of the whole episode, not just
                    # one transition: the entire episode is dropped from the loss.
                    for index in range(episode_start[env_id], len(valid[env_id])):
                        valid[env_id][index] = False

                if step.done:
                    monitor.observe_episode(env, self.episodic_returns[env_id])
                    episode_returns.append(float(self.episodic_returns[env_id]))
                    self.episodic_returns[env_id] = 0.0
                    episode_start[env_id] = len(valid[env_id])
                    env.reset()

        flat_states = [memory for env_states in states for memory in env_states]
        state_stack = stack_memory(
            flat_states,
            self.state_tokenizer,
            max_length=self.max_state_segment_length,
            device=self.device,
        )
        action_stack = torch.stack(
            [vector for env_actions in actions for vector in env_actions]
        )
        train_batch = SearchTrainBatch(
            state=state_stack,
            action=action_stack,
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            not_done=torch.tensor(not_done, dtype=torch.int32, device=self.device),
            q_values=torch.tensor(values, dtype=torch.float32, device=self.device),
            valid=torch.tensor(valid, dtype=torch.bool, device=self.device),
        )
        return episode_returns, train_batch, monitor.summary(self.envs)


class _Monitor:
    """Training-health counters.

    Line A has no safeguards, so these numbers are the only way to tell "the
    model is learning" from "the tower drifted and the pool collapsed".
    """

    def __init__(self) -> None:
        self.steps = 0
        self.injected = 0
        self.masked_argmax = 0
        self.logit_spread = 0.0
        self.state_norm = 0.0
        self.cosine_to_gte = 0.0
        self.action_in_gte_pool = 0
        self.gte_rank_sum = 0
        self.gold_recall_own = 0.0
        self.gold_recall_gte = 0.0
        self.gold_recall_steps = 0
        self.episodes = 0
        self.episodes_covered = 0
        self.reward_covered = 0.0
        self.reward_uncovered = 0.0
        self.second_chunks = 0
        self.selected_chunks = 0
        self.em = 0.0
        self.em_alias = 0.0
        self.judged = 0
        self.judge_correct = 0.0

    @staticmethod
    def _gold_title_recall(env, pool_titles: set[int]) -> float | None:
        """Share of the episode's gold titles that made it into the pool.

        Computed both for the own pool and for the GTE top-K of the same state:
        without the second number the first says nothing, since a drop can mean
        either a collapsed pool or just a hard question.

        ``None`` for examples without gold titles, which are not counted: the
        NQ half of the raw mix has none. So on the mix
        ``pool/gold_title_recall`` is recall over HotpotQA episodes, not the
        whole set; their share is shown in ``pool/gold_title_coverage``.
        """
        gold = env.sample.get("gold_title_ids")
        if not gold:
            return None
        return sum(1 for title_id in gold if title_id in pool_titles) / len(gold)

    def observe_step(
        self,
        envs,
        pool,
        gte_pool,
        chosen_ids,
        injected,
        s_online,
        s_gte,
        pool_titles,
        gte_pool_titles,
    ):
        batch = len(envs)
        self.steps += batch
        self.injected += int(injected.sum())
        self.masked_argmax += int(pool.masked_argmax.sum())
        self.logit_spread += float(pool.scores.std(dim=-1).sum())
        self.state_norm += float(s_online.norm(dim=-1).sum())
        self.cosine_to_gte += float(
            torch.nn.functional.cosine_similarity(s_online, s_gte, dim=-1).sum()
        )

        # Mean GTE rank of the chosen action: censored at top-K, so it is
        # always logged together with the share of actions inside GTE top-K.
        in_gte = gte_pool.row_ids == chosen_ids[:, None]
        present = in_gte.any(dim=1)
        self.action_in_gte_pool += int(present.sum())
        if bool(present.any()):
            ranks = in_gte.float().argmax(dim=1)
            self.gte_rank_sum += int(ranks[present].sum())

        own = pool_titles.tolist()
        gte = gte_pool_titles.tolist()
        for env_id, env in enumerate(envs):
            recall_own = self._gold_title_recall(env, set(own[env_id]))
            if recall_own is None:
                continue
            self.gold_recall_own += recall_own
            self.gold_recall_gte += self._gold_title_recall(env, set(gte[env_id]))
            self.gold_recall_steps += 1

    def observe_episode(self, env, episode_return: float) -> None:
        self.episodes += 1
        self.selected_chunks += len(env.selected_rows)
        self.second_chunks += env.second_chunk_count()
        # Reward is the max of EM over answer aliases and the judge verdict.
        # The two parts are accumulated separately: otherwise a rising curve
        # looks the same whether retrieval improved or the judge got lenient.
        getter = getattr(env.feedback_model, "get_metrics", None)
        metrics = getter() if getter is not None else {}
        self.em += float(metrics.get("EM", 0.0))
        self.em_alias += float(metrics.get("em_alias", 0.0))
        verdict = metrics.get("judge")
        if verdict is not None:
            self.judged += 1
            self.judge_correct += float(verdict)
        covered = env.gold_titles_covered
        if covered:
            self.episodes_covered += 1
            self.reward_covered += episode_return
        elif covered is not None:
            self.reward_uncovered += episode_return

    def summary(self, envs) -> dict[str, float]:
        steps = max(self.steps, 1)
        episodes = max(self.episodes, 1)
        uncovered = max(self.episodes - self.episodes_covered, 1)
        stats = {
            "explore/injected_share": self.injected / steps,
            "explore/masked_argmax_share": self.masked_argmax / steps,
            "policy/logit_spread": self.logit_spread / steps,
            "state/norm": self.state_norm / steps,
            "state/cosine_to_gte": self.cosine_to_gte / steps,
            "pool/action_in_gte_top_k": self.action_in_gte_pool / steps,
            "pool/gte_rank_of_action": (
                self.gte_rank_sum / max(self.action_in_gte_pool, 1)
            ),
            "pool/second_chunk_share": (
                self.second_chunks / max(self.selected_chunks, 1)
            ),
            "pool/gold_title_recall": (
                self.gold_recall_own / max(self.gold_recall_steps, 1)
            ),
            "pool/gold_title_recall_gte": (
                self.gold_recall_gte / max(self.gold_recall_steps, 1)
            ),
            # Share of steps where recall was computed at all: on the raw mix
            # this is the HotpotQA share, and without it the two curves above
            # read as a statement about the whole set.
            "pool/gold_title_coverage": self.gold_recall_steps / steps,
            "reward/em": self.em / episodes,
            "reward/em_alias": self.em_alias / episodes,
            "reward/judge_rescue": self.judge_correct / episodes,
            "reward/judged_share": self.judged / episodes,
            "reward/covered": self.reward_covered / max(self.episodes_covered, 1),
            "reward/uncovered": self.reward_uncovered / uncovered,
            "reward/covered_share": self.episodes_covered / episodes,
        }
        errors = 0
        requests = 0
        failing = False
        for env in envs:
            error_stats = getattr(env.feedback_model, "error_stats", None)
            if error_stats is None:
                continue
            values = error_stats()
            errors += int(values["vllm_errors"])
            requests += int(values["vllm_requests"])
            failing = failing or bool(values["vllm_failing"])
        stats["reward/vllm_errors"] = errors
        stats["reward/vllm_error_rate"] = errors / max(requests, 1)
        stats["reward/vllm_failing"] = float(failing)
        return stats
