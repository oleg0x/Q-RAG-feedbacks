"""Line A training: Q-RAG as an iterative dense retriever over Wiki-18.

The difference from ``train_q_rag.py`` is not in the training loop but in where
actions come from. Here they come from searching ``s @ M.T`` over all
21,015,324 rows rather than from a candidate list in the dataset, so the
objects are assembled differently:

* the matrix ``M`` (``wiki18-gte`` shards) stays resident on the GPU, 60.1 GiB;
* the title table is needed for the quota ``N`` and the episode coverage flag;
* the corpus is read through an offset table to get the selected chunk's text;
* environments are stepped in a batch, not a loop (see
  ``envs/parallel_search_env.py``).

Usage:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /path/to/venv/bin/python train_q_rag_search.py
"""

import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from envs.action_index import ActionIndex
from envs.corpus_reader import CorpusReader
from envs.parallel_search_env import ParallelSearchEnv
from envs.search_env import (
    DenseSearchEnv,
    SearchDatasetAdapter,
    load_weights,
    make_search_envs,
)
from envs.title_table import TitleTable
from rl.agents.pqn import PQN, AlphaSchedule
from rl.feedback import VllmUnavailableError
from rl.q_module import SearchBoltzmannPolicy


def split_config_name(argv: list[str], default: str) -> tuple[str, list[str]]:
    """Extract ``--config-name`` from the arguments; the rest are overrides.

    ``compose`` takes the config name as a parameter, not an override, and a
    hard-coded name would mean a second training config could only be run by
    editing the source. Parsed by hand rather than via ``@hydra.main``: the
    script composes the config itself on purpose to add the log directory.
    """
    name = default
    overrides = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item.startswith("--config-name="):
            name = item.split("=", 1)[1]
        elif item == "--config-name":
            index += 1
            if index >= len(argv):
                raise SystemExit("--config-name requires a value")
            name = argv[index]
        else:
            overrides.append(item)
        index += 1
    return name, overrides


def load_config(name: str = "training_fullwiki_search.yaml") -> DictConfig:
    name, overrides = split_config_name(sys.argv[1:], name)
    with initialize(version_base="1.3", config_path="./configs"):
        cfg = compose(config_name=name, overrides=overrides)
        if cfg.logger.log_dir is not None:
            directory = (
                datetime.now().strftime("%b%d_%H-%M-%S")
                + cfg.logger.tensorboard.comment
            )
            cfg.logger.log_dir = os.path.join(cfg.logger.log_dir, directory)
            cfg.logger.tensorboard.log_dir = os.path.join(cfg.logger.log_dir, "tb_logs/")
        return cfg


def stamp_manifest(cfg: DictConfig) -> None:
    """Record in the config what it does not show: the reward model and GPUs.

    ``base_url`` and ``model`` come from environment variables, and
    ``config.yaml`` is saved with interpolations unresolved, so the saved file
    alone does not tell which model computed the reward or which GPU the run
    used. Compared runs must match in everything except the reward model, and
    this has to be checked from the run's file, not from memory.
    """
    feedback = cfg.feedback.feedback_dict[cfg.feedback.type]
    OmegaConf.update(
        cfg,
        "manifest",
        {
            "reward_model": str(feedback.model),
            "reward_base_url": str(feedback.base_url),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(cfg.device),
            "started_utc": datetime.utcnow().isoformat() + "Z",
        },
        force_add=True,
    )


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_index(cfg: DictConfig, title_table: TitleTable) -> ActionIndex:
    return ActionIndex.from_shards(
        cfg.envs.index.shard_dir,
        device=cfg.device,
        expected_rows=int(cfg.envs.index.rows),
        dim=int(cfg.envs.index.dim),
        title_ids=title_table.title_ids,
    )


def build_envs(cfg: DictConfig, title_table: TitleTable):
    raw_dataset = instantiate(cfg.envs.train_dataset)
    dataset = SearchDatasetAdapter(raw_dataset, title_table)
    weights_path = cfg.envs.get("train_weights")
    weights = None
    if weights_path is not None:
        weights = load_weights(weights_path, dataset)
        drawn = int((weights > 0).sum())
        print(
            f"Episode weights: {weights_path}, {drawn} of {len(dataset)} samples "
            f"drawable ({len(dataset) - drawn} at weight 0)"
        )
    template = DenseSearchEnv(
        dataset=dataset,
        max_steps=int(cfg.envs.max_steps),
        feedback_model=instantiate(cfg.feedback.feedback_dict[cfg.feedback.type]),
        max_chunks_per_title=int(cfg.envs.max_chunks_per_title),
        separator=cfg.envs.separator,
        weights=weights,
    )
    return dataset, make_search_envs(template, int(cfg.envs_parallel), seed=cfg.seed)


def evaluation_samples(cfg: DictConfig, title_table: TitleTable) -> list[dict]:
    raw = instantiate(cfg.envs.eval_dataset)
    dataset = SearchDatasetAdapter(raw, title_table)
    limit = int(cfg.eval_episodes)
    if limit > len(dataset):
        raise ValueError(
            f"eval_episodes={limit} exceeds the eval dataset size={len(dataset)}"
        )
    return [dataset[index] for index in range(limit)]


def log_subset(
    writer: SummaryWriter, prefix: str, subset: list[dict], step: int
) -> None:
    """Reward and both of its parts in one block.

    reward == em_alias + share rescued by the judge, so both are logged side by
    side: without them a rising reward curve is unreadable, as it could come
    from EM or from a lenient judge. Plain EM (on the main answer) is logged
    third: it is the final metric, and its divergence from the reward must be
    visible during the run, not only in a post-hoc analysis.
    """
    writer.add_scalar(
        f"{prefix}/reward", float(np.mean([i["reward"] for i in subset])), step
    )
    writer.add_scalar(f"{prefix}/em", float(np.mean([i["em"] for i in subset])), step)
    writer.add_scalar(
        f"{prefix}/em_alias", float(np.mean([i["em_alias"] for i in subset])), step
    )
    writer.add_scalar(
        f"{prefix}/judge_rescue",
        float(np.mean([i["reward"] - i["em_alias"] for i in subset])),
        step,
    )


def log_evaluation(writer: SummaryWriter, results: list[dict], step: int) -> float:
    """Eval curves and the value used to select ``model_best``.

    Selection uses the **mean of the per-source curves**, not the overall mean:
    the halves of the mix differ in size and difficulty, and a micro-average
    would pick the checkpoint by whichever half dominates the holdout.
    """
    log_subset(writer, "eval", results, step)
    for covered in (True, False):
        subset = [
            item for item in results if item["gold_titles_covered"] is covered
        ]
        if not subset:
            continue
        name = "covered" if covered else "uncovered"
        writer.add_scalar(
            f"eval/{name}/reward",
            float(np.mean([item["reward"] for item in subset])),
            step,
        )
        writer.add_scalar(
            f"eval/{name}/em",
            float(np.mean([item["em"] for item in subset])),
            step,
        )
    invalid = sum(1 for item in results if not item.get("valid", True))
    writer.add_scalar("eval/invalid_episodes", invalid, step)

    per_source = []
    for source in sorted({item["source"] for item in results}):
        subset = [item for item in results if item["source"] == source]
        log_subset(writer, f"eval/{source}", subset, step)
        per_source.append(float(np.mean([item["reward"] for item in subset])))
    macro = float(np.mean(per_source))
    writer.add_scalar("eval/reward_macro", macro, step)
    return macro


def main() -> int:
    cfg = load_config()

    stamp_manifest(cfg)
    writer: SummaryWriter = instantiate(cfg.logger.tensorboard)
    os.makedirs(cfg.logger.log_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.logger.log_dir, "config.yaml"), resolve=False)
    ckpt_last_path = os.path.join(cfg.logger.log_dir, "model_last.pt")
    ckpt_best_path = os.path.join(cfg.logger.log_dir, "model_best.pt")
    best_eval_reward = -float("inf")

    torch.set_default_device(cfg.device)
    torch.set_float32_matmul_precision("high")
    set_all_seeds(cfg.seed)

    agent = PQN(cfg.algo)
    if agent.top_k_actions != int(cfg.envs.top_k):
        # The policy pool must equal the search pool: otherwise V is computed
        # over part of the candidates while the action is chosen from all.
        raise ValueError(
            f"pqn.hyperparams.top_k_actions={agent.top_k_actions} must equal "
            f"envs.top_k={cfg.envs.top_k}"
        )

    title_table = TitleTable.load(
        cfg.envs.title_table,
        expected_rows=int(cfg.envs.index.rows),
        load_titles=True,
    )
    index = build_index(cfg, title_table)
    dataset, envs = build_envs(cfg, title_table)
    eval_samples = evaluation_samples(cfg, title_table)
    # Title names are not needed from here on: masking and monitoring use
    # integer ids, and the list of strings takes hundreds of megabytes.
    title_table.titles = None

    parallel_env = ParallelSearchEnv(
        envs=envs,
        index=index,
        corpus=CorpusReader(
            cfg.envs.corpus, cfg.envs.offsets, expected_rows=int(cfg.envs.index.rows)
        ),
        state_tokenizer=agent.state_tokenizer,
        max_state_segment_length=int(cfg.max_action_length_in_memory),
        top_k=int(cfg.envs.top_k),
        policy=SearchBoltzmannPolicy(
            epsilon=float(cfg.envs.exploration.epsilon),
            temperature=float(cfg.envs.exploration.temperature.start),
            injection_sampling=str(cfg.envs.exploration.injection_sampling),
        ),
        max_workers=int(cfg.envs.feedback_workers),
        device=cfg.device,
    )

    # The exploration temperature has its own schedule, tied neither to lr
    # nor to the critic's α: its units differ (logits are already divided by
    # their spread).
    temperature_schedule = AlphaSchedule(
        start=float(cfg.envs.exploration.temperature.start),
        kind=str(cfg.envs.exploration.temperature.kind),
        final=cfg.envs.exploration.temperature.get("final"),
        warmup=int(cfg.envs.exploration.temperature.get("warmup", 0)),
        total=cfg.envs.exploration.temperature.get("total"),
    )

    total_steps = cfg.steps_count * cfg.accumulate_grads
    eval_interval = cfg.eval_interval * cfg.accumulate_grads
    progress_bar = tqdm(range(total_steps), desc="Training (search)")

    parallel_env.reset()
    step = 0
    train_rewards: list[float] = []

    try:
        for iteration in progress_bar:
            agent.train()
            parallel_env.policy.temperature = temperature_schedule.value(
                agent._optim_step
            )
            rewards, train_batch, stats = parallel_env.rollout(
                cfg.batch_size,
                agent,
                online_models_train_mode=bool(cfg.envs.rollout_train_mode),
            )
            step += train_batch.reward.numel()
            train_rewards.extend(rewards)

            qf_loss = agent.update(
                train_batch.state,
                train_batch.action,
                None,
                train_batch.q_values,
                train_batch.reward,
                train_batch.not_done,
                train_batch.valid,
            )

            if iteration % eval_interval == 0:
                agent.eval()
                if train_rewards:
                    writer.add_scalar("train r_sum", float(np.mean(train_rewards)), step)
                writer.add_scalar("qf_loss", qf_loss, step)
                writer.add_scalar("train/alpha", agent.alpha, step)
                writer.add_scalar(
                    "train/exploration_temperature", parallel_env.policy.temperature, step
                )
                if agent.critic.calibrated:
                    # w going to zero is an early sign of the same collapse
                    # seen in training without calibration: watch this curve.
                    writer.add_scalar(
                        "train/q_scale", float(agent.critic.q_scale), step
                    )
                    writer.add_scalar(
                        "train/q_bias", float(agent.critic.q_bias), step
                    )
                for name, value in stats.items():
                    writer.add_scalar(name, value, step)
                # Stats accumulate since the last log write, not the last
                # rollout: episodes do not finish in every call, and without
                # accumulation episode curves were systematically empty.
                parallel_env.reset_monitor()

                results = parallel_env.run_episodes(agent, eval_samples)
                mean_eval_reward = log_evaluation(writer, results, step)

                progress_bar.set_postfix({
                    "reward": float(np.mean(train_rewards)) if train_rewards else 0.0,
                    "eval_reward": mean_eval_reward,
                    "qf_loss": qf_loss,
                    "step": step,
                })
                agent.save(ckpt_last_path)
                if mean_eval_reward > best_eval_reward:
                    best_eval_reward = mean_eval_reward
                    agent.save(ckpt_best_path)
                train_rewards = []
    except VllmUnavailableError as error:
        # The server has been silent past the threshold: continuing would log
        # hours of empty curves. A checkpoint is saved so the run can resume.
        agent.save(ckpt_last_path, verbose=True)
        print(f"[ERROR] {error}")
        return 1
    finally:
        parallel_env.close()
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
