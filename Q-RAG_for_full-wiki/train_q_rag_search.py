"""Обучение линии A: Q-RAG как итеративный плотный ретривер по Wiki-18.

Отличия от ``train_q_rag.py`` — не в цикле обучения, а в том, откуда берутся
действия. Здесь их даёт поиск ``s @ M.T`` по всем 21 015 324 строкам, а не
список кандидатов из датасета, поэтому сборка объектов другая:

* матрица ``M`` (шарды ``wiki18-gte``) резидентно лежит на GPU, 60.1 ГиБ;
* таблица титулов нужна для квоты ``N`` и для флага покрытия эпизода;
* корпус читается по таблице смещений — за текстом выбранного чанка;
* среды шагаются батчем, а не циклом (см. ``envs/parallel_search_env.py``).

Запуск:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /home/a.anokhin/venvs/gpu/bin/python train_q_rag_search.py
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
    """Вынуть ``--config-name`` из аргументов, остальное отдать overrides.

    ``compose`` берёт имя конфига параметром, а не оверрайдом, и хардкод
    имени означал, что вторая ветка обучения запускается только правкой
    исходника. Отдельный разбор, а не ``@hydra.main``: скрипт сознательно
    собирает конфиг сам, чтобы дописать в него каталог лога.
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
                raise SystemExit("--config-name без значения")
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
    """Записать в конфиг то, чего в нём не видно: модель награды и карты.

    ``base_url`` и ``model`` приходят переменными окружения, а ``config.yaml``
    сохраняется без резолва интерполяций — по сохранённому файлу нельзя
    сказать, какой моделью считалась награда и на какой карте шёл ран. Руки
    ветки обязаны совпадать во всём, кроме модели награды, и сверять это надо
    по файлу рана, а не по памяти запускавшего.
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
    """Награда и обе её половины одним блоком.

    reward == em_alias + доля спасённых судьёй, поэтому обе величины пишутся
    рядом: без них рост кривой награды нечитаем — она могла вырасти и по EM,
    и по мягкости судьи. Чистый EM (по основному ответу) идёт третьим: это
    итоговая метрика ветки, и её расхождение с наградой обязано быть видно по
    ходу рана, а не в разборе потом.
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
    """Кривые eval и величина, по которой выбирается ``model_best``.

    Отбор идёт по **среднему двух кривых**, а не по общей средней: половины
    смеси разного размера и разной трудности, и микро-среднее означало бы
    выбор чекпоинта по той половине, которой в holdout больше. Ровно на этом
    погорела прошлая ветка — eval шёл только по HotpotQA при обучении на смеси.
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
        # Пул политики обязан совпадать с пулом поиска: иначе V считается по
        # части кандидатов, а действие выбирается из всех.
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
    # Имена титулов дальше не нужны: маскирование и мониторинг работают с
    # целыми идентификаторами, а список строк весит сотни мегабайт.
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

    # Температура exploration — своё расписание, не привязанное ни к lr, ни к
    # α критика: у неё другие единицы (логиты уже поделены на свой разброс).
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
                    # w, уходящий к нулю, — ранний признак того же распада,
                    # что убил переобучение без калибровки: следить глазами.
                    writer.add_scalar(
                        "train/q_scale", float(agent.critic.q_scale), step
                    )
                    writer.add_scalar(
                        "train/q_bias", float(agent.critic.q_bias), step
                    )
                for name, value in stats.items():
                    writer.add_scalar(name, value, step)
                # Статистика копится с прошлой записи, не с последнего rollout:
                # эпизоды завершаются не в каждом вызове, и без накопления
                # эпизодные кривые на точках записи были систематически пусты.
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
        # Сервер молчит дольше порога: продолжать значит писать в лог часы
        # пустых кривых. Чекпоинт сохраняется, чтобы ран можно было поднять.
        agent.save(ckpt_last_path, verbose=True)
        print(f"[ERROR] {error}")
        return 1
    finally:
        parallel_env.close()
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
