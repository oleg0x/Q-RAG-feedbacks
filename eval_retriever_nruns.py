import os
import sys
import argparse
import random
import json
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import compose, initialize
from tqdm import tqdm

# ---- add repository root to PYTHONPATH (so that rl.* modules resolve) ---- #
repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from rl.agents.pqn import PQN  # noqa: E402
from envs.qa_env import QAEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Seed everything (Python, NumPy, PyTorch, CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def prepare_eval_config(eval_cfg, train_cfg):
#     """Compute additional complex value changes"""
#     # Override only test‑environment‑specific fields so that
#     # the training configuration remains untouched.
#     max_chunks_count = eval_cfg.envs.get("max_chunks_count", None)
#     max_seq_len = eval_cfg.algo.model.predictor.get("max_seq_len", None)
#     interpolate_factor = eval_cfg.algo.model.predictor.get("interpolate_factor", None)
#
#     assert max_chunks_count == max_seq_len == interpolate_factor == None, \
#         'If you specified at least one of the [envs.max_chunks_count, algo.model.predictor.max_seq_len, algo.model.predictor.interpolate_factor] then you should also specify others'
#
#     eval_max_chunks = eval_cfg.envs.num_sentences
#     if train_cfg.index_type == 'random':
#             train_max_chunks = train_cfg.envs.max_chunks_count
#     elif train_cfg.index_type == 'absolute':
#             train_max_chunks = train_cfg.envs.num_sentences
#
#     if eval_max_chunks > train_max_chunks:
#         eval_cfg.envs.max_chunks_count = eval_max_chunks + 1
#         eval_cfg.algo.model.predictor.max_seq_len = max(eval_max_chunks + 1, train_cfg.algo.model.predictor.max_seq_len)
#         eval_cfg.algo.model.predictor.interpolate_factor = eval_max_chunks / train_max_chunks
#         print(f'Current indexing type is {train_cfg.index_type}')
#         print(
#             "The following parameters are updated:",
#             f"...eval_cfg.envs.max_chunks_count={eval_cfg.envs.max_chunks_count}",
#             f"...eval_cfg.algo.model.predictor.max_seq_len={eval_cfg.algo.model.predictor.max_seq_len}",
#             f"...eval_cfg.algo.model.predictor.interpolate_factor={eval_cfg.algo.model.predictor.interpolate_factor}",
#             sep='\n')
#
#     return eval_cfg

def calc_fact_f1_em(predicted_support_idxs, gt_support_idxs):
    # Taken from hotpot_eval
    pred_sf = set(map(int, predicted_support_idxs))
    gt_sf = set(map(int, gt_support_idxs))
    tp, fp, fn = 0, 0, 0
    for e in pred_sf:
        if e in gt_sf:
            tp += 1
        else:
            fp += 1
    for e in gt_sf:
        if e not in pred_sf:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if gt_sf.issubset(pred_sf) else 0.0

    # In case everything is empty, set both f1, em to be 1.0.
    # Without this change, em gets 1 and f1 gets 0
    if not pred_sf and not gt_sf:
        f1, em = 1.0, 1.0
    return f1, em


def config_group_is_overridden(overrides: List[str], group: str) -> bool:
    """Return whether a Hydra config group was explicitly selected on CLI."""
    prefix = f"{group}="
    return any(override.lstrip("+").startswith(prefix) for override in overrides)


def merge_eval_config(train_cfg, eval_cfg, overrides: List[str]):
    """Merge eval settings, replacing explicitly selected config groups."""
    cfg = OmegaConf.merge(train_cfg, eval_cfg)

    # A selected evaluation environment is a complete config, not a patch for
    # the environment used during training. Recursive merging would otherwise
    # leave stale keys such as dataset1/dataset2 from a combined environment.
    if config_group_is_overridden(overrides, "envs"):
        OmegaConf.update(cfg, "envs", eval_cfg.envs, merge=False)

    return cfg


@torch.no_grad()
def evaluate_episode(env: QAEnv, agent: PQN, sample=None, num_runs: int = 1) -> List[dict]:
    """Run K stochastic episodes and return list of trajectory metrics."""
    results = []

    for run_idx in range(num_runs):
        state = env.reset(sample) if sample is not None else env.reset()

        text_len = env.get_sample_len(agent.action_tokenizer)

        embeds, embeds_target = env.get_extra_embeds(
            agent.action_tokenizer,
            agent.critic.action_embed,
            agent.action_embed_target,
        )

        episode_return = 0.0
        num_steps = 0
        q_values = []
        done = False

        while not done:
            embeds = env.update_embeds(embeds, agent.critic.action_embed)
            embeds_target = env.update_embeds(embeds_target, agent.action_embed_target)

            action, q, _ = agent.select_action(state,
                embeds["rope"], embeds_target["rope"],
                random = False, evaluate = (num_runs == 1)
            )

            state, _, reward, done = env.step(action)
            episode_return += reward
            q_max = q.max()
            q_values.append(float(q_max))
            num_steps += 1

        pred_sf = [int(i) for i in state.item_ids]
        if sample is not None:
            pred_texts = [sample["chunks"][idx] for idx in pred_sf]
        else:
            pred_texts = []
        gt_sf = list(env.references_idx)
        f1, em = calc_fact_f1_em(pred_sf, gt_sf)

        results.append({
            'run_id': run_idx,
            'text_len': text_len,
            'return': episode_return,
            'f1': f1,
            'em': em,
            'q_values': q_values,
            'pred_idx': pred_sf,
            'pred_texts': pred_texts,
        })

    return results


# def _drop_eval_incompatible_dataset_keys(cfg):
#     dataset_cfg = cfg.envs.test_env.dataset.get("dataset", None)
#     if dataset_cfg is None:
#         return
#
#     target = str(dataset_cfg.get("_target_", ""))
#     if target.endswith("RetrievalHotPotQA") and "skip_bridge_comparison" in dataset_cfg:
#         was_struct = OmegaConf.is_struct(dataset_cfg)
#         OmegaConf.set_struct(dataset_cfg, False)
#         del dataset_cfg["skip_bridge_comparison"]
#         OmegaConf.set_struct(dataset_cfg, was_struct)


def load_eval_config(name):
    overrides = sys.argv[1:]
    with initialize(version_base="1.3", config_path="./configs"):
        eval_cfg = compose(
            config_name=name,
            overrides=overrides
        )
        # cli_cfg = OmegaConf.from_cli()
        # eval_cfg = OmegaConf.load(name)
        # eval_cfg = OmegaConf.merge(eval_cfg, cli_cfg)

    train_cfg_path = os.path.join(eval_cfg.pretrained_path, 'config.yaml')
    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(f"Could not find config.yaml at {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)
    #prepare_eval_config(eval_cfg, train_cfg)
    cfg = merge_eval_config(train_cfg, eval_cfg, overrides)
    #_drop_eval_incompatible_dataset_keys(cfg)
    OmegaConf.resolve(cfg)
    return cfg, train_cfg


def main(argv: List[str] | None = None) -> None:
    cfg, train_cfg = load_eval_config("testing.yaml")
    # Set global MAX_TOKEN_LENGTH constants before tokenisers are built
    # MAX_TOKEN_LENGTH["state"] = cfg.max_state_length
    # MAX_TOKEN_LENGTH["action"] = cfg.max_action_length
    num_runs = getattr(cfg, 'num_runs', 1)

    set_all_seeds(cfg.seed)

    # Respect the device stored in the training config; fall back to CPU if absent
    print("device", getattr(cfg, "device", "cpu"))
    torch.set_default_device(getattr(cfg, "device", "cpu"))
    torch.set_float32_matmul_precision("high")

    # -----------------------------------------------------------------------
    # Build agent & load checkpoint
    # -----------------------------------------------------------------------
    agent = PQN(cfg.algo)

    if 'GTE_pure' in cfg.pretrained_path:
        ckpt_filename = "model_pretrained.pt"
    else:
        ckpt_filename = "model_last.pt" if cfg.use_last else "model_best.pt"

    ckpt_path = os.path.join(cfg.pretrained_path, ckpt_filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    agent.load(ckpt_path, strict=True)
    agent.eval()

    env_test: QAEnv = instantiate(cfg.envs.test_env)

    max_samples = len(env_test.dataset)
    if not (0 < cfg.num_samples <= max_samples):
        print(f'set num samples from {cfg.num_samples} to {max_samples}')
        cfg.num_samples = max_samples

    # -----------------------------------------------------------------------
    # Evaluate with logging
    # -----------------------------------------------------------------------
    if cfg.log_filename is not None:
        log_name = cfg.log_filename
    elif cfg.envs.task not in ['hotpotqa', 'musique']:
        log_name = f"eval_seed{cfg.seed}_ns{cfg.envs.num_sentences}.jsonl"
    else:
        log_name = f"eval_seed{cfg.seed}.jsonl"

    log_path = os.path.join(cfg.pretrained_path, log_name)

    print(f"Writing eval log to {log_path}")

    existing = {}
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    existing[item["id"]] = item
                except Exception:
                    continue

    returns = []
    text_lens = []
    all_em = []
    all_f1 = []
    num_steps = []
    num_steps_list = []

    # metrics for already processed samples
    for item in existing.values():
        # f1, em = calc_fact_f1_em(item["pred_idx"], item["sf_idx"])
        # all_f1.append(f1)
        # all_em.append(em)
        # if "return" in item:
        #     returns.append(item["return"])
        # if "text_len" in item:
        #     text_lens.append(item["text_len"])

        if "runs" in item and len(item["runs"]) > 0:
            em = max(r["em"] for r in item["runs"])
            f1 = max(r["f1"] for r in item["runs"])
            mean_ret = np.mean([r["return"] for r in item["runs"]])
            mean_len = np.mean([r["text_len"] for r in item["runs"]])
            #mean_steps = np.mean([r["num_steps"] for r in item["runs"]])
        else:
            f1, em = calc_fact_f1_em(item["pred_idx"], item["sf_idx"])
            mean_ret = item.get("return", 0.0)
            mean_len = item.get("text_len", 0)
            #mean_steps = item.get("num_steps", 0)

        all_em.append(em)
        all_f1.append(f1)
        returns.append(mean_ret)
        text_lens.append(mean_len)
        #num_steps_list.append(mean_steps)

    f = open(log_path, "a")

    for i in tqdm(range(cfg.num_samples), desc="Evaluating", ncols=80):
        sample = env_test.dataset[i]
        if sample["id"] in existing:
            continue

        run_results = evaluate_episode(env_test, agent, sample, num_runs)

        best_em = max(r['em'] for r in run_results)
        best_f1 = max(r['f1'] for r in run_results)
        mean_return = float(np.mean([r['return'] for r in run_results]))
        mean_text_len = float(np.mean([r['text_len'] for r in run_results]))

        returns.append(mean_return)
        text_lens.append(mean_text_len)
        all_em.append(best_em)
        all_f1.append(best_f1)
        best_run = max(run_results, key=lambda r: (r['em'], r['f1']))

        entry = {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "sf_idx": [int(idx) for idx in sample["sf_idx"]],
            "pred_idx": best_run["pred_idx"],
            "sf_texts": [sample["chunks"][idx] for idx in sample["sf_idx"]],
            "pred_texts": [sample["chunks"][idx] for idx in best_run["pred_idx"]],
            "return": mean_return,
            "text_len": mean_text_len,
            "em": best_em,
            "f1": best_f1,
            "runs": run_results,
        }

        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()

    f.close()

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    fact_em = sum(all_em) / len(all_em)
    fact_f1 = sum(all_f1) / len(all_f1)
    total_eval = len(all_em)

    print(
        f"Evaluated on {total_eval} episodes, max_retrieves={cfg.envs.max_steps}, "
        f"num_runs={num_runs} | "
        f"Mean return: {mean_return:.3f} ± {std_return:.3f} (std) | "
        f"Mean text len: {np.mean(text_lens):.2f} | "
        f"EM (best-of-{num_runs}): {fact_em:.3f} | F1 (best-of-{num_runs}): {fact_f1:.3f}"
    )


if __name__ == "__main__":
    main()
