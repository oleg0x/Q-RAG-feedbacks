"""Evaluate a Q-ICL retriever checkpoint with its configured feedback model."""

import argparse
import json
import os
import random

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from rl.agents.pqn import PQN


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_sample(env, agent, sample):
    state = env.reset(new_sample=sample)
    done = False
    action_embeds, target_embeds = env.get_extra_embeds(
        agent.action_tokenizer,
        agent.critic.action_embed,
        agent.action_embed_target,
    )
    reward_sum = 0.0
    while not done:
        action_embeds = env.update_embeds(
            action_embeds, agent.critic.action_embed
        )
        target_embeds = env.update_embeds(
            target_embeds, agent.action_embed_target
        )
        action, _, _ = agent.select_action(
            state,
            action_embeds["rope"],
            target_embeds["rope"],
            random=False,
            evaluate=True,
        )
        state, _, reward, done = env.step(action)
        reward_sum += reward

    observation, info = env._make_obs_and_info()
    metrics = {
        "id": sample["id"],
        "source": sample.get("source"),
        "question": observation["question"],
        "answer": info["answer"],
        "pred_idx": observation["pred_idx"],
        "reward": reward_sum,
    }
    if hasattr(env.feedback_model, "get_metrics"):
        metrics.update(env.feedback_model.get_metrics())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="OmegaConf dotlist override; may be repeated",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.checkpoint_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    if args.device:
        cfg.device = args.device
    OmegaConf.resolve(cfg)

    set_all_seeds(int(cfg.seed))
    torch.set_default_device(cfg.device)
    torch.set_float32_matmul_precision("high")
    agent = PQN(cfg.algo)
    checkpoint_path = os.path.join(
        args.checkpoint_dir, f"model_{args.checkpoint}.pt"
    )
    agent.load(checkpoint_path, strict=True)
    agent.eval()
    env = instantiate(cfg.envs.test_env)

    sample_count = min(args.num_samples, len(env.dataset))
    results = []
    for index in tqdm(range(sample_count), desc="Q-ICL eval"):
        sample = env.dataset[index]
        results.append(evaluate_sample(env, agent, sample))

    rewards = [item["reward"] for item in results]
    summary = {
        "num_samples": len(results),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_em": float(
            np.mean([item["EM"] for item in results if "EM" in item])
        )
        if any("EM" in item for item in results)
        else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as destination:
        json.dump(
            {"summary": summary, "results": results},
            destination,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
