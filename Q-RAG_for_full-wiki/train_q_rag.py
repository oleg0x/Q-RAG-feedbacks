import sys
import os

repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    print(f'add repository dir: {repo_dir}')
    sys.path.append(repo_dir)

from torch.utils.tensorboard import SummaryWriter
import torch
import sys
from rl.agents.pqn import PQN
import numpy as np
from envs.qa_env import QAEnv
from envs.parallel_env import ParallelTextEnv
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from hydra import initialize, compose
import random
from datetime import datetime


@torch.no_grad()
def evaluate(env_test, agent):
    s_t = env_test.reset()
    done_t = False
    a_embeds_t, a_embeds_target_t = env_test.get_extra_embeds(agent.action_tokenizer, agent.critic.action_embed, agent.action_embed_target)
    r_sum_t = 0
    while not done_t:
        a_embeds_t = env_test.update_embeds(a_embeds_t, agent.critic.action_embed)
        a_embeds_target_t = env_test.update_embeds(a_embeds_target_t, agent.action_embed_target)
        
        action_t, _, _ = agent.select_action(s_t, a_embeds_t["rope"], a_embeds_target_t["rope"], random=False, evaluate=True)
        s_t, _, reward_t, done_t = env_test.step(action_t)
        r_sum_t += reward_t
    
    return r_sum_t


def calc_fact_f1_em(predicted_support_idxs, gt_support_idxs):
    pred_sf = set(map(int, predicted_support_idxs))
    gt_sf = set(map(int, gt_support_idxs))
    true_positives = len(pred_sf.intersection(gt_sf))
    false_positives = len(pred_sf - gt_sf)
    false_negatives = len(gt_sf - pred_sf)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    em = 1.0 if gt_sf.issubset(pred_sf) else 0.0
    if not pred_sf and not gt_sf:
        f1, em = 1.0, 1.0
    return f1, em, true_positives


@torch.no_grad()
def evaluate_fixed_sample(env_test, agent, sample):
    s_t = env_test.reset(new_sample=sample)
    done_t = False
    a_embeds_t, a_embeds_target_t = env_test.get_extra_embeds(
        agent.action_tokenizer,
        agent.critic.action_embed,
        agent.action_embed_target,
    )
    r_sum_t = 0.0
    retrieval_steps = 0
    while not done_t:
        a_embeds_t = env_test.update_embeds(
            a_embeds_t, agent.critic.action_embed
        )
        a_embeds_target_t = env_test.update_embeds(
            a_embeds_target_t, agent.action_embed_target
        )
        action_t, _, _ = agent.select_action(
            s_t,
            a_embeds_t["rope"],
            a_embeds_target_t["rope"],
            random=False,
            evaluate=True,
        )
        s_t, _, reward_t, done_t = env_test.step(action_t)
        r_sum_t += reward_t
        retrieval_steps += 1

    fact_f1, fact_em, supporting_facts_found = calc_fact_f1_em(
        s_t.item_ids, env_test.references_idx
    )
    return {
        "reward": r_sum_t,
        "source": sample.get("source", "unknown"),
        "fact_f1": fact_f1,
        "fact_em": fact_em,
        "supporting_facts_found": supporting_facts_found,
        "retrieval_steps": retrieval_steps,
    }


def load_config(name, overrides=None):
    with initialize(version_base="1.3", config_path="./configs"):

        cfg = compose(
            config_name=name,
            overrides=sys.argv[1:] #overrides if overrides else []
        )
        #cli_cfg = OmegaConf.from_cli()
        #cfg = OmegaConf.merge(cfg, cli_cfg)
        cfg = prepare_config(cfg)
        return cfg


def prepare_config(cfg):
    """
    modifies config for parameters that should depend on each other
    """
    if cfg.logger.log_dir is not None:
        dir_name = datetime.now().strftime("%b%d_%H-%M-%S") + cfg.logger.tensorboard.comment
        cfg.logger.log_dir = os.path.join(cfg.logger.log_dir, dir_name)
        cfg.logger.tensorboard.log_dir = os.path.join(cfg.logger.log_dir, 'tb_logs/')

    # enumerate_facts = (cfg.positional_coding == 'enum') #TODO: add version that enumerate all chunks
    # cfg.envs.env.dataset.task_dataset.add_sentence_idx = enumerate_facts
    # cfg.envs.test_env.dataset.task_dataset.add_sentence_idx = enumerate_facts
    return cfg


def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


cfg: DictConfig = load_config(name="training.yaml")
#cfg: DictConfig = load_config(name=os.getenv("Q_RAG_CONFIG_NAME", "training.yaml"))
#cfg: DictConfig = load_config(name="training_hotpotqa.yaml")


writer: SummaryWriter = instantiate(cfg.logger.tensorboard)
os.makedirs(cfg.logger.log_dir, exist_ok=True)
config_save_path = os.path.join(cfg.logger.log_dir, "config.yaml")
OmegaConf.save(config=cfg, f=config_save_path, resolve=False)
print(f"[INFO] Training config saved to {config_save_path}")

agent_config: DictConfig = cfg.algo
env_config: DictConfig = cfg.envs
print("Embedder model:", agent_config.model.model_name)

# path to checkpoints and metric to determine the best model
ckpt_last_path = os.path.join(cfg.logger.log_dir, "model_last.pt")
ckpt_best_path = os.path.join(cfg.logger.log_dir, "model_best.pt")
best_eval_reward = -float("inf")

torch.set_default_device(cfg.device)
torch.set_float32_matmul_precision('high')
set_all_seeds(cfg.seed)

# MAX_TOKEN_LENGTH["state"] = cfg.max_state_length
# MAX_TOKEN_LENGTH["action"] = cfg.max_action_length

agent = PQN(agent_config)

# if bf16:
#     for m in [agent.critic, agent.policy, agent.random_policy,
#               agent.v_net_target, agent.action_embed_target]:
#         m.to(dtype=torch.bfloat16)
#
# if args.fp16:
#     # import apex
#     # apex.amp.register_half_function(torch, 'einsum')
#     from torch.cuda.amp import autocast, GradScaler
#
#     scaler = GradScaler()
#
# device_type = torch.device(cfg.device).type
# amp_dtype = torch.bfloat16 if bf16 else torch.float16
# amp_enabled = bf16 or mixed_precision
# autocast = torch.cuda.amp.autocast if device_type == 'cuda' else torch.autocast

env: QAEnv = instantiate(env_config.env)
env_test: QAEnv = instantiate(env_config.test_env)
parallel_env = ParallelTextEnv(
    [env] + [env.copy() for _ in range(cfg.envs_parallel - 1)], 
    state_tokenizer=agent.state_tokenizer,
    action_tokenizer=agent.action_tokenizer)

total_steps = cfg.steps_count * cfg.accumulate_grads
eval_interval = cfg.eval_interval * cfg.accumulate_grads
#assuming we don't need to scale cfg.learning_start with grad_accumulation
progress_bar = tqdm(range(total_steps), desc="Training")

states_list, _ = parallel_env.reset()
step = 0
train_rewards = []

for it in progress_bar:
    
    agent.train()
    states_list, rewards, train_batch = parallel_env.rollout(
        cfg.batch_size,
        states_list,
        agent,
        random=(step < 2 * cfg.learning_start),
        online_models_train_mode=True,
    )
    step += train_batch.reward.numel()
    #assert train_batch.reward.numel() == np.prod(train_batch.reward.shape)
    train_rewards.extend(rewards)

    qf_loss = agent.update(
        train_batch.state, 
        train_batch.action, 
        train_batch.next_state, 
        train_batch.q_values, 
        train_batch.reward, 
        train_batch.not_done)
    
    if it % eval_interval == 0:

        agent.eval()
        
        writer.add_scalar("train r_sum", np.mean(train_rewards), step)
        writer.add_scalar("qf_loss", qf_loss, step)

        r_eval = []
        eval_strategy = cfg.get("eval_strategy", "random_with_replacement")
        fixed_eval_metrics = []
        if eval_strategy == "random_with_replacement":
            for _ in range(cfg.eval_episodes):
                r_eval.append(evaluate(env_test, agent))
                print(f"\reval prog: {len(r_eval)}/{cfg.eval_episodes}", end="")
        elif eval_strategy == "fixed_without_replacement":
            if cfg.eval_episodes > len(env_test.dataset):
                raise ValueError(
                    f"eval_episodes={cfg.eval_episodes} exceeds fixed eval dataset "
                    f"size={len(env_test.dataset)}"
                )
            seen_sample_keys = set()
            for index in range(cfg.eval_episodes):
                sample = env_test.dataset[index]
                sample_key = (sample.get("source"), sample["id"])
                if sample_key in seen_sample_keys:
                    raise ValueError(f"Duplicate sample in fixed eval: {sample_key}")
                seen_sample_keys.add(sample_key)
                metrics = evaluate_fixed_sample(env_test, agent, sample)
                fixed_eval_metrics.append(metrics)
                r_eval.append(metrics["reward"])
                print(f"\reval prog: {len(r_eval)}/{cfg.eval_episodes}", end="")

            metric_names = (
                "fact_em",
                "fact_f1",
                "supporting_facts_found",
                "retrieval_steps",
            )
            for metric_name in metric_names:
                writer.add_scalar(
                    f"eval/{metric_name}",
                    np.mean([item[metric_name] for item in fixed_eval_metrics]),
                    step,
                )
            for source in sorted(
                {item["source"] for item in fixed_eval_metrics}
            ):
                source_metrics = [
                    item
                    for item in fixed_eval_metrics
                    if item["source"] == source
                ]
                writer.add_scalar(
                    f"eval/{source}/reward",
                    np.mean([item["reward"] for item in source_metrics]),
                    step,
                )
                for metric_name in metric_names:
                    writer.add_scalar(
                        f"eval/{source}/{metric_name}",
                        np.mean(
                            [item[metric_name] for item in source_metrics]
                        ),
                        step,
                    )
                print(
                    f"\n[INFO] Fixed eval source={source}: "
                    f"samples={len(source_metrics)}, "
                    f"reward={np.mean([item['reward'] for item in source_metrics]):.3f}, "
                    f"fact_em={np.mean([item['fact_em'] for item in source_metrics]):.3f}, "
                    f"fact_f1={np.mean([item['fact_f1'] for item in source_metrics]):.3f}"
                )
        else:
            raise ValueError(f"Unknown eval_strategy: {eval_strategy}")

        writer.add_scalar("eval r_sum", np.mean(r_eval), step)

        progress_bar.set_postfix({
                'reward': np.mean(train_rewards),
                "eval_reward": np.mean(r_eval),
                'qf_loss': qf_loss,
                'step': step,
            })
        agent.save(ckpt_last_path)
            #torch.save(agent.state_dict(), ckpt_last_path)

        mean_eval_reward = np.mean(r_eval)
        if mean_eval_reward > best_eval_reward:
            best_eval_reward = mean_eval_reward
            agent.save(ckpt_best_path)
            #torch.save(agent.state_dict(), ckpt_best_path)
            #print(f"[INFO] New best model saved with reward {best_eval_reward:.3f}")

        train_rewards = []

