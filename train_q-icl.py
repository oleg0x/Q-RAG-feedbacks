import logging
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

from envs.parallel_env import ParallelTextEnv
from envs.qa_env import QAEnv
from rl.agents.pqn import PQN



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



def setup_logging(log_file: str) -> None:
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)



setup_logging("log.txt")
cfg: DictConfig = load_config(name="training_gte_hotpotqa.yaml")

writer: SummaryWriter = instantiate(cfg.logger.tensorboard)
os.makedirs(cfg.logger.log_dir, exist_ok=True)
config_save_path = os.path.join(cfg.logger.log_dir, "config.yaml")
OmegaConf.save(config=cfg, f=config_save_path, resolve=False)
print(f"[INFO] Training config saved to {config_save_path}")

agent_config: DictConfig = cfg.algo
env_config: DictConfig = cfg.envs
print("[INFO] Embedder model:", agent_config.model.model_name)

ckpt_last_path = os.path.join(cfg.logger.log_dir, "model_last.pt")
ckpt_best_path = os.path.join(cfg.logger.log_dir, "model_best.pt")
best_eval_reward = -float("inf")

torch.set_default_device(cfg.device)
print("[INFO] GPU in use:", cfg.device)
torch.set_float32_matmul_precision('high')
set_all_seeds(cfg.seed)

agent = PQN(agent_config)

env: QAEnv = instantiate(env_config.env)
env_test: QAEnv = instantiate(env_config.test_env)

parallel_env = ParallelTextEnv(
    [env] + [env.copy() for _ in range(cfg.envs_parallel - 1)], 
    state_tokenizer=agent.state_tokenizer,
    action_tokenizer=agent.action_tokenizer)

total_steps = cfg.steps_count * cfg.accumulate_grads
eval_interval = cfg.eval_interval * cfg.accumulate_grads
#assuming we don't need to scale cfg.learning_start with grad_accumulation

states_list, _ = parallel_env.reset()
step = 0
train_rewards = []
progress_bar = tqdm(range(total_steps), desc="Train", ncols=90)

#agent.save(os.path.join(cfg.logger.log_dir, "model_pretrained.pt"))
#print("Checkpoint before finetuning has been saved.")

for it in progress_bar:

    agent.train()
    states_list, rewards, train_batch = parallel_env.rollout(cfg.batch_size, states_list, agent, random=(step < 2 * cfg.learning_start))
    step += np.prod(train_batch.reward.shape)
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
        r_eval = []
        for j in range(cfg.eval_episodes):
            r_eval.append(evaluate(env_test, agent))
            print(f"\r Eval progress: {len(r_eval)}/{cfg.eval_episodes}", end="")

        mean_train_reward = np.mean(train_rewards)
        mean_eval_reward = np.mean(r_eval)
        train_rewards = []

        writer.add_scalar("train r_sum", mean_train_reward, step)
        writer.add_scalar("eval r_sum", mean_eval_reward, step)
        writer.add_scalar("qf_loss", qf_loss, step)

        progress_bar.set_postfix({
            't_r': mean_train_reward,
            "e_r": mean_eval_reward,
            'l': qf_loss,
            'st': step,
        })

        agent.save(ckpt_last_path)
        if mean_eval_reward > best_eval_reward:
            best_eval_reward = mean_eval_reward
            agent.save(ckpt_best_path)
            print(f"\nNew best model has been saved. step={step}, train_reward={mean_train_reward:.3f}, eval_reward={mean_eval_reward:.3f}")
