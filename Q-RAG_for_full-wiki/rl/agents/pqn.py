from functools import partial
from contextlib import contextmanager
import os
from typing import Tuple
import torch
from torch import nn, Tensor
from torch import optim
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from collections import namedtuple
from rl.bert_predictor import EmbedderWithAbsoluteEncoding
from envs.utils import custom_pad_sequence, stack_memory, stack_text_list
from ..q_module import TextQNet, TextQNetPolicy, TextRandomPolicy, ActionEmbedTarget, TextMaxQNet, TextVNet
from envs.utils import TextMemory, TextMemoryItem
import copy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


@partial(torch.compile)
def policy_apply(policy, v_net, state, a_embeds,  a_embeds_target, alpha, return_argmax: bool):
    action, q_values = policy(state, a_embeds, alpha, return_argmax)
    v1, v2 = v_net(state, a_embeds_target, alpha=alpha)
    q_values_target = v1 + v2
    return action, q_values, q_values_target


@torch.no_grad()
def compute_returns(rewards, values_next, not_done, gamma=0.99, lambda_coef=.0):
    """
        Compute finite-horizon TD(lambda) / lambda-return targets for a batch of rollouts.

        Args:
            rewards: Tensor of shape [num_envs, num_steps].
                Immediate rewards r_t for each environment and rollout step.

            values_next: Tensor of shape [num_envs, num_steps].
                Bootstrap value estimates for the next states, V(s_{t+1}), for each
                transition. For Q-learning/PQN this is typically max_a Q_target(s_{t+1}, a),
                or the corresponding Double-Q target estimate.

            not_done: Tensor of shape [num_envs, num_steps].
                Binary mask for episode continuation after each transition.
                Should be 1 when the transition can bootstrap from the next state and
                0 when the transition ends the episode and no future value should be used.

            gamma: float.
                Discount factor.

            lambda_coef: float.
                TD(lambda) coefficient. lambda_coef=0 gives one-step TD targets;
                lambda_coef=1 gives Monte Carlo-style returns over the collected rollout,
                with bootstrap from values_next at the final rollout step.

        Returns:
            Tensor of shape [num_envs, num_steps].
            Lambda-return targets G_t^lambda computed backwards according to:

                G^{λ}_t =r_t + not_done * γ ( (1. - λ) * V(s_{t+1}) + λ * G^λ_{t+1})

            For the final collected transition, the recursion is truncated as:

                G^{λ}_t = r_t + not_done * γ * V(s_{t+1})
    """

    num_envs, num_steps = rewards.shape
    # variable for λ-returns G^{λ}_t:
    returns = rewards.new_empty(num_envs, num_steps)

    for i in reversed(range(num_steps)):
        if i == num_steps - 1:
            # at final rollout step we have only V(s_{t+1}) estimates and no future rewards
            # therefore: G^{λ}_t = r_t + not_done * γ * V(s_{t+1})
            value_target = (gamma * values_next[:, i])
        else:
            # G^{λ}_t =r_t + not_done * γ ( (1. - λ) * V(s_{t+1}) + λ * G^λ_{t+1})
            value_target = gamma * ((1. - lambda_coef) * values_next[:, i] + lambda_coef * returns[:, i + 1])
        returns[:, i] = rewards[:, i] + not_done[:, i] * value_target

    return returns

class AlphaSchedule:
    """Расписание температуры Boltzmann, не привязанное к learning rate.

    Раньше α пересчитывалась как `alpha_start * lr(t) / lr(0)`. При warmup в
    1000 шагов это означало почти нулевую температуру в начале обучения и
    рост к середине рана — расписание exploration было перевёрнуто. Теперь
    это явная ручка конфига: `constant` или линейный спад от `start` к
    `final` за `total` оптимизационных шагов после `warmup`.
    """

    KINDS = ("constant", "linear")

    def __init__(self, start: float, kind: str = "constant", final=None, warmup: int = 0, total=None):
        if kind not in self.KINDS:
            raise ValueError(f"Unknown alpha schedule: {kind}. Use one of {self.KINDS}")
        if start <= 0:
            raise ValueError(f"alpha must be positive: {start}")
        self.kind = kind
        self.start = float(start)
        self.final = float(start if final is None else final)
        self.warmup = int(warmup)
        self.total = None if total is None else int(total)
        if self.kind == "linear":
            if self.total is None:
                raise ValueError("Linear alpha schedule needs pqn.hyperparams.alpha_schedule.total")
            if self.final <= 0:
                raise ValueError(f"Final alpha must be positive: {self.final}")
            if self.total <= self.warmup:
                raise ValueError(
                    f"Alpha schedule total={self.total} must exceed warmup={self.warmup}"
                )

    @classmethod
    def from_config(cls, config: DictConfig, start: float) -> "AlphaSchedule":
        settings = OmegaConf.select(config, "pqn.hyperparams.alpha_schedule", default=None)
        if settings is None:
            return cls(start=start, kind="constant")
        settings = OmegaConf.to_container(settings, resolve=True)
        return cls(
            start=float(settings.get("start", start)),
            kind=str(settings.get("kind", "constant")),
            final=settings.get("final"),
            warmup=int(settings.get("warmup", 0)),
            total=settings.get("total"),
        )

    def value(self, optim_step: int) -> float:
        if self.kind == "constant" or optim_step <= self.warmup:
            return self.start
        if optim_step >= self.total:
            return self.final
        fraction = (optim_step - self.warmup) / (self.total - self.warmup)
        return self.start + fraction * (self.final - self.start)


class PQN(object):

    def __init__(self, config: DictConfig):

        self.config = copy.deepcopy(config)
        self.gamma = config.pqn.hyperparams.gamma
        self.alpha = config.pqn.hyperparams.alpha
        self.alpha_start = self.alpha
        self.alpha_schedule = AlphaSchedule.from_config(config, self.alpha_start)
        self.Lambda = config.pqn.hyperparams.Lambda
        self.tau = config.pqn.hyperparams.tau
        self.start_lr = config.pqn.optimizer.lr
        # Пул политики: сколько действий она вообще рассматривает. Раньше
        # число было зашито пятёркой, и на пуле в сто кандидатов политика
        # видела 5% множества действий.
        self.top_k_actions = int(
            OmegaConf.select(config, "pqn.hyperparams.top_k_actions", default=5)
        )
        print("tau", self.tau)
        # ===new===
        self.max_grad_norm = config.pqn.hyperparams.max_grad_norm
        self.accumulate_grads = config.pqn.hyperparams.accumulate_grads
        if self.accumulate_grads < 1:
            raise ValueError("cfg.accumulate_gradients must be a positive integer")
        self._update_step = 0  # number of updates from the start of the training
        self._optim_step = 0  # number of optimizer steps; drives the alpha schedule
        # self.action_embed_length = config.pqn.hyperparams.action_embed_length
        self.max_action_length_in_memory = config.pqn.hyperparams.max_action_length_in_memory
        self.train_state_embed = OmegaConf.select(
            config, "pqn.hyperparams.train_state_embed", default=True
        )
        self.train_action_embed = OmegaConf.select(
            config, "pqn.hyperparams.train_action_embed", default=True
        )

        state_embed: nn.Module = instantiate(config.pqn.state_embed)
        action_embed: nn.Module = instantiate(config.pqn.action_embed)
        state_embed_target: nn.Module = instantiate(config.pqn.state_embed_target)
        action_embed_target: nn.Module = instantiate(config.pqn.action_embed_target)
        self._set_module_trainable(state_embed, self.train_state_embed)
        self._set_module_trainable(action_embed, self.train_action_embed)
        
        q_head = OmegaConf.select(config, "pqn.hyperparams.q_head", default=None)
        if q_head is not None:
            q_head = OmegaConf.to_container(q_head, resolve=True)
        self.critic = TextQNet(state_embed, action_embed, q_head=q_head).to(torch.get_default_device())
        head_params = (
            [self.critic.q_scale, self.critic.q_bias] if self.critic.calibrated else []
        )
        head_ids = {id(p) for p in head_params}
        tower_params = [
            p
            for p in self.critic.parameters()
            if p.requires_grad and id(p) not in head_ids
        ]
        self.critic_trainable_params = tower_params + head_params
        if not tower_params:
            raise ValueError(
                "At least one PQN embedder must be trainable. "
                "Set pqn.hyperparams.train_state_embed or "
                "pqn.hyperparams.train_action_embed to true."
            )
        param_groups = [{"params": tower_params}]
        if head_params:
            # Скаляры калибровки в своей группе с быстрым lr: рассогласование
            # масштаба Q с наградой должно чиниться головой за единицы шагов,
            # иначе давление масштаба снова пойдёт в веса башни.
            param_groups.append({"params": head_params, "lr": float(q_head["lr"])})
        # _partial_: hydra не должен видеть группы параметров — переданные в
        # instantiate kwargs он оборачивает в DictConfig, и AdamW падает на
        # «optimizer can only optimize Tensors».
        self.critic_optim = instantiate(config.pqn.optimizer, _partial_=True)(param_groups)
        self.scheduler = instantiate(config.pqn.scheduler, optimizer=self.critic_optim)
       
        # The policy represents the online critic, so it uses the very same state
        # embedder instead of keeping an eagerly synchronized duplicate of it.
        self.policy = TextQNetPolicy(state_embed, top_k_actions=self.top_k_actions)
        self.random_policy = TextRandomPolicy().to(torch.get_default_device())

        self.v_net_target = TextVNet(
            state_embed_target, self.critic, top_k_actions=self.top_k_actions
        ).to(torch.get_default_device())
        if self.critic.calibrated:
            # EMA-копии калибровки для таргета: V(s') обязан считаться тем же
            # масштабом, которым жил target-критик, а не сегодняшним онлайн-w.
            self.q_scale_target = self.critic.q_scale.detach().clone()
            self.q_bias_target = self.critic.q_bias.detach().clone()
        self.action_embed_target = ActionEmbedTarget(action_embed_target, self.critic).to(torch.get_default_device())
        self.train()

        self.state_tokenizer = state_embed.tokenizer
        self.action_tokenizer = action_embed.tokenizer

        self.train_step = self.make_train_step()


    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(trainable)
        if not trainable:
            module.eval()


    def _apply_fixed_eval_modes(self) -> None:
        """Enforce eval mode for modules that must never enter train mode."""
        self.v_net_target.eval()
        self.action_embed_target.eval()

        if not self.train_state_embed:
            self.policy.eval()
            self.critic.state_embed.eval()
        if not self.train_action_embed:
            self.critic.action_embed.eval()


    def _set_online_models_mode(self, training: bool) -> None:
        """Set trainable online modules to the requested mode.

        Frozen online embedders and all target embedders remain in eval mode.
        """
        state_training = training and self.train_state_embed
        action_training = training and self.train_action_embed

        # TextQNet and TextQNetPolicy do not have mode-dependent layers of their
        # own, but keeping their flags consistent makes module state unambiguous.
        self.policy.train(state_training)
        self.critic.train(state_training or action_training)
        self.critic.state_embed.train(state_training)
        self.critic.action_embed.train(action_training)
        self._apply_fixed_eval_modes()


    @contextmanager
    def online_models_mode(self, training: bool):
        """Temporarily set online models to train/eval and restore their modes."""
        modules = list(dict.fromkeys([
            *self.critic.modules(),
            *self.policy.modules(),
        ]))
        previous_modes = [(module, module.training) for module in modules]

        try:
            self._set_online_models_mode(training)
            yield
        finally:
            # Assigning the flag directly restores the exact per-module state;
            # recursive train() calls could overwrite modes of frozen children.
            for module, previous_mode in previous_modes:
                module.training = previous_mode
            self._apply_fixed_eval_modes()


    def make_train_step(self):

        # @partial(torch.compile)
        def train_step(
                    critic,
                    state_batch: TextMemory,
                    action_batch: TextMemoryItem,
                    reward_batch: Tensor,
                    valid_batch: Tensor = None
        ):

            reward_batch = reward_batch.squeeze()

            qf_1, qf_2 = critic(state_batch, action_batch)
            qf_1, qf_2 = qf_1.squeeze(), qf_2.squeeze()
            # С таргетом сравниваются калиброванные головы: без w,b логиты
            # s·M ≈ ±20 при таргетах [0,1], и MSE убивает геометрию башни.
            h_1, h_2 = critic.head_values(qf_1, qf_2)
            if valid_batch is None:
                qf_loss = 0.5 * F.mse_loss(h_1, reward_batch) + 0.5 * F.mse_loss(h_2, reward_batch)
            else:
                # Переход без награды (сервер не ответил) выпадает из лосса
                # целиком, а не приходит нулём: ноль означал бы «контекст
                # бесполезен», и критик учился бы на выдуманном сигнале.
                weight = valid_batch.reshape(-1).to(dtype=qf_1.dtype)
                divisor = weight.sum().clamp(min=1.0)
                qf_loss = (
                    0.5 * (weight * (h_1 - reward_batch) ** 2).sum() / divisor
                    + 0.5 * (weight * (h_2 - reward_batch) ** 2).sum() / divisor
                )

            (qf_loss / self.accumulate_grads).backward()

            return qf_loss

        return train_step


    @torch.no_grad()
    def select_action_batch(self, state: TextMemory, a_embeds: Tensor,  a_embeds_target: Tensor, evaluate=False, random=False):
        
        torch_state = TextMemory(
            item_ids=None,
            available_ids=None,
            available_mask=state.available_mask,
            text=None,
            input_ids=state.input_ids,
            attention_mask=state.attention_mask,
        )
        action, q_values, q_values_target = policy_apply(self.policy, self.v_net_target, torch_state, a_embeds,  a_embeds_target, torch.tensor(self.alpha), evaluate)

        if random:
            action = self.random_policy.forward(state)
            
        return action.squeeze(), q_values.squeeze(), q_values_target.squeeze()


    @torch.no_grad()
    def select_action(self, state: TextMemory, a_embeds: Tensor, a_embeds_target: Tensor, evaluate=False, random=False):

        state = stack_memory([state], self.critic.action_embed.tokenizer, max_length=self.max_action_length_in_memory)
        a_embeds = custom_pad_sequence([a_embeds], padding_value=0.0, batch_first=True, pad_to_power_2=False)
        a_embeds_target = custom_pad_sequence([a_embeds_target], padding_value=0.0, batch_first=True, pad_to_power_2=False)
                
        action, q_values, q_values_target =  self.select_action_batch(state, a_embeds, a_embeds_target, evaluate, random)

        return action.item(), q_values, q_values_target
        
    
    # @torch.no_grad()
    # def _get_target(self, lambda_returns, next_q, q_values, rewards, dones_mask):
    #     target_bootstrap = (
    #         rewards + self.gamma * dones_mask * next_q
    #     )
    #     delta = lambda_returns - next_q
    #     lambda_returns = (
    #         target_bootstrap + self.gamma * self.Lambda * delta
    #     )
    #     lambda_returns = dones_mask * lambda_returns + (1.0 - dones_mask) * rewards
    #     next_q = q_values
    #
    #     return lambda_returns, next_q


    # def update_old(self,
    #             state_batch: TextMemory,
    #             action_batch: TextMemoryItem,
    #             next_state_batch: TextMemory,
    #             q_values_batch: Tensor,
    #             reward_batch: Tensor,
    #             mask_batch: Tensor):
    #
    #
    #     last_q = mask_batch[:, -2] * q_values_batch[:, -1]
    #     lambda_returns = reward_batch[:, -2] + self.gamma * last_q
    #
    #     targets = [lambda_returns]
    #
    #     for t in range(q_values_batch.shape[1] - 3, -1, -1):
    #         lambda_returns, last_q = self._get_target(lambda_returns, last_q, q_values_batch[:, t], reward_batch[:, t], mask_batch[:, t])
    #         targets.append(lambda_returns)
    #
    #     targets.reverse()
    #     targets = torch.stack(targets, dim=1)
    #     assert targets.shape[0] == q_values_batch.shape[0]
    #     assert targets.shape[1] == q_values_batch.shape[1] - 1
    #     targets = targets.reshape(-1)
    #
    #     state_batch = TextMemory(
    #             item_ids=None,
    #             available_ids=None,
    #             available_mask=state_batch.available_mask,
    #             text=None,
    #             input_ids=state_batch.input_ids,
    #             attention_mask=state_batch.attention_mask
    #         )
    #
    #     action_batch = TextMemoryItem(
    #         index=None,
    #         position=torch.tensor(action_batch.position, device=action_batch.input_ids.device, dtype=torch.float32),
    #         input_ids=action_batch.input_ids,
    #         attention_mask=action_batch.attention_mask,
    #         text=None
    #     )
    #
    #     qf_loss = self.train_step(self.critic, state_batch, action_batch, targets) #computes backward inside
    #
    #     self._update_step += 1
    #     if self._update_step % self.accumulate_grads == 0:
    #         torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
    #         self.critic_optim.step()
    #         self.scheduler.step()
    #         self.critic_optim.zero_grad()
    #
    #         self.alpha = self.alpha_start * float(self.scheduler.get_lr()[0]) / self.start_lr
    #         self.v_net_target.update(self.critic, self.tau)
    #         self.action_embed_target.update(self.critic, self.tau)
    #         self.policy.update(self.critic)
    #
    #     return qf_loss.item()


    def update(
            self,
            state_batch: TextMemory,
            action_batch: TextMemoryItem,
            next_state_batch: TextMemory,
            q_values_batch: Tensor,
            reward_batch: Tensor,
            mask_batch: Tensor,
            valid_batch: Tensor = None):

        # Rollout collection may temporarily use eval mode. Gradient updates
        # must always run with trainable online modules in training mode.
        self.train()

        state_values = q_values_batch
        rewards = reward_batch
        not_done = mask_batch.to(dtype=rewards.dtype)

        if rewards.ndim != 2:
            raise ValueError(f"reward_batch must be 2-D, got shape {tuple(rewards.shape)}")

        num_envs, num_steps = rewards.shape

        if not_done.shape != rewards.shape:
            raise ValueError(
                f"mask_batch shape {tuple(not_done.shape)} must match "
                f"reward_batch shape {tuple(rewards.shape)}"
            )

        if state_values.shape != (num_envs, num_steps + 1):
            raise ValueError(
                f"q_values_batch must have shape [num_envs, num_steps + 1]. "
                f"Got {tuple(state_values.shape)}, expected {(num_envs, num_steps + 1)}"
            )

        returns = compute_returns(rewards, state_values[:,1:], not_done, self.gamma, self.Lambda)
        flat_targets = returns.reshape(-1)

        # next_state_batch is not needed here: rollout has already stored
        # the target-network state values in q_values_batch.
        critic_states = TextMemory(
            item_ids=None,
            available_ids=None,
            available_mask=state_batch.available_mask,
            text=None,
            input_ids=state_batch.input_ids,
            attention_mask=state_batch.attention_mask,
        )

        if isinstance(action_batch, Tensor):
            # Линия A: действие — готовая строка `M`, перекодировать нечего.
            critic_actions = action_batch
        else:
            critic_actions = TextMemoryItem(
                index=None,
                position=torch.as_tensor(
                    action_batch.position,
                    device=action_batch.input_ids.device,
                    dtype=torch.float32,
                ),
                input_ids=action_batch.input_ids,
                attention_mask=action_batch.attention_mask,
                text=None,
            )

        flat_valid = None
        if valid_batch is not None:
            if valid_batch.shape != rewards.shape:
                raise ValueError(
                    f"valid_batch shape {tuple(valid_batch.shape)} must match "
                    f"reward_batch shape {tuple(rewards.shape)}"
                )
            flat_valid = valid_batch.reshape(-1)

        qf_loss = self.train_step(
            self.critic, critic_states, critic_actions, flat_targets, flat_valid
        )

        self._update_step += 1
        if self._update_step % self.accumulate_grads == 0:
            torch.nn.utils.clip_grad_norm_(self.critic_trainable_params, self.max_grad_norm)
            self.critic_optim.step()
            self.scheduler.step()
            self.critic_optim.zero_grad()

            self._optim_step += 1
            self.alpha = self.alpha_schedule.value(self._optim_step)
            if self.train_state_embed:
                self.v_net_target.update(self.critic, self.tau)
                if self.critic.calibrated:
                    self.q_scale_target.mul_(1 - self.tau).add_(
                        self.tau * self.critic.q_scale.detach()
                    )
                    self.q_bias_target.mul_(1 - self.tau).add_(
                        self.tau * self.critic.q_bias.detach()
                    )
            if self.train_action_embed:
                self.action_embed_target.update(self.critic, self.tau)

        return qf_loss.item()

    def train(self):
        self._set_online_models_mode(training=True)

    def eval(self):
        self._set_online_models_mode(training=False)

    def save(self, checkpoint_path: str, verbose=False) -> None:
        """
        Save state of all networks, optimizer and scheduler into a single file

        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            "critic": self.critic.state_dict(),
            "random_policy": self.random_policy.state_dict(),
            "v_net_target": self.v_net_target.state_dict(),
            "action_embed_target": self.action_embed_target.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "alpha": self.alpha, #changes in training phase
            "optim_step": self._optim_step, #position on the alpha schedule
        }
        if self.critic.calibrated:
            checkpoint["q_scale_target"] = self.q_scale_target
            checkpoint["q_bias_target"] = self.q_bias_target
        torch.save(checkpoint, checkpoint_path)
        if verbose:
            print(f"[INFO] PQN checkpoint saved → {checkpoint_path}")

    def load(self, checkpoint_path: str, strict: bool = True, verbose=False) -> None:
        """
        load network state_dict from checkpoint
        `strict` goes to `load_state_dict`.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=torch.get_default_device(), weights_only=False)

        self.critic.load_state_dict(checkpoint["critic"], strict=strict)
        # policy.state_embed is the same module as critic.state_embed. Older
        # checkpoints may contain a separate policy entry; the critic is now the
        # single source of truth, so no second load is needed.
        # random_policy have no parameters right now, but this may change in the future:
        if "random_policy" in checkpoint:
            self.random_policy.load_state_dict(checkpoint["random_policy"], strict=False)
        self.v_net_target.load_state_dict(checkpoint["v_net_target"], strict=strict)
        self.action_embed_target.load_state_dict(checkpoint["action_embed_target"], strict=strict)

        if self.critic.calibrated and "q_scale_target" in checkpoint:
            self.q_scale_target = checkpoint["q_scale_target"].to(
                self.q_scale_target.device
            )
            self.q_bias_target = checkpoint["q_bias_target"].to(
                self.q_bias_target.device
            )

        # restore scheduler
        if "critic_optim" in checkpoint:
            self.critic_optim.load_state_dict(checkpoint["critic_optim"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        # restore α, which may have changed during training
        self.alpha = checkpoint.get("alpha", self.alpha)
        # Позиция на расписании α: без неё дообучение с чекпоинта начинало бы
        # расписание заново, даже когда α сохранена.
        self._optim_step = int(checkpoint.get("optim_step", self._optim_step))

        print(f"[INFO] PQN checkpoint loaded  ← {checkpoint_path}")


class PQNActor:

    def __init__(self, agent: PQN):
        self.agent = agent
        self.embeds = []
        self.embeds_target = []

    @torch.no_grad()
    def get_embeds(self, all_texts, positions) -> Tuple[Tensor, Tensor]:
        tokenizer = self.agent.action_tokenizer
        embedder = self.agent.critic.action_embed
        embedder_target = self.agent.action_embed_target

        batch = stack_text_list(list(all_texts), tokenizer)
        positions = torch.tensor(positions, device=torch.get_default_device())
        embeds, embeds_target = embedder(**batch, positions=positions), embedder_target(**batch, positions=positions)

        return embeds, embeds_target
    
    @torch.no_grad()
    def update_embeds(self, k, positions):
        positions = torch.tensor(positions, device=torch.get_default_device())
        embedder = self.agent.critic.action_embed
        embedder_target = self.agent.action_embed_target
        self.embeds[k] = embedder.update_pos(self.embeds[k], positions=positions)
        self.embeds_target[k] = embedder_target.update_pos(self.embeds_target[k], positions=positions)
        
    def step(self, s_seq, chunks, positions, is_random):
        for k, ch, pos in zip(range(len(chunks)), chunks, positions):
            if ch is not None:
                self.embeds[k], self.embeds_target[k] = self.get_embeds(ch, pos)
            if pos is not None:
                self.update_embeds(k, pos)

        s_par = stack_memory(s_seq, self.state_tokenizer, max_length=self.agent.max_action_length_in_memory)
        
        a_embeds_pos = [emb["rope"] for emb in self.embeds]
        a_embeds_target_pos = [emb["rope"] for emb in self.embeds_target]
             
        embeds_pt = custom_pad_sequence(a_embeds_pos, padding_value=0.0, batch_first=True, pad_to_power_2=False)
        embeds_target_pt = custom_pad_sequence(a_embeds_target_pos, padding_value=0.0, batch_first=True, pad_to_power_2=False)
        
        action, _, q_values  = self.agent.select_action_batch(s_par, embeds_pt, embeds_target_pt, random=is_random)
        
        return action, q_values
            
        
