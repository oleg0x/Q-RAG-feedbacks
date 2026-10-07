import numpy as np
from torch import nn, Tensor
import torch
from envs.utils import TextMemory, TextMemoryItem
import copy


def logsumexp(inputs: Tensor, attention_mask: Tensor, dim=1, keepdim=False):
    """Numerically stable logsumexp over the available actions.

    The shift is the maximum of the **already masked** logits, not of all of
    them. Shifting by the global maximum breaks at alpha≈0.005: a gap above
    ~0.1 zeroes the exponent, and if the global maximum is masked, V collapses
    to `max − 0.1`, i.e. the Q-value of an **unavailable** action. This is rare
    with ten candidates and two steps but constant with per-title masking over
    21M rows.

    Masked terms are set to `-inf` before the exponent rather than multiplied
    by the mask afterwards: `exp` of a large logit is `inf`, `inf * 0` is
    `nan`, and a single masked outlier would poison the whole batch.
    """
    mask = attention_mask.to(torch.bool)
    masked_inputs = torch.where(mask, inputs, torch.full_like(inputs, float("-inf")))

    s, _ = torch.max(masked_inputs, dim=dim, keepdim=True)
    # A row with no available action would give -inf - (-inf) = nan.
    s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))

    exp_x = torch.exp(masked_inputs - s)

    outputs = s + torch.log(exp_x.sum(dim=dim, keepdim=True).clamp(min=1e-10))

    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def masked_top_k_mask(logits: Tensor, available_mask: Tensor, top_k_actions: int) -> Tensor:
    """Top-K mask computed **after** the availability mask.

    The order matters: `topk` over raw logits would give pool slots to
    unavailable actions, and the policy would see a top-K from which some
    rows are dropped later.
    """
    available = available_mask.to(torch.bool)
    masked = torch.where(
        available, logits, torch.full_like(logits, float("-inf"))
    )
    top_k = min(masked.size(-1), top_k_actions)
    top_ids = torch.topk(masked, top_k, dim=-1).indices
    top_mask = torch.zeros_like(available).scatter_(-1, top_ids, True)
    # `topk` returns masked rows if fewer than K actions are available.
    return top_mask & available


def masked_soft_value(
    logits_1: Tensor,
    logits_2: Tensor,
    available_mask: Tensor,
    alpha,
    top_k_actions: int,
) -> tuple[Tensor, Tensor]:
    """`V(s) = α·logsumexp(Q/α)` over the top-K available actions, both heads.

    Factored out of `TextVNet` because in line A the logits are already
    computed by the search and the state need not be re-encoded for `V`; the
    masking must be identical on both paths.
    """
    top_mask_1 = masked_top_k_mask(logits_1, available_mask, top_k_actions)
    top_mask_2 = masked_top_k_mask(logits_2, available_mask, top_k_actions)

    v1 = alpha * logsumexp(logits_1 / alpha, attention_mask=top_mask_1, dim=-1)
    v2 = alpha * logsumexp(logits_2 / alpha, attention_mask=top_mask_2, dim=-1)
    return v1, v2


def calibrated_soft_value(
    logits_1: Tensor,
    logits_2: Tensor,
    available_mask: Tensor,
    alpha,
    top_k_actions: int,
    scale: Tensor,
    bias: Tensor,
) -> Tensor:
    """V(s) over the calibrated heads: the mean of the two α·logsumexp.

    The heads are fitted to the target as `w·(2·q_i) + b`, so V must use the
    same quantities; otherwise the target `r + γ·V` would live on a different
    scale and the calibration would fix one side of the Bellman equation while
    breaking the other. Mean rather than sum: each calibrated head already
    estimates the full value, and the sum would double it.
    """
    h1 = scale * (2 * logits_1) + bias
    h2 = scale * (2 * logits_2) + bias
    v1, v2 = masked_soft_value(h1, h2, available_mask, alpha, top_k_actions)
    return 0.5 * (v1 + v2)


def normalized_boltzmann_probs(logits: Tensor, alpha, eps: float = 1e-6) -> Tensor:
    """Boltzmann over the pool with logits normalized by their own spread.

    With `normalize=false` the logit scale is proportional to ‖s‖, which can
    shrink several-fold during training (20.68 → 3.21 in one run). With a
    fixed α, exploration is then suppressed early and grows mid-run, an
    inverted schedule. Dividing by the spread of the current pool makes α
    dimensionless and the probabilities invariant to any positive affine
    transform of the logits.
    """
    centered = logits - logits.max(dim=-1, keepdim=True).values
    spread = logits.std(dim=-1, keepdim=True)
    scale = (spread * alpha).clamp(min=eps)
    return (centered / scale).softmax(-1)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    target.load_state_dict({
            k: v.clone() for k, v in source.state_dict().items()
    })
    # for target_param, param in zip(target.parameters(), source.parameters()):
    #     target_param.data.copy_(param.data)


class TextQNet(nn.Module):

    def __init__(self, state_embed, action_embed, q_head: dict | None = None) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed
        # Calibration head: Q = w·(2·q_head_i) + b. Without it, line A starts
        # incompatible with the reward: logits s·M ≈ ±20 against targets in
        # [0,1], and the cheapest way for MSE is to shrink and flip the state
        # vector globally (overfitting 100 examples drove the cosine to GTE to
        # −0.25 and own-pool recall to zero). A scalar w initialized at 1/‖s₀‖
        # takes the scale pressure off the tower weights without changing the
        # search ranking (monotone for w > 0). The parameters are created only
        # on request: line B checkpoints load strictly and must not see new keys.
        if q_head is not None:
            self.q_scale = nn.Parameter(
                torch.tensor(float(q_head["scale_init"]))
            )
            self.q_bias = nn.Parameter(torch.tensor(0.0))

    @property
    def calibrated(self) -> bool:
        return hasattr(self, "q_scale")

    def head_values(self, logits_1: Tensor, logits_2: Tensor) -> tuple[Tensor, Tensor]:
        """Head Q-values on the reward scale: what is compared with the target."""
        if self.calibrated:
            return (
                self.q_scale * (2 * logits_1) + self.q_bias,
                self.q_scale * (2 * logits_2) + self.q_bias,
            )
        return 2 * logits_1, 2 * logits_2

    def forward(self, s: TextMemory, a):
        """Q(s, a) with two heads; `a` is either the action text or its vector.

        Line A passes a ready vector: the action was found by searching matrix
        `M`, its row is known, and re-encoding the text would cost an extra BERT
        pass per transition. The positional rotation is disabled on purpose:
        `Q(s,a)` would use a vector rotated by RoPE according to the chunk's
        position in memory, while the search uses raw rows of `M`; at positions
        above zero these differ, breaking the search/evaluation identity that
        line A relies on.
        """
        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        if isinstance(a, Tensor):
            a_embed = a.to(dtype=s_embed.dtype)
        else:
            # self.action_embed.eval()
            a_embed = self.action_embed(input_ids=a.input_ids, attention_mask=a.attention_mask, positions=a.position)["rope"]
            # a_embed = self.action_embed.update_pos(a_embed, positions=a.position)

        D = s_embed.shape[-1] // 2
        logits_1 = (s_embed[..., :D] * a_embed[..., :D]).sum(-1)
        logits_2 = (s_embed[..., D:] * a_embed[..., D:]).sum(-1)

        # logits_1 = logits_1 * self.weight + self.bias
        # logits_2 = logits_2 * self.weight + self.bias

        return logits_1, logits_2

        # return (s_embed * a_embed).sum(-1)


class TextQNetTarget(TextQNet):
    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01):
        soft_update(self, q_net, decay)


class ActionEmbedTarget(nn.Module):
    action_embed: nn.Module

    def __init__(self, action_embed: nn.Module, q_net: TextQNet) -> None:
        super().__init__()
        self.action_embed = action_embed
        self.action_embed.load_state_dict({
            k: v.clone() for k, v in q_net.action_embed.state_dict().items()
        })

    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01):
        soft_update(self.action_embed, q_net.action_embed, decay)

    @torch.no_grad()
    def forward(self, *args, **kw):
        return self.action_embed.forward(*args, **kw)
    
    @torch.no_grad()
    def update_pos(self, *args, **kw):
        return self.action_embed.update_pos(*args, **kw)



class TextQNetPolicy(nn.Module):
    state_embed: nn.Module

    def __init__(self, state_embed: nn.Module, q_net: TextQNet = None, top_k_actions=5) -> None:
        super().__init__()
        self.state_embed = state_embed
        # q_net is optional so an online policy can share the critic's state
        # embedder. Other agents can still pass a distinct embedder and use the
        # original snapshot/update behaviour.
        if q_net is not None and self.state_embed is not q_net.state_embed:
            self.state_embed.load_state_dict({
                k: v.clone() for k, v in q_net.state_embed.state_dict().items()
            })
        self.top_k_actions = top_k_actions

    @torch.no_grad()
    def update(self, q_net: TextQNet):
        if self.state_embed is not q_net.state_embed:
            hard_update(self.state_embed, q_net.state_embed)

    @torch.no_grad()
    def forward(self, s: TextMemory, a_embeds: Tensor, alpha: float, return_arg_max=False):
        
        # a_embeds = a_embeds.unsqueeze(1)
        # print("a_embeds", a_embeds.shape)

        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        s_embed = s_embed.unsqueeze(1)
        
        logits = (s_embed * a_embeds).sum(-1).squeeze(-1) 
        # print("logits", logits.shape)
        logits[s.available_mask == False] = logits.min() - 1

        #print('\033[96m'+f'logits: {logits.shape}  topk: {self.top_k_actions}'+"\033[0m")
        top_k_actions = min(logits.size(1), self.top_k_actions)
        top_ids = torch.topk(logits, top_k_actions, dim=1).indices
        top_mask = torch.zeros_like(logits > 0).scatter_(1, top_ids, True)
        # print("top_mask", top_mask.shape)

        if return_arg_max:
            return torch.argmax(logits, -1), logits

        probs = ((logits - logits.max(-1, keepdim=True).values) / alpha).softmax(-1)
        probs[(s.available_mask & top_mask) == False] = 0
        # print(f'probs.sum(): {probs.sum(-1).item()}')
        # print(f'availables: {s.available_mask[0].tolist()}')
        # print(f'top_mask: {top_mask[0].tolist()}')
        # print(f'top_mask & avail: {(top_mask & s.available_mask)[0].tolist()}')
        probs = probs / probs.sum(-1, keepdim=True)
        dist = torch.distributions.Categorical(probs = probs)
        action = dist.sample()

        # print("action", action.shape)

        return action, logits
    

class TextRandomPolicy(nn.Module):


    @torch.no_grad()
    def forward(self, s: TextMemory):

        mask = s.available_mask
        
        probs = (torch.ones(mask.shape[0], mask.shape[1], device=mask.device)).softmax(-1)
        probs[mask == False] = 0
        dist = torch.distributions.Categorical(probs = probs)
        action = dist.sample()

        return action


class SearchBoltzmannPolicy(nn.Module):
    """Line A exploration: Boltzmann over the own pool plus ε-injection from GTE.

    The pool logits are the search scores themselves (`Q(s,a) = s · M[idx]`),
    so the policy needs no extra pass: it ranks by the sum of both heads that
    the search has already computed.

    The model only sees its own top-K and without injection never leaves its
    own ranking. With probability ε the action is taken from the GTE top-K of
    the **current state**: the same search with the same masks, but the query
    is encoded by the frozen action tower. Its tendency on refresh to drift
    towards an already found entity is accepted, since this is an admixture,
    not the main pool.

    Line A has no random phase: uniform choice over 21M rows is noise, not
    exploration, so the old `random=(step < 2 * learning_start)` knob is not
    used here.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        temperature: float = 1.0,
        injection_sampling: str = "boltzmann",
    ) -> None:
        super().__init__()
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1]: {epsilon}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive: {temperature}")
        if injection_sampling not in ("boltzmann", "uniform"):
            raise ValueError(f"Unknown injection sampling: {injection_sampling}")
        self.epsilon = epsilon
        # A separate temperature, not the critic's α: α enters
        # `V = α·logsumexp(Q/α)` and is in logit units, whereas here the logits
        # are already divided by their spread and the number is dimensionless.
        # A shared knob would make TD-target softness silently change how
        # aggressive exploration is.
        self.temperature = temperature
        self.injection_sampling = injection_sampling

    @torch.no_grad()
    def forward(
        self,
        pool_scores: Tensor,
        gte_scores: Tensor | None = None,
        temperature: float | None = None,
        evaluate: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Position of the chosen action in its pool, and the injection flag.

        Returns `(positions, injected)`: where `injected` is true, the position
        indexes `gte_scores`, otherwise `pool_scores`.
        """
        batch = pool_scores.shape[0]
        device = pool_scores.device
        if evaluate:
            return (
                pool_scores.argmax(dim=-1),
                torch.zeros(batch, dtype=torch.bool, device=device),
            )

        temperature = self.temperature if temperature is None else temperature
        probs = normalized_boltzmann_probs(pool_scores, temperature)
        positions = torch.multinomial(probs, 1, generator=generator).squeeze(-1)

        if gte_scores is None or self.epsilon <= 0.0:
            return positions, torch.zeros(batch, dtype=torch.bool, device=device)

        injected = (
            torch.rand(batch, device=device, generator=generator) < self.epsilon
        )
        if self.injection_sampling == "uniform":
            gte_probs = torch.full_like(gte_scores, 1.0 / gte_scores.shape[-1])
        else:
            gte_probs = normalized_boltzmann_probs(gte_scores, temperature)
        gte_positions = torch.multinomial(gte_probs, 1, generator=generator).squeeze(-1)
        return torch.where(injected, gte_positions, positions), injected


class TextVNet(nn.Module):

    state_embed: nn.Module

    def __init__(self, state_embed: nn.Module, q_net: TextQNet, top_k_actions=5) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.state_embed.load_state_dict({
            k: v.clone() for k, v in q_net.state_embed.state_dict().items()
        })
        self.top_k_actions = top_k_actions


    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01):
        soft_update(self.state_embed, q_net.state_embed, decay)

    @torch.no_grad()
    def forward(self, s: TextMemory, a_embeds_target: Tensor, alpha: float):
        # assert alpha > 1e-8

        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        s_embed = s_embed.unsqueeze(1)
        a_embeds: Tensor = a_embeds_target

        # logits = (s_embed * a_embeds).sum(-1)
        D = s_embed.shape[-1] // 2
        logits_1 = (s_embed[:, :, :D] * a_embeds[:, :, :D]).sum(-1)
        logits_2 = (s_embed[:, :, D:] * a_embeds[:, :, D:]).sum(-1)

        return masked_soft_value(
            logits_1,
            logits_2,
            s.available_mask,
            alpha,
            self.top_k_actions,
        )


class TextMaxQNet(nn.Module):

    state_embed: nn.Module

    def __init__(self, state_embed: nn.Module, q_net: TextQNet) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.state_embed.load_state_dict({
            k: v.clone() for k, v in q_net.state_embed.state_dict().items()
        })

        self.weight = nn.Parameter(torch.ones(1)).cuda()
        self.bias = nn.Parameter(torch.zeros(1)).cuda()


    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01):
        soft_update(self.state_embed, q_net.state_embed, decay)
        self.weight.data = self.weight.data * (1 - decay) + q_net.weight.data * decay
        self.bias.data = self.bias.data * (1 - decay) + q_net.bias.data * decay

    @torch.no_grad()
    def forward(self, s: TextMemory):

        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        s_embed = s_embed[:, None, :]
        a_embeds: Tensor = s.embeds
        
        D = s_embed.shape[-1] // 2
        logits_1 = (s_embed[:, :, :D] * a_embeds[:, :, :D]).sum(-1) 
        logits_2 = (s_embed[:, :, D:] * a_embeds[:, :, D:]).sum(-1) 

        logits_1[s.available_mask == False] = torch.min(logits_1)
        logits_2[s.available_mask == False] = torch.min(logits_2)

        return (logits_1.max(-1).values, 
                logits_2.max(-1).values)
