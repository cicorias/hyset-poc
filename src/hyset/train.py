"""Training loop (paper Alg. 1, Eq. 8-11).

L(theta) = L_ret + eta * L_self + lambda * sum_m ||M_m||_F^2

- L_ret:  InfoNCE of the annotated set against a K_neg-sized negative pool (Eq. 8).
- L_self: reward-weighted log-likelihood of the model's own argmax set in the pool,
          rho_i from an execution-feedback provider; Ê_i and rho_i are constants (Eq. 9).
- Early stopping on validation Recall@5 (Sec. 3.1, implementation details).
"""

from __future__ import annotations

import copy
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .data import Dataset, Example
from .encoders import Encoder
from .evaluate import evaluate_model
from .inference import InferenceConfig, SetRetriever
from .model import HySet, ModelConfig
from .negatives import NegativeMix, NegativeSampler, ToolSet, canon, pools_to_tensor
from .reward import ExecutionFeedback, RewardProvider

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    z_lr_mult: float = 1.0  # lr multiplier for the tool embeddings Z
    k_neg: int = 64
    mix: NegativeMix = field(default_factory=NegativeMix)
    n_neighbors: int = 10  # neighbourhood size for hard negatives
    eta: float = 0.5  # weight of L_self
    lam: float = 1e-2  # weight of the ||M_m||_F^2 regularizer (also curbs F_set drift)
    patience: int = 3  # early-stopping patience, in epochs
    eval_every: int = 0  # also validate every N steps (0 = once per epoch)
    val_metric: str = "recall@5"
    max_val: int = 1000  # cap on validation queries scored per epoch
    seed: int = 0
    # execution feedback (only used when a reward provider is given)
    n_sub: int = 512  # N_sub: queries that receive feedback at any time
    refresh_steps: int = 500  # T_ref: re-draw the feedback subset every this many steps
    max_reward_calls: int = 20_000  # total rollout / judge budget
    self_train_start_epoch: int = 1  # 0-based; let L_ret warm up first

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        d = dict(d)
        d["mix"] = NegativeMix(**d.get("mix", {}))
        return cls(**d)


@dataclass
class TrainResult:
    model: HySet
    history: list[dict]
    best_epoch: int
    tool_emb: np.ndarray
    best_step: int = 0
    best_val: dict | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _usable(examples: list[Example], max_card: int) -> list[Example]:
    kept = [e for e in examples if len(e.tools) <= max_card]
    if dropped := len(examples) - len(kept):
        log.warning("dropping %d examples with more than M=%d tools", dropped, max_card)
    return kept


def train(
    dataset: Dataset,
    encoder: Encoder,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    infer_cfg: InferenceConfig | None = None,
    reward: RewardProvider | None = None,
    on_epoch: Callable[[HySet, int], dict] | None = None,
) -> TrainResult:
    """Train HySet. ``on_epoch(model, epoch)`` may return extra fields for the history."""
    model_cfg = model_cfg or ModelConfig()
    cfg = train_cfg or TrainConfig()
    infer_cfg = infer_cfg or InferenceConfig()
    _seed_everything(cfg.seed)

    lib = dataset.library
    M = model_cfg.max_card
    train_ex = _usable(dataset.splits["train"], M)
    val_ex = _usable(dataset.splits.get("val", []), M)[: cfg.max_val]

    log.info(
        "encoding %d tools and %d train queries with %s", len(lib), len(train_ex), encoder.spec
    )
    tool_emb = encoder.encode(lib.texts(), "tool")
    r_train = torch.from_numpy(encoder.encode([e.query for e in train_ex], "query"))
    gts: list[ToolSet] = [canon(lib.indices(e.tools)) for e in train_ex]
    if val_ex:
        r_val = torch.from_numpy(encoder.encode([e.query for e in val_ex], "query"))
        val_gts = [canon(lib.indices(e.tools)) for e in val_ex]

    model = HySet(len(lib), encoder.dim, model_cfg, tool_init=tool_emb)
    opt = torch.optim.Adam(
        [
            {"params": [model.z_raw], "lr": cfg.lr * cfg.z_lr_mult},
            {"params": [model.P, model.A], "lr": cfg.lr},
        ]
    )
    sampler = NegativeSampler(
        len(lib), cfg.k_neg, M, cfg.mix, n_neighbors=cfg.n_neighbors, seed=cfg.seed
    )
    feedback = (
        ExecutionFeedback(
            reward,
            train_ex,
            lib,
            n_sub=cfg.n_sub,
            refresh_steps=cfg.refresh_steps,
            max_calls=cfg.max_reward_calls,
            seed=cfg.seed,
        )
        if reward is not None
        else None
    )

    history: list[dict] = []
    best = {"score": float("-inf"), "epoch": -1, "step": 0, "val": None, "state": None}
    evals_since_best = 0
    n_steps_epoch = -(-len(train_ex) // cfg.batch_size)
    evals_per_epoch = 1 + (n_steps_epoch // cfg.eval_every if cfg.eval_every else 0)
    patience_evals = cfg.patience * evals_per_epoch

    def validate(epoch: int, step: int) -> dict | None:
        nonlocal evals_since_best
        if not val_ex:
            return None
        model.eval()
        rows, _ = evaluate_model(model, infer_cfg, r_val, val_gts)
        model.train()
        val = rows["hyset"]
        if val[cfg.val_metric] > best["score"]:
            best.update(score=val[cfg.val_metric], epoch=epoch, step=step, val=val,
                        state=copy.deepcopy(model.state_dict()))  # fmt: skip
            evals_since_best = 0
        else:
            evals_since_best += 1
        return val

    rng = np.random.default_rng(cfg.seed)
    step = 0
    stop = False
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        sampler.refresh_neighbors(model.Z.detach())
        perm = rng.permutation(len(train_ex))
        tot = {"loss": 0.0, "l_ret": 0.0, "l_self": 0.0, "reward": 0.0, "n_fb": 0}
        n_batches = 0
        for start in range(0, len(perm), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size].tolist()
            batch_gts = [gts[i] for i in idx]
            mined: list[list[ToolSet]] = [[] for _ in idx]
            if sampler.n_model:
                model.eval()
                miner = SetRetriever(model, infer_cfg)
                rb = r_train[idx]
                mined = miner.top_sets(rb, miner.shortlist(rb), sampler.n_model + 1)
                model.train()
            pools = [sampler.pool(gts[i], batch_gts, mined[b]) for b, i in enumerate(idx)]
            sets, valid = pools_to_tensor(pools, M)
            f = model.score_sets(r_train[idx], sets).masked_fill(~valid, float("-inf"))
            logp = F.log_softmax(f, dim=-1)
            l_ret = -logp[:, 0].mean()

            l_self = torch.zeros(())
            if feedback is not None and epoch >= cfg.self_train_start_epoch:
                feedback.on_step(step)
                picks = f.detach().argmax(-1).tolist()
                chosen = [pools[b][j] for b, j in enumerate(picks)]
                fb = feedback.rewards(idx, chosen)
                if fb:
                    rows = torch.tensor([row for row, _ in fb])
                    rho = torch.tensor([r for _, r in fb])
                    cols = torch.tensor([picks[row] for row, _ in fb])
                    l_self = -(rho * logp[rows, cols]).sum() / len(idx)
                    tot["reward"] += float(rho.sum())
                    tot["n_fb"] += len(fb)

            loss = l_ret + cfg.eta * l_self + cfg.lam * model.regularizer()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            n_batches += 1
            tot["loss"] += loss.item()
            tot["l_ret"] += l_ret.item()
            tot["l_self"] += l_self.item()

            if cfg.eval_every and step % cfg.eval_every == 0:
                val = validate(epoch, step)
                if val is not None:
                    log.info("  step %d val %s=%.4f comp@5=%.4f set_exact=%.4f", step,
                             cfg.val_metric, val[cfg.val_metric], val["comp@5"],
                             val["set_exact"])  # fmt: skip
                if val_ex and evals_since_best >= patience_evals:
                    stop = True
                    break

        rec = {
            "epoch": epoch,
            "step": step,
            "loss": tot["loss"] / max(n_batches, 1),
            "l_ret": tot["l_ret"] / max(n_batches, 1),
            "l_self": tot["l_self"] / max(n_batches, 1),
            "secs": round(time.time() - t0, 1),
        }
        if tot["n_fb"]:
            rec["mean_reward"] = tot["reward"] / tot["n_fb"]
            rec["reward_calls"] = feedback.calls if feedback else 0
        if not stop:
            val = validate(epoch, step)
            if val is not None:
                rec["val"] = val
        if on_epoch is not None:
            model.eval()
            rec.update(on_epoch(model, epoch))
            model.train()
        history.append(rec)
        val_txt = (
            f" val {cfg.val_metric}={rec['val'][cfg.val_metric]:.4f}"
            f" comp@5={rec['val']['comp@5']:.4f} set_exact={rec['val']['set_exact']:.4f}"
            if "val" in rec
            else ""
        )
        fb_txt = f" reward={rec['mean_reward']:.3f}" if "mean_reward" in rec else ""
        log.info(
            "epoch %d loss=%.4f l_ret=%.4f l_self=%.4f%s%s (%.1fs)",
            epoch, rec["loss"], rec["l_ret"], rec["l_self"], fb_txt, val_txt, rec["secs"],
        )  # fmt: skip
        if stop or (val_ex and evals_since_best >= patience_evals):
            log.info("early stop: no %s gain for %d evaluations", cfg.val_metric, evals_since_best)
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        log.info("restored best checkpoint: epoch %d step %d %s=%.4f", best["epoch"],
                 best["step"], cfg.val_metric, best["score"])  # fmt: skip
    else:
        best.update(epoch=len(history) - 1, step=step)
    model.eval()
    return TrainResult(model=model, history=history, best_epoch=best["epoch"],
                       tool_emb=tool_emb, best_step=best["step"], best_val=best["val"])  # fmt: skip
