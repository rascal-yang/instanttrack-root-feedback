"""Construct policies from checkpoint metadata, including legacy observations."""
from __future__ import annotations

from collections.abc import Mapping
import torch

from .networks import Policy


def position_mode(args: Mapping) -> str:
    mode = args.get("position_obs", "none")
    if mode not in ("none", "root", "torso"):
        raise ValueError(f"Unsupported checkpoint position_obs: {mode!r}")
    return mode


def observation_dimension(num_joints: int, mode: str) -> int:
    if mode not in ("none", "root", "torso"):
        raise ValueError(f"Unsupported observation mode: {mode!r}")
    return 3 * num_joints + 12 + (0 if mode == "none" else 3)


def policy_from_checkpoint(checkpoint: Mapping, goal_dim: int, num_joints: int,
                           device: str | torch.device = "cpu") -> Policy:
    args = checkpoint["args"]
    mode = position_mode(args)
    dim = observation_dimension(num_joints, mode)
    stored_dim = checkpoint["pi"]["obs_norm.mean"].numel()
    if stored_dim != dim:
        raise ValueError(f"Checkpoint declares position_obs={mode!r} ({dim} inputs), "
                         f"but stores {stored_dim} normalized observations")

    def widths(key):
        value = args.get(key, "2048,1024,512,256")
        return tuple(map(int, value.split(","))) if isinstance(value, str) else tuple(value)

    policy = Policy(dim, goal_dim, num_joints, hist=args["hist"],
                    enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"),
                    base_obs_dim=observation_dimension(num_joints, "none")).to(device)
    policy.load_state_dict(checkpoint["pi"])
    return policy.eval()
