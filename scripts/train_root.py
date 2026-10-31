#!/usr/bin/env python3
"""Train the validated root-feedback preset with explicit GPU binding."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def state_digest(module, base_dim=None):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        if base_dim is not None and key == "dec.0.weight":
            dimension = module.obs_norm.mean.numel()
            indexes = list(range(32)) + [32 + t * dimension + j for t in range(module.hist) for j in range(base_dim)]
            value = value[:, indexes]
        elif base_dim is not None and key in ("obs_norm.mean", "obs_norm.var"):
            value = value[:base_dim]
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="physical GPU index or full UUID")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--motion-dir", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--config", type=Path, help="custom native training configuration")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="native train.py overrides after --")
    args = parser.parse_args()
    config_path = args.config or ROOT / "configs" / ("baseline.json" if args.baseline else "root_feedback.json")
    config = json.loads(config_path.read_text())
    config.update(seed=args.seed, log_dir=str(args.log_dir or Path("logs") / f"{'baseline' if args.baseline else 'root'}_s{args.seed}"))
    if args.motion_dir:
        config["motion_dir"] = str(args.motion_dir)
    command = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                command.append("--" + key)
        else:
            command.extend(("--" + key, str(value)))
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    command.extend(extra)
    if args.dry_run:
        print(json.dumps(dict(module="instanttrack.train", gpu=args.gpu, arguments=command), indent=2))
        return
    output = Path(config["log_dir"])
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty --log-dir to avoid mixing runs")
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    pairs = [tuple(x.strip() for x in line.split(",")) for line in devices.splitlines()]
    matches = [(index, uuid) for index, uuid in pairs if args.gpu in (index, uuid)]
    if len(matches) != 1:
        parser.error(f"GPU {args.gpu!r} did not identify one device")
    physical_index, expected = matches[0]
    os.environ.update(CUDA_VISIBLE_DEVICES=expected, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                      TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE="0", TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC="0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import warp as wp
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible CUDA device")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    wp.init()
    if actual.removeprefix("GPU-") != expected.removeprefix("GPU-") or wp.get_cuda_device(0).uuid != expected:
        raise RuntimeError("Torch and Warp must both bind to the selected GPU UUID")

    holder = {}

    def capture_initial_state(frame, event, argument):
        if event != "call" or frame.f_code.co_name != "collect" or Path(frame.f_code.co_filename).resolve() != ROOT / "src/instanttrack/train.py":
            return
        sys.setprofile(None)
        g = frame.f_globals
        holder["globals"] = g
        native_args = g["args"]
        folder = Path(native_args.log_dir)
        json_write(folder / "initialization.json", dict(world_model=state_digest(g["wm"]),
            policy_common=state_digest(g["pi"], g["mdp"].base_obs_dim), critic=state_digest(g["critic"]),
            cpu_rng=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
            cuda_rng=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
            position_obs=native_args.position_obs, physical_gpu=int(physical_index), device_uuid=expected))
        iteration = g["start_it"]
        checkpoint = dict(wm=g["wm"].state_dict(),
            pi=g["pi_avg"].module.state_dict() if g["pi_avg"] is not None else g["pi"].state_dict(),
            pi_live=g["pi"].state_dict(), iteration=iteration, wall_time=g["wall_offset"],
            sim_steps=g["sim_steps"], args=vars(native_args), saved_at="initial state")
        target = folder / f"ckpt_{iteration:06d}.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(target)

    sys.argv = ["instanttrack.train"] + command
    sys.setprofile(capture_initial_state)
    try:
        runpy.run_module("instanttrack.train", run_name="__main__")
    finally:
        sys.setprofile(None)
    g = holder["globals"]
    g["log_f"].flush()
    completed = int(g["it"]) + 1
    json_write(Path(g["args"].log_dir) / "training_result.json", dict(iterations=completed,
        target_iterations=g["args"].iterations, completed_target=completed == g["args"].iterations,
        actor_updates=max(0, completed - g["args"].wm_only_iters),
        wm_gradient_steps=completed * g["args"].wm_batches, real_env_transitions=g["sim_steps"],
        position_obs=g["args"].position_obs))


if __name__ == "__main__":
    main()
